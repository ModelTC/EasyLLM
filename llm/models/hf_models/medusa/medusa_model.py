import torch
import torch.nn as nn
from transformers import PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import mc_sim_7b_3_head_top5
import os
from torch.nn import CrossEntropyLoss


IGNORE_TOKEN_ID = -100


class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModel(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        base_model,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        tokenizer=None
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
            medusa_num_layers (int, optional): Number of ResBlock layers for each Medusa head. Defaults to 0.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = tokenizer
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )

        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_head.to(self.base_model.dtype).to(self.base_model.device)

        for i in range(medusa_num_heads):
            # Initialize the weights of each medusa_head using the base model's weights
            self.medusa_head[i][-1].weight.data[:] = base_model.lm_head.weight.data[:]

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        medusa_head_name_or_path,
        base_model=None,
        medusa_num_heads=None,
        medusa_num_layers=None,
        base_model_name_or_path=None,
        **kwargs,
    ):
        """
        Args:
            medusa_head_name_or_path (str): Name or path of the Medusa head to load.
            **kwargs: Additional keyword arguments for loading the base model.
        Returns:
            MedusaModel: A MedusaModel instance loaded from the given path.
        """
        model = cls(
            base_model,
            medusa_num_heads,
            medusa_num_layers,
            base_model_name_or_path,
        )
        if medusa_head_name_or_path is None:
            return model
        medusa_head_path = os.path.join(medusa_head_name_or_path, "medusa_lm_head.pth")
        if os.path.exists(medusa_head_path):
            filename = medusa_head_path
            medusa_head_state_dict = torch.load(filename, map_location=base_model.device)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=True)
            return model
        else:
            return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        return_dict=None,
        use_cache=False,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))

        if output_orig:
            logits = torch.stack(medusa_logits, dim=0), outputs, orig
        else:
            logits = torch.stack(medusa_logits, dim=0)
        if self.training and labels is not None:
            return self.compute_loss(logits, labels, return_dict)
        else:
            return logits

    def compute_loss(self, logits, labels, return_dict=False):
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        output = {}
        for i in range(self.medusa):
            medusa_logits = logits[i, :, : -(2 + i)].contiguous()
            medusa_labels = labels[..., 2 + i:].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            # loss += loss_i
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 6, 2):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                output[f"medusa{i}_top{k}.accuracy"] = correct.float().mean()

            output[f"medusa{i}.loss"] = loss_i
        from easydict import EasyDict
        output = EasyDict(output)
        if return_dict:
            return output
        else:
            return loss

    def medusa_generate(
        self,
        input_ids,
        medusa_generate=True,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next
        # token, top-6 predictions for the next next token.
        medusa_choices=mc_sim_7b_3_head_top5,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        generate_mode="interactive"
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            true_medusa_choices = []
            for mc in medusa_choices:
                if len(mc) <= self.medusa:
                    true_medusa_choices.append(mc)
            medusa_choices = true_medusa_choices

            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        if medusa_generate:
            # Initialize tree attention mask and process prefill tokens
            medusa_logits, logits = initialize_medusa(
                input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
            )
        else:
            medusa_logits, logits = initialize_medusa(
                input_ids, self, None, past_key_values
            )

        new_token = 0
        ave_accept_length = 0
        count = 0
        out_tokens = [0, 0, 0, 0]
        for idx in range(max_steps):
            if medusa_generate:
                # Generate candidates with topk predictions from Medusa heads
                candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                )

                # Use tree attention to verify the candidates and get predictions
                medusa_logits, logits, outputs = tree_decoding(
                    self,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                )

                # Evaluate the posterior of the candidates to select the accepted candidate prefix
                best_candidate, accept_length = evaluate_posterior(
                    logits, candidates, temperature, posterior_threshold, posterior_alpha
                )
                out_tokens[accept_length.item()] += 1
                ave_accept_length = ave_accept_length + accept_length + 1
                if generate_mode == "interactive":
                    print("accept_length: ", accept_length)
                # Update the input_ids and logits
                input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    medusa_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    medusa_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                )
            else:
                candidates = torch.argmax(logits[:, -1]).unsqueeze(0)[None, :]
                position_ids = torch.zeros(1).type_as(input_ids) + input_ids.shape[1]

                # Use the model to decode the tree candidates.
                # The model is expected to return logits for the Medusa structure,
                # original logits, and possibly other outputs.
                medusa_logits, outputs, logits = self(
                    candidates,
                    output_orig=True,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                input_ids = torch.cat([input_ids, candidates], dim=-1)
            count += 1
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                for idx in range(input_ids.shape[-1]):
                    if input_ids[:, -idx - 1] == self.tokenizer.eos_token_id:
                        flag_eos = idx
                        break
                if flag_eos > 0:
                    input_ids = input_ids[0, :-flag_eos].unsqueeze(0)
                    ave_accept_length -= flag_eos
                break
        if generate_mode == "interactive":
            print("ave accept_length: ", ave_accept_length / float(count))
        count = count.item() if isinstance(count, torch.Tensor) else count
        ave_accept_length = ave_accept_length.item() if isinstance(ave_accept_length, torch.Tensor) else ave_accept_length
        text_output = self.tokenizer.decode(
            input_ids[0, input_len:],
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        infos = {
            "count": count,
            "accept_length": ave_accept_length,
            "ave_accept_length": ave_accept_length / float(count),
            "out_token_one": out_tokens[0],
            "out_token_two": out_tokens[1],
            "out_token_three": out_tokens[2],
            "out_token_four": out_tokens[3]
        }
        return text_output, infos
