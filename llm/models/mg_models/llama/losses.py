import torch
import torch.nn.functional as F

from llm.models.mg_models.base_modules.functions.cross_entropy import vocab_parallel_cross_entropy
from llm.utils.env import dist_env
from llm.data.nlp_dataset import IGNORE_INDEX
from llm.utils.general.registry_factory import LOSS_REGISTRY
from llm.utils.general.log_helper import default_logger as logger


@LOSS_REGISTRY.register('softmax_cross_entropy')
class CrossEntropy(object):
    def __init__(self, loss_on_targets_only=False, reweight_loss_based_on_position_frequency=False,
                 is_prefix=True, cut_size=None, **kwargs):
        self.loss_on_targets_only = loss_on_targets_only
        self.reweight_loss_based_on_position_frequency = reweight_loss_based_on_position_frequency
        self.is_prefix = is_prefix
        self.cut_size = cut_size

    def get_expected_number_of_tokens(self, labels, loss_mask):
        ignore_mask = (labels == IGNORE_INDEX)
        loss_mask = loss_mask.view(-1)
        loss_mask = loss_mask * (~ignore_mask.view(-1))
        if self.is_prefix:
            micro_batch_size, sequence_length = labels.shape
            average_tokens_per_sample: torch.Tensor
            if self.loss_on_targets_only:
                # HACK: This is useful when we obtain loss masks that are microbatch dependent.
                #   Consequently, if we want to preserve the notion that all tokens have the same
                #   impact on the loss, we can only normalise using a microbatch independent value.
                #   It should be expected weight over a microbatch. Here we still use `sequence_length`,
                #   that's batch size dependent, in order to be backwards compatible with
                #   current experiment on vanilla gpt.
                if self.reweight_loss_based_on_position_frequency:
                    reweight = torch.arange(
                        sequence_length, 0, -1, dtype=torch.float, device=loss_mask.device
                    ) / (sequence_length + 1) * 2
                    average_tokens_per_sample = reweight.flip(-1).cumsum(-1).mean()
                else:
                    average_tokens_per_sample = (sequence_length + 1) / 2
            else:
                average_tokens_per_sample = sequence_length
            expected_number_of_tokens = average_tokens_per_sample * micro_batch_size
        else:
            expected_number_of_tokens = loss_mask.sum()
        return expected_number_of_tokens, loss_mask

    def __call__(self, output, labels):
        labels, loss_mask = labels[0], labels[1]

        losses = vocab_parallel_cross_entropy(output.contiguous().float(),
                                              labels, self.cut_size)
        expected_number_of_tokens, loss_mask = self.get_expected_number_of_tokens(labels, loss_mask)

        loss = torch.sum(losses.view(-1) * loss_mask) / expected_number_of_tokens
        return loss


@LOSS_REGISTRY.register('reward_sigmoid_entropy')
class RewardSigmoidCrossEntropy(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, output, labels):
        input_ids = labels[0]
        rewards = output[..., 0]
        from llm.data.nlp_dataset import IGNORE_INDEX
        if dist_env.get_tensor_model_parallel_rank() == 0:
            assert len(input_ids.shape) == 2
            bs = input_ids.shape[0] // 2
            seq_len = input_ids.shape[1]

            chosen_ids = input_ids[:bs]  # bs x seq x 1
            rejected_ids = input_ids[bs:]
            chosen_rewards = rewards[:bs]
            rejected_rewards = rewards[bs:]

            # Compute pairwise loss. Only backprop on the different tokens before padding
            loss = 0
            for i in range(bs):
                chosen_id = chosen_ids[i]
                rejected_id = rejected_ids[i]
                chosen_reward = chosen_rewards[i]
                rejected_reward = rejected_rewards[i]

                c_inds = (chosen_id == IGNORE_INDEX).nonzero()
                c_ind = c_inds[0].item() if len(
                    c_inds
                ) > 0 else seq_len
                check_divergence = (chosen_id != rejected_id).nonzero()

                if len(check_divergence) == 0:
                    end_ind = rejected_reward.size(-1)
                    divergence_ind = end_ind - 1
                    r_ind = c_ind
                else:
                    # Check if there is any padding otherwise take length of sequence
                    r_inds = (rejected_id == IGNORE_INDEX).nonzero()
                    r_ind = r_inds[0].item(
                    ) if len(r_inds) > 0 else seq_len
                    end_ind = max(c_ind, r_ind)
                    divergence_ind = check_divergence[0]
                # if not (divergence_ind > 0):
                #     import ipdb; ipdb.set_trace()
                # assert divergence_ind > 0, print(divergence_ind)
                c_truncated_reward = chosen_reward[divergence_ind:end_ind]
                r_truncated_reward = rejected_reward[divergence_ind:end_ind]
                # chosen_mean_scores.append(
                #     chosen_reward[c_ind - 1])  #use the end score for reference
                # rejected_mean_scores.append(rejected_reward[r_ind - 1])
                loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()
        else:
            loss = output.mean() * 0
        torch.distributed.all_reduce(loss,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=dist_env.get_tensor_model_parallel_group())
        return loss


@LOSS_REGISTRY.register('mini_rlhf_entropy')
class MiniRLHFCrossEntropy(CrossEntropy):
    def __init__(self, rlhf_weight=1.0, length_penalty=1.0, sentense_pairs=6, **kwargs):
        self.rlhf_weight = rlhf_weight
        self.length_penalty = length_penalty
        self.sentense_pairs = sentense_pairs

    def gather_logits_labels(self, logits, labels):
        mask = (labels != IGNORE_INDEX).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == IGNORE_INDEX] = 0
        output = torch.gather(new_logits.reshape(-1, logits.shape[-1]),
                              dim=-1, index=labels.reshape(-1, 1)).reshape(mask.shape)
        output = output * mask      # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != IGNORE_INDEX).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.length_penalty)
        return scores

    def rlhf_loss(self, scores, rw_scores):
        diff = scores.unsqueeze(-2) - scores.unsqueeze(-1)       # b * b
        rw_diff = rw_scores.unsqueeze(-2) - rw_scores.unsqueeze(-1)      # b * b
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)         # [0]
        return -diff[aval].sum()

    def sft_loss(self, logit_label, rw_scores, labels):
        mask = (labels != IGNORE_INDEX).float()
        max_idx = torch.argmax(rw_scores, dim=-1)
        pick_pos = max_idx.new_zeros(rw_scores.shape)
        pick_pos.scatter_(1, max_idx.unsqueeze(-1), src=max_idx.new_ones((max_idx.shape[0], 1)))
        pick_pos = pick_pos.reshape(-1).bool()
        loss_mask = mask[pick_pos].reshape(-1)
        logit_label = logit_label[pick_pos].reshape(-1)
        expected_number_of_tokens = loss_mask.sum()
        loss = -torch.sum(logit_label * loss_mask) / expected_number_of_tokens
        return loss

    def __call__(self, output, labels):
        labels, rw_scores = labels[0], labels[2]

        actor_prob = dist_env.gather_from_tensor_model_parallel_region(output)

        logits = F.log_softmax(actor_prob, dim=-1)

        logit_label = self.gather_logits_labels(logits, labels.clone())
        scores = self.get_score(logit_label, labels.clone())

        scores = scores.reshape(-1, self.sentense_pairs)
        rw_scores = rw_scores.reshape(-1, self.sentense_pairs)

        rlhf_loss = self.rlhf_loss(scores, rw_scores)
        sft_loss = self.sft_loss(logit_label, rw_scores, labels)
        loss = self.rlhf_weight * rlhf_loss + sft_loss
        return loss


@LOSS_REGISTRY.register('dpo_loss')
class DPOLoss(CrossEntropy):
    def __init__(self, rlhf_weight=1.0, length_penalty=1.0, sentense_pairs=2, beta=0.1, loss_fn='sigmoid', reference_free=False, **kwargs):
        self.rlhf_weight = rlhf_weight
        self.length_penalty = length_penalty
        self.sentense_pairs = sentense_pairs
        self.ignore_idx = -100
        self.loss_fn = loss_fn
        self.beta = beta
        self.reference_free = reference_free
        logger.warning_once("DPOLoss only support micro_batch_size set to 1 now.")
        if sentense_pairs != 2:
            logger.warning_once("DPOLoss only support sentense_pairs set to 2 now.")

        super().__init__(**kwargs)  # support sft

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized).
                - Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored.
                - Shape: (batch_size, sequence_length)
            average_log_prob:
                - If True, return the average log probability per (non-masked) token.
                - Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        loss_mask = labels != self.ignore_idx

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.ignore_idx] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _sft_loss(self, output, labels, loss_mask):
        losses = vocab_parallel_cross_entropy(output.contiguous().float(),
                                              labels, self.cut_size)
        expected_number_of_tokens, loss_mask = self.get_expected_number_of_tokens(labels, loss_mask)

        loss = torch.sum(losses.view(-1) * loss_mask) / expected_number_of_tokens
        return loss

    def _loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if self.reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if self.loss_fn == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits).sum()
        elif self.loss_fn == "hinge":
            losses = torch.relu(1 - self.beta * logits).sum()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_fn}. Should be one of ['sigmoid', 'hinge']")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def __call__(self, output, labels):
        labels, loss_mask, ref_logps = labels[0], labels[1], labels[2]

        if ref_logps[0] > 1.:
            assert len(ref_logps) == 2  # only support micro_bs=1
            sft_loss = self._sft_loss(output, labels, loss_mask)
            return sft_loss

        output = dist_env.gather_from_tensor_model_parallel_region(output)

        logps = self._get_batch_logps(output, labels)
        logps = logps.reshape(-1, self.sentense_pairs)

        ref_logps = ref_logps.reshape(-1, self.sentense_pairs)

        plcy_yw_logps, plcy_yl_logps = logps[:, 0], logps[:, 1]
        ref_yw_logps, ref_yl_logps = ref_logps[:, 0], ref_logps[:, 1]

        loss, chosen_rewards, rejected_rewards = self._loss(plcy_yw_logps, plcy_yl_logps, ref_yw_logps, ref_yl_logps)
        return loss
