import torch
import torch.nn.functional as F

from llm.utils.env import dist_env


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits.new_ones(logits.shape).bool()
        top_k_logits = logits.topk(top_k, -1, True, True)[1]
        indices_to_remove.scatter_(1, top_k_logits, False)
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def pad_batch(batch, pad_id, context_lengths, out_seq_length=0):
    for i, tokens in enumerate(batch):
        tokens.extend([pad_id] * (out_seq_length + context_lengths[i].item() - len(tokens)))
    return batch


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_step(model, tokens):
    output_tensor = model.eval_batch(iter([tokens]), compute_loss=False,
                                     reduce_output=None, dynamic=True,
                                     sequence_parallel=model.sequence_parallel)
    if dist_env.is_pipeline_last_stage():
        return dist_env.gather_from_tensor_model_parallel_region(output_tensor[0])
    return output_tensor


def sample_sequence_batch(args, model, tokenizer, context_tokens,
                          context_lengths, maxlen=None, force_eos_id=None):
    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eos_id = tokenizer.eos_token_id if force_eos_id is None else force_eos_id

        counter = 0
        org_context_length = context_length

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        if maxlen is None:
            maxlen = org_context_length + args.out_seq_length - 1

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length <= (maxlen):
            assert args.recompute is True, 'Current Deepspeed mode do not support reuse tokens and positions'
            output = forward_step(model, tokens)
            if dist_env.is_pipeline_last_stage():
                assert output is not None
                logits = output[:, context_length - 1, :]

            if dist_env.is_pipeline_last_stage():
                logits = logits.float()
                logits /= args.temperature
                logits = top_k_logits(logits, top_k=args.top_k,
                                      top_p=args.top_p)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)

                started = context_lengths <= context_length

                new_tokens = switch(
                    tokens[:, context_length].view(-1), prev, started)
                tokens[:, context_length] = new_tokens
                src = dist_env.get_pipeline_model_parallel_last_rank()
                group = dist_env.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                done_token = (prev == eos_id).byte() & started.byte()
                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = dist_env.get_pipeline_model_parallel_last_rank()
                group = dist_env.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                yield tokens, lengths
            else:
                if dist_env.is_pipeline_first_stage():
                    src = dist_env.get_pipeline_model_parallel_last_rank()
                    group = dist_env.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None
                else:
                    yield None, None

                done = torch.cuda.ByteTensor([0])
                src = dist_env.get_pipeline_model_parallel_last_rank()
                group = dist_env.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break


def get_token_stream(args, model, tokenizer, context_tokens, force_eos_id=None):
    context_lengths = [len(tks) for tks in context_tokens]

    torch.distributed.barrier()
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    torch.distributed.broadcast(context_length_tensor,
                                dist_env.get_tensor_model_parallel_src_rank(),
                                group=dist_env.get_tensor_model_parallel_group())
    torch.distributed.barrier()

    context_tokens = pad_batch(context_tokens, tokenizer.eos_token_id,
                               context_length_tensor, args.out_seq_length)

    torch.distributed.barrier()
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    torch.distributed.broadcast(context_tokens_tensor,
                                dist_env.get_tensor_model_parallel_src_rank(),
                                group=dist_env.get_tensor_model_parallel_group())
    torch.distributed.barrier()

    context_length = context_length_tensor.min().item()
    # tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    batch_token_iterator = sample_sequence_batch(args, model, tokenizer,
                                                 context_tokens_tensor,
                                                 context_length_tensor,
                                                 force_eos_id=force_eos_id)
    for tokens, lengths in batch_token_iterator:
        context_length += 1
        if tokens is not None:
            yield tokens[:, :context_length], lengths
        else:
            yield None, None
