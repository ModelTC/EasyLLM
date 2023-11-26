import torch

from llm.utils.env import dist_env
from llm.utils.general.log_helper import default_logger as logger
from llm.data.nlp_transforms import build_augmentation, SenseTokenization
from .text_generation_utils import get_token_stream


def generate_samples_interactive(args, config, model, tokenizer, force_eos_id=None):
    history_metas = []
    system_flag = False
    assert 'infer_tokenization' in config, 'inference mode must provide a inference tokenizaition (e.g. sense_tokenization)'        # noqa
    config['infer_tokenization']['kwargs'].update({'tokenizer': tokenizer})
    infer_tokenization = build_augmentation(config['infer_tokenization'])
    infer_tokenization.parser.inference_mode = True
    while True:
        terminate_runs = 0
        continue_runs = 0
        model._compute_loss = False
        model.fwd_outputs = []
        model.total_loss = None
        if dist_env.get_tensor_model_parallel_rank() == 0 and dist_env.is_pipeline_first_stage():
            logger.info("Input your questions (quit to exit) >>> ")
            raw_text = input()
            input_meta = {}
            if isinstance(infer_tokenization, SenseTokenization) and system_flag:
                input_meta['content'] = raw_text
                input_meta['role'] = "system"
                history_metas.append(input_meta)
                system_flag = False
                continue_runs = 1
            if len(raw_text.strip()) == 0:
                logger.info('Input questions must not be empty!')
                continue_runs = 1
            if raw_text == 'quit':
                logger.info('Input quit, quit process!')
                terminate_runs = 1
            if raw_text == 'system':
                system_flag = True
                continue_runs = 1
            if raw_text == 'clean':
                logger.info('Clean dialog, the conversation will restart from scratch...')
                history_metas = []
                system_flag = False
                continue_runs = 1

            ori_raw_text = raw_text
            if hasattr(infer_tokenization.parser, 'build_inference_meta'):
                if not system_flag:
                    meta = infer_tokenization.parser.build_inference_meta(raw_text, history_metas)
                    context_tokens, _ = infer_tokenization(meta)
                else:
                    context_tokens = tokenizer.encode("EMPTY TEXT")
            else:
                context_tokens, _ = infer_tokenization({"text": ori_raw_text, "dialog_history": history_metas})
            context_length = len(context_tokens)
        else:
            # the context_tokens will be sycned in get_token_stream
            # therefore, here we set a random inout text
            context_tokens = tokenizer.encode("EMPTY TEXT")
            context_length = 0

        # Note that since all other context_length is 0, all_reduce is equal to boardcast.
        torch.distributed.barrier()
        input_info_tensor = torch.cuda.LongTensor([terminate_runs, context_length, continue_runs])
        torch.distributed.all_reduce(input_info_tensor,
                                     group=dist_env.get_model_parallel_group(),
                                     op=torch.distributed.ReduceOp.SUM)
        terminate_runs = input_info_tensor[0].item()
        context_length = input_info_tensor[1].item()
        continue_runs = input_info_tensor[2].item()

        if terminate_runs == 1:
            return
        if continue_runs == 1:
            continue
        # For pipeline parallel we send context tokens to other stages
        # so they get the lengths correct
        torch.distributed.barrier()
        if dist_env.get_tensor_model_parallel_rank() == 0 and dist_env.get_pipeline_model_parallel_world_size() > 1:
            if dist_env.is_pipeline_first_stage():
                src = dist_env.get_pipeline_model_parallel_first_rank()
                group = dist_env.get_pipeline_model_parallel_group()
                context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
                torch.distributed.broadcast(context_tokens_tensor, src, group)
            else:
                src = dist_env.get_pipeline_model_parallel_first_rank()
                group = dist_env.get_pipeline_model_parallel_group()
                context_tokens_tensor = torch.empty(context_length,
                                                    dtype=torch.int64,
                                                    device=torch.device("cuda"))
                torch.distributed.broadcast(context_tokens_tensor, src, group)
                context_tokens = context_tokens_tensor.cpu().numpy().tolist()
        torch.distributed.barrier()

        token_stream = get_token_stream(args, model, tokenizer, [context_tokens], force_eos_id=force_eos_id)

        for counter, decode_tokens in enumerate(token_stream):
            if counter != 0 and args.generate_log_frequency != -1\
                    and counter % args.generate_log_frequency == 0 \
                    and dist_env.get_tensor_model_parallel_rank() == 0 \
                    and dist_env.is_pipeline_first_stage():
                logger.info('Be patient for answers, {} tokens have been generated.'.format(counter))

        if dist_env.is_pipeline_first_stage() \
                and dist_env.get_tensor_model_parallel_rank() == 0:
            logger.info("\nContext: {}".format(ori_raw_text))

            if not isinstance(decode_tokens, list):
                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()
            trim_decode_tokens = tokenizer.decode(decode_tokens[context_length:], skip_special_tokens=True)
            logger.info("\nEasyLLM: {}".format(trim_decode_tokens))

            input_meta['content'] = ori_raw_text
            input_meta['role'] = 'user'
            history_metas.append(input_meta)
            out_meta = {}
            out_meta['content'] = trim_decode_tokens
            out_meta['role'] = 'assistant'
            history_metas.append(out_meta)


def generate_samples_eval(args, context_tokens, model, tokenizer, force_eos_id=None):

    terminate_runs = 0
    continue_runs = 0
    model._compute_loss = False
    model.fwd_outputs = []
    model.total_loss = None
    if dist_env.get_tensor_model_parallel_rank() == 0 and dist_env.is_pipeline_first_stage():
        # context_tokens = tokenizer.encode(raw_text)
        context_length = len(context_tokens)
    else:
        context_tokens = tokenizer.encode("EMPTY TEXT")
        context_length = 0

    torch.distributed.barrier()
    input_info_tensor = torch.cuda.LongTensor([terminate_runs, context_length, continue_runs])
    torch.distributed.all_reduce(input_info_tensor,
                                 group=dist_env.get_model_parallel_group(),
                                 op=torch.distributed.ReduceOp.SUM)
    terminate_runs = input_info_tensor[0].item()
    context_length = input_info_tensor[1].item()
    continue_runs = input_info_tensor[2].item()

    if terminate_runs == 1:
        return

    torch.distributed.barrier()
    if dist_env.get_tensor_model_parallel_rank() == 0 and dist_env.get_pipeline_model_parallel_world_size() > 1:
        if dist_env.is_pipeline_first_stage():
            src = dist_env.get_pipeline_model_parallel_first_rank()
            group = dist_env.get_pipeline_model_parallel_group()
            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            torch.distributed.broadcast(context_tokens_tensor, src, group)
        else:
            src = dist_env.get_pipeline_model_parallel_first_rank()
            group = dist_env.get_pipeline_model_parallel_group()
            context_tokens_tensor = torch.empty(context_length,
                                                dtype=torch.int64,
                                                device=torch.device("cuda"))
            torch.distributed.broadcast(context_tokens_tensor, src, group)
            context_tokens = context_tokens_tensor.cpu().numpy().tolist()
    torch.distributed.barrier()

    token_stream = get_token_stream(args, model, tokenizer, [context_tokens], force_eos_id=force_eos_id)

    for counter, decode_tokens in enumerate(token_stream):
        if counter != 0 and args.generate_log_frequency != -1\
                and counter % args.generate_log_frequency == 0 \
                and dist_env.get_tensor_model_parallel_rank() == 0 \
                and dist_env.is_pipeline_first_stage():
            logger.info('Be patient for answers, {} tokens have been generated.'.format(counter))

    if dist_env.is_pipeline_first_stage() \
            and dist_env.get_tensor_model_parallel_rank() == 0:
        if not isinstance(decode_tokens, list):
            decode_tokens, _ = decode_tokens
            decode_tokens = decode_tokens[0].cpu().numpy().tolist()
        trim_decode_tokens = tokenizer.decode(decode_tokens[context_length:])
        return trim_decode_tokens
