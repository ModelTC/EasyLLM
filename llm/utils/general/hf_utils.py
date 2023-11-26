import os
import json
import random
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.utils.general.log_helper import default_logger as logger
from llm.utils.env.hf_dist_helper import get_world_size, get_rank, dist_barrier, all_gather
from tools.utils.prompt import text_postprocess, save_results, evaluate
from tools.utils.dataset import EvalDataset, SampleEvalDataset


def get_ceph_path(path):
    ceph_bucket = os.environ.get('CEPHBUCKET')
    if ceph_bucket != '':
        if ceph_bucket[-1] != '/':
            ceph_bucket += '/'
        # remove /
        if path[0] == '/':
            path = path[1:]
        # remove ./
        if path[:2] == './':
            path = path[2:]
        ceph_path = ceph_bucket + path
    else:
        ceph_path = path
    return ceph_path


def ceph_save(state_dict, path):
    from petrel_helper import PetrelHelper
    ceph_path = get_ceph_path(path)
    with open(path + '.ceph', 'w') as f:
        print(ceph_path, file=f)
    PetrelHelper.save(state_dict, ceph_path)


def save_hf_checkpoint(runner, save_cfg, global_step, state_dict=None):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    WEIGHTS_NAME = "pytorch_model.bin"
    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{global_step}"
    run_dir = save_cfg.get('save_path', "checkpoints")

    output_dir = os.path.join(run_dir, checkpoint_folder)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {output_dir}")
    if "CEPHBUCKET" in os.environ and os.environ.get("CEPHBUCKET") is not None:
        save_function = ceph_save
    else:
        save_function = torch.save
    if isinstance(runner.model, DDP):
        runner.model.module.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=False, save_function=save_function
        )
    else:
        runner.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=False, save_function=save_function
        )
    logger.info("Saving model state dict done.")

    if runner.tokenizer is not None:
        try:
            if hasattr(runner.tokenizer, "tokenizer"):
                runner.tokenizer.tokenizer.save_pretrained(output_dir)
            else:
                runner.tokenizer.save_pretrained(output_dir)
            logger.info("Saving tokenizer done.")
        except Exception:
            logger.warning("Failed to saving tokenizer done!!!")

    if os.environ.get("CEPHBUCKET", None) is not None:
        all_files = os.listdir(output_dir)
        for file_path in all_files:
            if file_path.endswith('.' + WEIGHTS_NAME.split('.')[-1]):
                continue
            local_path = os.path.join(output_dir, file_path)
            if os.path.isdir(local_path):
                continue
            ceph_file_path = get_ceph_path(local_path)
            from petrel_helper import PetrelHelper
            with open(local_path, 'rb') as f:
                PetrelHelper.write(f, ceph_file_path)


def save_ds_checkpoints(runner, save_cfg, global_step):
    output_dir = save_cfg.get('save_path', "checkpoints")
    checkpoint_folder = f"checkpoint-{global_step}"
    output_dir = os.path.join(output_dir, checkpoint_folder)
    os.makedirs(output_dir, exist_ok=True)
    tag = f"global_step{global_step}"
    state_dict = {}
    state_dict['iteration'] = global_step
    if save_cfg.get('save_rng_state', False):
        state_dict['random_rng_state'] = random.getstate()
        state_dict['np_rng_state'] = np.random.get_state()
        state_dict['torch_rng_state'] = torch.get_rng_state()
        state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
    runner.model.save_checkpoint(output_dir, tag=tag, client_state=state_dict)


def load_sharded_checkpoint(runner, folder):
    import gc
    WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    load_index = index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)
    shard_files = list(set(index["weight_map"].values()))

    loaded_keys = index["weight_map"].keys()
    model_keys = runner.model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    loader = partial(torch.load, map_location="cpu")
    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        runner.model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)


def load_from_ds(runner, load_cfg):
    resume_from_checkpoint = load_cfg['load_path']
    deepspeed_checkpoint_dirs = []
    if resume_from_checkpoint is not None:
        import glob
        deepspeed_checkpoint_dirs = sorted(glob.glob(f"{resume_from_checkpoint}/global_step*"))
        if len(deepspeed_checkpoint_dirs) <= 0:
            deepspeed_checkpoint_dirs = sorted(glob.glob(f"{resume_from_checkpoint}/global-latest"))
    logger.info(f"Resuming deepspeed weights from {resume_from_checkpoint}")
    load_optim = load_cfg.get('load_optim', False)
    if len(deepspeed_checkpoint_dirs) > 0:
        # this magically updates self.optimizer and self.lr_scheduler
        load_path, state_dict = runner.model.load_checkpoint(
            resume_from_checkpoint, load_optimizer_states=load_optim, load_lr_scheduler_states=load_optim
        )
        runner.start_iter = state_dict['iteration']
        if load_path is None:
            raise ValueError(f"[deepspeed] failed to resume from checkpoint {resume_from_checkpoint}")
    else:
        logger.info(f"[deepspeed] Can't find checkpoint from checkpoint {resume_from_checkpoint}")


def load_from_hf(runner, load_cfg):
    load_dir = load_cfg['load_path']
    WEIGHTS_NAME = "pytorch_model.bin"
    OPTIMIZER_NAME = "optimizer.pt"
    SCHEDULER_NAME = "scheduler.pt"
    SCALER_NAME = "scaler.pt"
    weights_file = os.path.join(load_dir, WEIGHTS_NAME)
    if os.path.isfile(weights_file):
        state_dict = torch.load(weights_file, map_location="cpu")
        runner.model.load_state_dict(state_dict, False)
        del state_dict
    else:
        runner.load_sharded_checkpoint(load_dir)
    logger.info("Loading checkpoint done.")
    if load_cfg.get('load_optim', False):
        # load trainer
        checkpoint_file_exists = os.path.isfile(os.path.join(load_dir, OPTIMIZER_NAME))
        if checkpoint_file_exists and os.path.isfile(os.path.join(load_dir, SCHEDULER_NAME)):
            map_location = "cuda" if get_world_size() > 1 else "cpu"
            runner.optimizer.load_state_dict(
                torch.load(os.path.join(load_dir, OPTIMIZER_NAME), map_location=map_location)
            )
            logger.info("Loading optimizer done.")
            runner.lr_scheduler.load_state_dict(torch.load(os.path.join(load_dir, SCHEDULER_NAME)))
            logger.info("Loading lr_scheduler done.")
            runner.scaler.load_state_dict(torch.load(os.path.join(load_dir, SCALER_NAME)))
            logger.info("Loading scaler done.")
    if load_cfg.get('load_rng_state', False):
        # load rng
        if get_world_size() > 1:
            rng_file = os.path.join(load_dir, f"rng_state_{get_rank()}.pth")
        else:
            rng_file = os.path.join(load_dir, "rng_state.pth")
        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if get_world_size() > 1:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
        logger.info("Loading rng_state done.")


def hf_inference(config, model, sense_tokenization, device, args):
    generation_cfg = config["generation_cfg"]
    tokenizer = sense_tokenization.parser.tokenizer
    pad_token_id = len(tokenizer) - 1
    history_metas = []
    with torch.no_grad():
        if args.generate_mode == "interactive":
            system_flag = False
            while True:
                logger.info("请输入问题，退出请输入 quit")
                raw_input_text = input()
                input_meta = {}
                if system_flag:
                    input_meta['content'] = raw_input_text
                    input_meta['role'] = "system"
                    history_metas.append(input_meta)
                    system_flag = False
                    continue

                if len(raw_input_text.strip()) == 0:
                    break
                if raw_input_text == 'quit':
                    break
                if raw_input_text == 'system':
                    system_flag = True
                    continue
                if raw_input_text == "clean":
                    history_metas = []
                    continue

                if hasattr(sense_tokenization.parser, 'build_inference_meta'):
                    prompt = sense_tokenization.parser.build_inference_meta(raw_input_text, history_metas)
                    context_tokens, _ = sense_tokenization(prompt)
                else:
                    context_tokens, _ = sense_tokenization({"text": raw_input_text, "dialog_history": history_metas})
                context_tokens = torch.LongTensor([context_tokens])
                attention_mask = context_tokens.ne(pad_token_id)

                generation_output = model.generate(
                    input_ids=context_tokens.to(device),
                    attention_mask=attention_mask.to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_cfg
                )
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                logger.info(f"SenseChat: {output}")

                input_meta['content'] = raw_input_text
                input_meta['role'] = 'user'
                history_metas.append(input_meta)
                out_meta = {}
                out_meta['content'] = output
                out_meta['role'] = 'assistant'
                history_metas.append(out_meta)
        elif args.generate_mode == "eval":
            samples = []
            eval_task = config.get("eval_task", "base")
            question_file = config.get("question_file", "questions.jsonl")
            result_file = config.get("result_file", "results.jsonl")
            # load dataset
            eval_dataset = EvalDataset(eval_task, question_file)
            dist_dataset = SampleEvalDataset(eval_dataset)
            iter_datasets = dist_dataset.get_items()
            # generate tokens
            for _ in tqdm(range(len(dist_dataset)), desc='Processing'):
                task_id, prompt, answer = next(iter_datasets)
                if hasattr(sense_tokenization.parser, 'build_inference_meta'):
                    prompt = sense_tokenization.parser.build_inference_meta(prompt, history_metas)
                    context_tokens, _ = sense_tokenization(prompt)
                else:
                    context_tokens, _ = sense_tokenization({"text": prompt, "dialog_history": history_metas})
                context_tokens = torch.LongTensor([context_tokens])
                attention_mask = context_tokens.ne(pad_token_id)

                generation_output = model.generate(
                    input_ids=context_tokens.to(device),
                    max_new_tokens=generation_cfg["max_new_tokens"]
                )
                # generation_output = model.generate(
                #     input_ids=context_tokens.to(device),
                #     attention_mask=attention_mask.to(device),
                #     eos_token_id=tokenizer.eos_token_id,
                #     pad_token_id=tokenizer.pad_token_id,
                #     **generation_cfg
                # )
                s = generation_output[0]
                accept_length = s.numel() - context_tokens.numel()
                output = tokenizer.decode(s, skip_special_tokens=True)
                actual_input = tokenizer.decode(context_tokens.to(device)[0], skip_special_tokens=True)
                raw_output = output.split(actual_input)[-1]
                infos = {
                    "count": accept_length,
                    "accept_length": accept_length,
                    "ave_accept_length": 1
                }
                # postprocess output
                output = text_postprocess(raw_output, eval_task)
                if eval_task == "human_eval":
                    samples.append(
                        dict(task_id=task_id, completion=output)
                    )
                elif eval_task in ["cmmlu", "ceval", "base"]:
                    samples.append(
                        dict(
                            task_id=task_id,
                            input=prompt,
                            output=output,
                            raw_output=raw_output,
                            answer=answer,
                            infos=infos)
                    )
            dist_barrier()

            samples_list = all_gather(samples)
            all_samples = []
            for temps in samples_list:
                all_samples.extend(temps)
            if get_rank() == 0:
                # save results
                save_results(result_file, all_samples, eval_task)
                # evaluate
                evaluate(result_file, eval_task)
