import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from tqdm import tqdm
from llm.utils.general.yaml_loader import load_yaml
from llm.data.nlp_transforms import build_augmentation
from llm.utils.env.hf_dist_helper import setup_distributed, dist_barrier, all_gather, get_rank

from utils.prompt import text_postprocess, save_results, evaluate
from utils.dataset import EvalDataset, SampleEvalDataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, type=str)
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,
                    help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--tokenizer_name', default=None, type=str)
parser.add_argument('--interactive', action='store_true',
                    help="run in the instruction mode (single-turn)")
parser.add_argument('--model_type', default='llama', type=str)
parser.add_argument("--port", default="13333", type=str)
args = parser.parse_args()


if __name__ == '__main__':
    setup_distributed(port=args.port)

    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    if args.model_type == 'llama':
        base_model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=False,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            # device_map='auto',
        )
    elif args.model_type == "baichuan2":
        from llm.models.hf_models.baichuan2.model_7b.modeling_baichuan import BaichuanForCausalLM as BaiChuan2ForCausalLM # noqa
        base_model = BaiChuan2ForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=False,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            # device_map='auto',
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            revision="main",
            torch_dtype=load_type,
            use_cache=False
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    # if model_vocab_size != tokenzier_vocab_size:
    #     assert tokenzier_vocab_size > model_vocab_size
    #     print("Resize model embeddings to fit tokenizer")
    #     base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(
            base_model, args.lora_model, torch_dtype=load_type, device_map='auto',)
    else:
        model = base_model

    if device == torch.device('cpu'):
        model.float()
    model.eval()
    mdoel = model.cuda()
    device = model.device
    # model = DDP(model,
    #             broadcast_buffers=False,
    #             find_unused_parameters=False)

    cfg = load_yaml(args.config)
    cfg['infer_tokenization']['kwargs'].update({'tokenizer': tokenizer})
    infer_tokenization = build_augmentation(cfg['infer_tokenization'])
    infer_tokenization.parser.inference_mode = True

    history_metas = []
    pad_token_id = len(tokenizer) - 1
    generation_config = cfg["infer_cfg"].get("generation_cfg", {})
    with torch.no_grad():
        if args.interactive:
            system_flag = False
            while True:
                print('请输入问题, 输入quit结束:')
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

                if hasattr(infer_tokenization.parser, 'build_inference_meta'):
                    prompt = infer_tokenization.parser.build_inference_meta(raw_input_text, history_metas)
                    context_tokens, _ = infer_tokenization(prompt)
                else:
                    context_tokens, _ = infer_tokenization({"text": raw_input_text, "dialog_history": history_metas})
                context_tokens = torch.LongTensor([context_tokens])
                attention_mask = context_tokens.ne(pad_token_id)

                generation_output = model.generate(
                    input_ids=context_tokens.to(device),
                    attention_mask=attention_mask.to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s, skip_special_tokens=True)
                print(f"SenseChat: {output}")

                input_meta['content'] = raw_input_text
                input_meta['role'] = 'user'
                history_metas.append(input_meta)
                out_meta = {}
                out_meta['content'] = output
                out_meta['role'] = 'assistant'
                history_metas.append(out_meta)
        else:
            samples = []
            infer_config = cfg["infer_cfg"]
            eval_task = infer_config["eval_task"]
            question_file = infer_config["question_file"]
            result_file = infer_config["result_file"]
            # load dataset
            eval_dataset = EvalDataset(eval_task, question_file)
            dist_dataset = SampleEvalDataset(eval_dataset)
            iter_datasets = dist_dataset.get_items()
            # generate tokens
            for idx in tqdm(range(len(dist_dataset)), desc='Processing'):
                # process prompt
                task_id, prompt, answer = next(iter_datasets)
                if hasattr(infer_tokenization.parser, 'build_inference_meta'):
                    prompt = infer_tokenization.parser.build_inference_meta(prompt, history_metas)
                    context_tokens, _ = infer_tokenization(prompt)
                else:
                    context_tokens, _ = infer_tokenization({"text": prompt, "dialog_history": history_metas})
                context_tokens = torch.LongTensor([context_tokens])

                generation_output = model.generate(
                    input_ids=context_tokens.to(device),
                    max_new_tokens=generation_config["max_new_tokens"]
                )
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
