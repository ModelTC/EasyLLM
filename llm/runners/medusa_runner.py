from llm.runners.hf_runner import HFRunner
from llm.utils.general.yaml_loader import load_yaml
from llm.utils.general.parser_helper import parse_args
from llm.utils.env.hf_dist_helper import setup_distributed
from llm.data.nlp_transforms import build_augmentation, SenseTokenization
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.tools.prompt import text_postprocess, save_results, evaluate
import torch
import os
from tqdm import tqdm


class MedusaRunner(HFRunner):
    def __init__(self, args, cfg=None, training=True):
        super().__init__(args, cfg, training)

    def build(self):
        self.build_tokenizer()
        self.build_model()
        self.model.tokenizer = self.tokenizer
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        self.build_hooks()
        self.build_data()
        self.build_trainer()
        if self.deepspeed and self.training:
            self.deepspeed_init()
        self.load_checkpoints(self.config['loader'])

    def train(self):
        self.model.train()
        self._hooks('before_train')
        for iteration in range(
            self.start_iter * self.gradient_accumulation_steps,
            self.train_iters * self.gradient_accumulation_steps,
        ):
            self.cur_iter = iteration // self.gradient_accumulation_steps
            batch = self.get_batch()
            self._hooks('before_train_iter', self.cur_iter, batch)
            with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
                output = self.model(batch['input_ids'],
                                    batch['attention_mask'],
                                    labels=batch['labels'],
                                    return_dict=True,
                                    use_cache=False)
            losses = [val for name, val in output.items() if name.find('loss') >= 0]
            loss = sum(losses)
            if self.deepspeed:
                self.model.backward(loss)
                self.model.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
            if (iteration + 1) % self.gradient_accumulation_steps == 0:
                self._save(self.cur_iter)
                self._hooks('after_train_iter', self.cur_iter, output)
        self.save_medusa_head()
        self._hooks('after_train')

    def generate(self):
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            if self.args.generate_mode is None:
                generate_mode = 'interactive'
            else:
                generate_mode = self.args.generate_mode

            assert 'infer_tokenization' in self.config, 'inference mode must provide a inference tokenizaition (e.g. sense_tokenization)'        # noqa
            self.config['infer_tokenization']['kwargs'].update({'tokenizer': self.tokenizer})
            infer_tokenization = build_augmentation(self.config['infer_tokenization'])
            infer_tokenization.parser.inference_mode = True

            if generate_mode == "interactive":
                history_metas = []
                system_flag = False
                while True:
                    logger.info("Input your questions (quit to exit) >>> ")
                    raw_text = input()
                    input_meta = {}
                    if isinstance(infer_tokenization, SenseTokenization) and system_flag:
                        input_meta['content'] = raw_text
                        input_meta['role'] = "system"
                        history_metas.append(input_meta)
                        system_flag = False
                        continue
                    if len(raw_text.strip()) == 0:
                        logger.info('Input questions must not be empty!')
                        continue
                    if raw_text == 'quit':
                        logger.info('Input quit, quit process!')
                        break
                    if raw_text == 'system':
                        system_flag = True
                        continue
                    if raw_text == 'clean':
                        logger.info('Clean dialog, the conversation will restart from scratch...')
                        history_metas = []
                        system_flag = False
                        continue
                    ori_raw_text = raw_text
                    if hasattr(infer_tokenization.parser, 'build_inference_meta'):
                        meta = infer_tokenization.parser.build_inference_meta(raw_text, history_metas)
                        context_tokens, _ = infer_tokenization(meta)
                    else:
                        raw_text = infer_tokenization({"text": ori_raw_text, "dialog_history": history_metas})
                        context_tokens = self.tokenizer.encode(raw_text)
                    output, infos = self.model.medusa_generate(torch.LongTensor([context_tokens]).cuda(),
                                                               medusa_generate=self.args.medusa_generate,
                                                               temperature=self.args.medusa_temperature,
                                                               max_steps=self.args.medusa_max_steps)
                    logger.info("\nContext: {}".format(ori_raw_text))
                    logger.info("\nEasyLLM: {}".format(output))
            elif generate_mode == "eval":
                from llm.utils.tools.dataset import EvalDataset, LocalEvalDataset
                cfg_medusa_infer = self.config.get("medusa_infer_config", {})
                samples, history_metas = [], []
                # load dataset
                assert "eval_task" in cfg_medusa_infer, "eval_task not in cfg_medusa_infer"
                assert "question_file" in cfg_medusa_infer, "question_file not in cfg_medusa_infer"
                eval_dataset = EvalDataset(
                    cfg_medusa_infer["eval_task"],
                    question_dir=cfg_medusa_infer["question_file"],
                    load_type=cfg_medusa_infer.get("load_type", "line")
                )
                local_eval_dataset = LocalEvalDataset(eval_dataset)
                # generate tokens
                for idx in tqdm(range(len(eval_dataset)), desc='Processing'):
                    task_id, prompt, answer = local_eval_dataset.get_data(idx)
                    if args.with_infer_tokenization:
                        if hasattr(infer_tokenization.parser, 'build_inference_meta'):
                            prompt = infer_tokenization.parser.build_inference_meta(prompt, history_metas)
                            context_tokens, _ = infer_tokenization(prompt)
                        else:
                            prompt = infer_tokenization({"text": prompt, "dialog_history": history_metas})
                            context_tokens = self.tokenizer.encode(prompt)
                    else:
                        context_tokens = self.tokenizer.encode(prompt)

                    from llm.models.hf_models.medusa.medusa_choices import mc_sim_7b_63, mc_sim_7b_top5, mc_sim_7b_3_head_top5
                    dict_medusa_choices = {
                        "mc_sim_7b_63": mc_sim_7b_63,
                        "mc_sim_7b_top5": mc_sim_7b_top5,
                        "mc_sim_7b_3_head_top5": mc_sim_7b_3_head_top5
                    }
                    medusa_choices = dict_medusa_choices[cfg_medusa_infer["medusa_choices"]]
                    raw_output, infos = self.model.medusa_generate(torch.LongTensor([context_tokens]).cuda(),
                                                                   medusa_generate=cfg_medusa_infer["medusa_generate"],
                                                                   temperature=cfg_medusa_infer["medusa_temperature"],
                                                                   max_steps=cfg_medusa_infer["medusa_max_steps"],
                                                                   medusa_choices=medusa_choices,
                                                                   generate_mode=generate_mode)
                    # postprocess output
                    output = text_postprocess(raw_output, cfg_medusa_infer["eval_task"])
                    if args.eval_task == "human_eval":
                        results = dict(task_id=task_id, completion=output)
                    elif args.eval_task in ["cmmlu", "ceval", "base"]:
                        results = dict(
                            task_id=task_id,
                            input=prompt,
                            output=output,
                            raw_output=raw_output,
                            answer=answer,
                            infos=infos)
                    samples.append(results)
                # save results
                save_results(
                    cfg_medusa_infer["infer_file"],
                    samples,
                    cfg_medusa_infer["eval_task"]
                )
                # evaluate
                evaluate(cfg_medusa_infer["infer_file"], cfg_medusa_infer["eval_task"])

    def save_medusa_head(self):
        if hasattr(self.model, "module"):
            lm_head = self.model.module.medusa_head
        else:
            lm_head = self.model.medusa_head
        output_dir = self.config['saver'].get('save_path', "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            lm_head.state_dict(),
            os.path.join(output_dir, "medusa_lm_head.pth"),
        )


def main(args):
    cfg = load_yaml(args.config)
    cfg['runtime'] = cfg.setdefault('runtime', {})
    if not args.inference:
        runner = MedusaRunner(args, cfg, training=True)
        runner.train()
    else:
        runner = MedusaRunner(args, cfg, training=False)
        runner.generate()


if __name__ == "__main__":
    args = parse_args()
    setup_distributed(launcher=args.launcher, backend=args.distributed_backend, port=args.port)
    main(args)
