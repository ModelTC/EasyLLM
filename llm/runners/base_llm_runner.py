import sys
import time
import os

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from tqdm import tqdm
import json

import deepspeed
from llm.utils.env import dist_env
from llm.utils.env import (initialize_distributed, get_distributed_info,
                           setup_deepspeed_random_and_activation_checkpointing,
                           set_logging_verbosity,
                           set_random_seed)
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.general.cfg_helper import merge_opts_into_cfg
from llm.utils.general.hook_helper import build_hooks
from llm.utils.general.parser_helper import parse_args
from llm.utils.general.yaml_loader import load_yaml
from llm.utils.general.microbatches import build_num_microbatches_calculator
from llm.utils.model.optimizer_helper import build_optimizer
from llm.utils.model.lr_helper import build_learning_rate_scheduler
from llm.utils.model.initializer import set_defaults_if_not_set_tensor_model_parallel_attributes
from llm.utils.model.ckpt_helper import load_checkpoint, save_checkpoint, load_base_state
from llm.utils.general.utils import get_train_iters

from llm.data import build_tokenizer, build_batch_pipe_fn, build_data_iterator
from llm.models.mg_models import build_model, log_trainable_params, get_layer_info
from llm.models.mg_models import unwrap_model, Float16Module
from llm.models.mg_models import generate_samples_interactive, generate_samples_eval
from llm.utils.tools.dataset import EvalDataset, LocalEvalDataset
from llm.utils.tools.prompt import text_postprocess, save_results, evaluate

_TRAIN_START_TIME = time.time()


class BaseRunner(object):
    def __init__(self, args, cfg=None, training=True, base_type='train'):
        self.args = args
        self.config = cfg
        self.training = training
        self.base_type = base_type
        self.build()

    def build(self):
        self.set_param_components()
        self.build_env()
        if self.args.opts is not None:
            self.config = merge_opts_into_cfg(self.args.opts, self.config)
        self.build_num_microbatches_calculator()
        self.build_tokenizer()
        self.build_hooks()
        self.build_model()
        self.build_data_engine()
        self.build_trainer()
        self.deepspeed_init()
        self.load_checkpoint()

    def set_param_components(self):
        self.start_iteration = 0
        self.consumed_train_samples = 0
        self.consumed_train_tokens = 0
        cfg_loader = self.config["loader"]
        if (not cfg_loader.get("debug", False)) and cfg_loader.get("load_base_state", False):
            self.start_iteration, self.consumed_train_samples, \
                self.consumed_train_tokens = load_base_state(cfg_loader)

        # set deepspeed configs
        self.deepspeed = self.config['runtime'].get('deepspeed', True)
        assert self.deepspeed is True, 'only support deepspeed mode now!'
        cfg_deepspeed = self.config.get('deepspeed', None)
        assert cfg_deepspeed is not None and (cfg_deepspeed.get('config', None) is not None), 'deepspeed mode must provide the configs of deepspeed!'     # noqa
        cfg_deepspeed['config'].update({'train_micro_batch_size_per_gpu': self.config['data'][self.base_type]['micro_batch_size'],     # noqa
                                        'train_batch_size': self.config['data'][self.base_type]['global_batch_size']})
        self.cfg_deepspeed = cfg_deepspeed
        assert self.args.deepspeed_config is None, 'please pass the deepspeed config in the config.json file, and the deepspeed_config is disabled now.'     # noqa
        # set lora configs
        self.lora_mode = self.config['runtime'].get('lora_mode', False)
        cfg_lora = self.config.get('lora', None)
        if self.lora_mode:
            assert cfg_lora is not None, 'lora mode must provide the configs of lora!'
        self.cfg_lora = cfg_lora
        # fp16 and bf16
        assert self.config['runtime'].get('fp16', False) == cfg_deepspeed['config'].get('fp16', {}).get('enabled', False)       # noqa
        assert self.config['runtime'].get('bf16', False) == cfg_deepspeed['config'].get('bf16', {}).get('enabled', False)       # noqa

    def build_env(self, rank=None, local_rank=None):
        cfg_runtime = self.config['runtime']
        # get env info
        # rank, local_rank, world_size, tensor_model_parallel_size, \
        #     pipeline_model_parallel_size = get_distributed_info(cfg_runtime, self.args.launcher, self.args.port)
        rank, local_rank, world_size, tensor_model_parallel_size, \
            pipeline_model_parallel_size, context_parallel_size = get_distributed_info(cfg_runtime, self.args.launcher, self.args.port)
        # initialize env
        # Pytorch distributed.
        # initialize_distributed(rank, local_rank, world_size, tensor_model_parallel_size,
        #                        pipeline_model_parallel_size, cfg_runtime.get('distributed_backend', 'nccl'),
        #                        self.args.launcher)
        initialize_distributed(rank, local_rank, world_size, tensor_model_parallel_size,
                               pipeline_model_parallel_size, context_parallel_size, cfg_runtime.get('distributed_backend', 'nccl'),
                               self.args.launcher)
        # Initialize deepspeed random and activation checkpointing.
        if self.deepspeed:
            num_layers, checkpoint_num_layers = get_layer_info(self.config['model'])
            cfg_activation_checkpoint = self.cfg_deepspeed.get('activation_checkpoint', {})
            cfg_activation_checkpoint.update({'base_num_layers': num_layers,
                                              'checkpoint_num_layers': checkpoint_num_layers})
            setup_deepspeed_random_and_activation_checkpointing(**cfg_activation_checkpoint)
        # Set logging verbosity
        set_logging_verbosity(rank, cfg_runtime.get('log_level', 'info'),
                              cfg_runtime.get('log_level_replica', 'error'),
                              deepspeed=self.deepspeed)
        # Set random seed.
        set_random_seed(cfg_runtime.get('seed', 42), cfg_runtime.get('dp_random_init', False))
        # get global start time
        global _TRAIN_START_TIME
        start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        self.start_time = start_time_tensor.item()
        logger.info('Initialize env done! Times (seconds): {:.3f}'.format(time.time() - self.start_time))

    def build_num_microbatches_calculator(self):
        if self.training:
            self.num_microbatches_calculator = build_num_microbatches_calculator(self.config['data']['train']['batch_calculator'])      # noqa
            self.num_microbatches_calculator.update(self.consumed_train_samples, True)
        else:
            self.num_microbatches_calculator = None

    def build_tokenizer(self):
        self.tokenizer = build_tokenizer(self.config['tokenizer'])

    def build_hooks(self):
        cfg_hooks = self.config.get('hooks', [])
        self._hooks = build_hooks(self, cfg_hooks, is_train=self.training, add_log_if_not_exists=True)
        logger.info('build hooks done')

    def build_data_engine(self):
        cfg_data = self.config['data']
        data_types = cfg_data.get('data_types', ['train', 'test'])
        self.data_iterators = {}
        self.batch_pipe_func = {}
        for data_type in data_types:
            assert data_type in ['train', 'valid', 'test', 'infer'], 'data type only support train, valid, test, and infer'       # noqa
            self.batch_pipe_func[data_type] = build_batch_pipe_fn(cfg_data[data_type]['batch_pipe'], self.tokenizer)
            if data_type == 'infer':
                infer_type = cfg_data[data_type].get('infer_type', 'interactive')
                if infer_type == 'interactive':
                    continue        # skip build data_iterators for inference mode
            data_iterator, dataset_size = build_data_iterator(self.tokenizer, cfg_data, self.consumed_train_samples, data_type)  # noqa
            self.data_iterators[data_type] = data_iterator
        if self.training:
            epoch = self.config['trainer'].get('epoch', -1)
            if epoch > 0:
                global_batch_size = self.num_microbatches_calculator.global_batch_size
                train_iters = int((dataset_size.item() // global_batch_size + 1) * epoch)
            else:
                train_iters = self.config['trainer'].get('train_iters', 100)
            self.set_train_iters(train_iters)

    def build_model(self):
        if self.deepspeed:
            remote_device = self.cfg_deepspeed.get('remote_device', 'none')
            zero_stage = self.cfg_deepspeed['config'].get('zero_optimization', {}).get('stage', 1.0)
            # assert args.virtual_pipeline_model_parallel_size is None, "Interleaved pipeline schedule is not yet supported for text generation."      # noqa
            with deepspeed.zero.Init(data_parallel_group=dist_env.get_data_parallel_group(),
                                     remote_device=None if (remote_device == 'none') else remote_device,
                                     config_dict_or_path=self.cfg_deepspeed['config'],
                                     enabled=(zero_stage == 3), mpu=dist_env):
                model = build_model(self.tokenizer, self.config, self.lora_mode,
                                    self.cfg_lora, base_type=self.base_type)
                # set trainable params
                if hasattr(model, 'set_train_params'):
                    model.set_train_params(model, lora_mode=self.lora_mode,
                                           cfg_lora=self.cfg_lora, is_train=self.training)
                # log trainable params
                log_trainable_params(model)
            for param in model.parameters():
                set_defaults_if_not_set_tensor_model_parallel_attributes(param)     # noqa
            self.model = model
        else:
            raise NotImplementedError

    def set_train_iters(self, train_iters):
        self.total_train_iters = get_train_iters(self.num_microbatches_calculator,
                                                 train_iters,
                                                 self.config['trainer'].get('train_samples', None))

    def build_trainer(self):
        unwrapped_model = unwrap_model(self.model, (torchDDP, Float16Module))
        if self.training:
            cfg_optim = self.config['trainer']['optimizer']
            optimizer = build_optimizer(cfg_optim, unwrapped_model, deepspeed=self.deepspeed)
            cfg_lr_scheduler = self.config['trainer']['lr_scheduler']
            cfg_lr_scheduler['kwargs']['max_lr'] = cfg_optim['kwargs']['lr']        # noqa
            if cfg_lr_scheduler['type'] == 'iter_base_annealing':
                cfg_lr_scheduler['kwargs']['global_batch_size'] = self.num_microbatches_calculator.global_batch_size
                train_iters = cfg_lr_scheduler['kwargs'].get('train_iters', None)
                if train_iters is None:
                    cfg_lr_scheduler['kwargs']['train_iters'] = self.total_train_iters
            lr_scheduler = build_learning_rate_scheduler(cfg_lr_scheduler, optimizer)
            if hasattr(lr_scheduler, 'consumed_train_tokens'):
                lr_scheduler.consumed_train_tokens = self.consumed_train_tokens
            if hasattr(lr_scheduler, 'num_steps'):
                lr_scheduler.num_steps = self.consumed_train_samples     # deepspeed based
        else:
            optimizer = None
            lr_scheduler = None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def deepspeed_init(self):
        # args = self.args
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=self.cfg_deepspeed['config'],
            args=None,
        )

        assert model.fp16_enabled() == self.config['runtime'].get('fp16', False), "fp16 config does not match deepspeed"
        assert model.bfloat16_enabled() == self.config['runtime'].get('bf16', False), "bf16 config does not match deepspeed"        # noqa

        if isinstance(model, deepspeed.PipelineEngine):
            assert model.grid.get_pipe_parallel_rank() == dist_env.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == dist_env.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == dist_env.get_data_parallel_rank()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def load_checkpoint(self):
        # args = self.args
        torch.distributed.barrier()
        load_checkpoint(self.model, self.optimizer, self.config['loader'],
                        self.start_iteration, self.lora_mode, self.cfg_lora)     # noqa

        # TODO hardcode fix torch load bugs
        for k, v in self.optimizer.state_dict()['base_optimizer_state']['state'].items():
            if 'step' in v:
                v['step'] = v['step'].to(torch.cuda.current_device())
        torch.distributed.barrier()

    def save_checkpoint(self, cur_iter):
        # args = self.args
        cfg_saver = self.config['saver']
        save_interval = cfg_saver.get('save_interval', 0)
        if (save_interval and (cur_iter + 1) % save_interval == 0) or (cur_iter + 1 == self.total_train_iters):
            save_checkpoint((cur_iter + 1), self.consumed_train_samples, self.consumed_train_tokens,
                            self.model, cfg_saver, self.lora_mode, self.cfg_lora)
            logger.info('Saving checkpoint at the {}_th iter.'.format(cur_iter + 1))

    def forward_step(self, data_iterator):
        # args = self.args
        if self.deepspeed:
            assert isinstance(self.model, deepspeed.PipelineEngine), self.model
            loss = self.model.train_batch(data_iter=data_iterator,
                                          dynamic=self.config['runtime'].get('dynamic', False),
                                          sequence_parallel=self.model.sequence_parallel)
            return {'lm_loss': loss}
        else:
            raise NotImplementedError

    def train(self):
        # args = self.args
        data_type = 'train'
        assert data_type == self.base_type, 'Training type! But the base type is {}'.format_map(self.base_type)
        assert self.deepspeed, 'Only support deepspeed now'

        # Iterations.
        micro_batch_size = self.config['data'][data_type]['micro_batch_size']
        seq_length = self.config['data'][data_type]['seq_length']
        # hooks
        self._hooks('before_train')
        # model status
        self.model.set_batch_fn(self.batch_pipe_func[data_type])
        if hasattr(self.model, 'set_train_status'):
            self.model.set_train_status(self.model, self.lora_mode)
        else:
            self.model.train()

        for iteration in range(self.start_iteration, self.total_train_iters):
            self._hooks('before_train_iter', iteration)
            self.num_microbatches_calculator.update(self.consumed_train_samples, True)
            # inform deepspeed of any batch size changes
            global_batch_size = dist_env.get_data_parallel_world_size() * micro_batch_size * \
                self.num_microbatches_calculator.get()
            self.model.set_train_batch_size(global_batch_size)
            # forward step
            output = self.forward_step(self.data_iterators[data_type])

            self.consumed_train_samples += global_batch_size
            self.consumed_train_tokens += global_batch_size * seq_length
            if hasattr(self.lr_scheduler, 'consumed_train_tokens'):
                self.lr_scheduler.consumed_train_tokens = self.consumed_train_tokens
            self._hooks('after_train_iter', iteration, output)
            self.save_checkpoint(iteration)
        self._hooks('after_train')

    def generate(self):
        args = self.args
        data_type = 'infer'
        assert data_type == self.base_type, 'Inference type! But the base type is {}'.format_map(self.base_type)
        self.model.set_batch_fn(self.batch_pipe_func[data_type])

        self.model.eval()
        with torch.no_grad():
            if self.deepspeed:
                args.out_seq_length = int(self.config['data'][data_type]['seq_length'])
                if args.generate_mode == 'interactive':
                    generate_samples_interactive(
                        args,
                        self.config,
                        self.model,
                        self.tokenizer,
                        force_eos_id=args.force_eos_id)
                elif args.generate_mode == 'logp':
                    self.infer_logp(args.logp_file)
                elif args.generate_mode == "eval":
                    args.out_seq_length = self.config["infer_cfg"]["generation_cfg"].get("max_new_tokens", 100)
                    # build tokenization
                    from llm.data.nlp_transforms import build_augmentation
                    self.config['infer_tokenization']['kwargs'].update({'tokenizer': self.tokenizer})
                    sense_tokenization = build_augmentation(self.config["infer_tokenization"])
                    sense_tokenization.parser.inference_mode = True

                    eval_task = self.config["infer_cfg"].get("eval_task", "base")
                    question_file = self.config["infer_cfg"].get("question_file", "questions.jsonl")
                    result_file = self.config["infer_cfg"].get("result_file", "results.jsonl")
                    # load dataset
                    eval_dataset = EvalDataset(eval_task, question_file)
                    local_dataset = LocalEvalDataset(eval_dataset)
                    # dist_dataset = SampleEvalDataset(eval_dataset)
                    # iter_datasets = dist_dataset.get_items()
                    history_metas, samples = [], []

                    # generate tokens
                    for _ in tqdm(range(len(eval_dataset)), desc='Processing'):
                        # task_id, prompt, answer = next(iter_datasets)
                        task_id, prompt, answer = local_dataset.get_data(_)
                        if hasattr(sense_tokenization.parser, 'build_inference_meta'):
                            prompt = sense_tokenization.parser.build_inference_meta(prompt, history_metas)
                            context_tokens, _ = sense_tokenization(prompt)
                        else:
                            context_tokens, _ = sense_tokenization({"text": prompt, "dialog_history": history_metas})

                        raw_output = generate_samples_eval(
                            args, context_tokens, self.model, self.tokenizer, force_eos_id=args.force_eos_id)
                        accept_length = len(raw_output) if raw_output else 0

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

                    if dist_env.is_pipeline_first_stage() and dist_env.get_tensor_model_parallel_rank() == 0:
                        # save results
                        save_results(result_file, samples, eval_task)
                        # evaluate
                        evaluate(result_file, eval_task)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        sys.stdout.flush()

    def infer_logp(self, logp_file):
        data_type = 'infer'
        logger.warning_once("Warning: infer_logp() only support model_parallel, batch size 1 now.")
        cfg = self.config
        assert cfg['data']['infer']['global_batch_size'] == 1

        from llm.data.nlp_dataset import IGNORE_INDEX

        assert not os.path.exists(logp_file), "{} already exist".format(logp_file)

        data_size = len(self.data_iterators[data_type]) if self.data_iterators[data_type] is not None else -1
        data_size_cuda = torch.cuda.LongTensor([data_size])
        gather_data_size = dist_env.gather_from_tensor_model_parallel_region(data_size_cuda)
        data_size = gather_data_size.cpu().max().item()
        logger.info(f'Infer Logp! The dataset has {data_size} items.')
        logger.warning_once("LLMs output the probobilities from the second token")

        with torch.no_grad():
            res = []
            for idx in tqdm(range(data_size)):
                output = self.model.eval_batch(self.data_iterators[data_type],
                                               compute_loss=False,
                                               reduce_output=None,
                                               dynamic=True,
                                               sequence_parallel=self.model.sequence_parallel)
                output_tensor = output[0]
                output_labels = output[1][0]
                if dist_env.is_pipeline_last_stage():
                    output_tensor = dist_env.gather_from_tensor_model_parallel_region(output_tensor)
                    # output_labels = dist_env.gather_from_tensor_model_parallel_region(output_labels)

                bs = output_tensor.shape[0]
                for i in range(bs):
                    answer_mask = (output_labels[i] != IGNORE_INDEX)
                    tensor = output_tensor[i][answer_mask]
                    label = output_labels[i][answer_mask]
                    per_token_logps = torch.gather(tensor.log_softmax(dim=-1), dim=1, index=label.unsqueeze(1)).squeeze(1)
                    per_token_logps = per_token_logps.cpu().numpy()
                    res.append(per_token_logps)
            if dist_env.get_tensor_model_parallel_rank() == 0 and dist_env.get_data_parallel_rank() == 0:
                with open(logp_file, "w") as wf:
                    for idx, logps in enumerate(res):
                        wf.write(json.dumps({"logp": logps.tolist()}) + '\n')
                    logger.info(f'Successfully save logp to {logp_file}.')

    def reorder(self, res):
        '''
        res list, [device_num, [batch_num]]
        reorder dist test sampler to original list order
        '''
        all_res_num = 0
        for d_res in res:
            all_res_num += len(d_res)
        ordered_results = [0] * all_res_num
        data_world_size = dist_env.get_data_parallel_world_size()
        for device_id in range(data_world_size):
            ordered_results[device_id::data_world_size] = res[device_id]
        return ordered_results


def main():
    args = parse_args()
    if os.environ.get('ACCELERATOR_BACKEND') == 'TORCH_NPU':
        args.distributed_backend = 'hccl'
    assert args.config is not None, 'please provide a config file'
    cfg = load_yaml(args.config)
    runtime_none_keys = ['seed', 'local_rank', 'tensor_model_parallel_size',
                         'pipeline_model_parallel_size', 'distributed_backend']
    runtime_store_true_keys = ['fp16', 'bf16', 'deepspeed', 'lora_mode']
    cfg['runtime'] = cfg.setdefault('runtime', {})
    for key in (runtime_none_keys + runtime_store_true_keys):
        val = getattr(args, key)
        if key in runtime_none_keys and val is not None:
            cfg['runtime'].update({key: val})
        elif key in runtime_store_true_keys and val is True:
            cfg['runtime'].update({key: val})
    if args.inference:
        # sequence_parallel is not supported in inference
        if 'kwargs' in cfg['model']:
            if 'sequence_parallel' in cfg['model']['kwargs']:
                cfg['model']['kwargs']['sequence_parallel'] = False
        runner = BaseRunner(args, cfg, training=False, base_type='infer')
        runner.generate()
    else:
        runner = BaseRunner(args, cfg, training=True, base_type='train')
        runner.train()


if __name__ == "__main__":
    main()
