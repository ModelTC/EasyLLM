import torch
import deepspeed
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.utils.general.yaml_loader import load_yaml
from llm.utils.general.parser_helper import parse_args
from llm.utils.model.optimizer_helper import build_optimizer
from llm.utils.model.lr_helper import build_learning_rate_scheduler
from llm.utils.general.hook_helper import build_hooks
from llm.utils.general.log_helper import default_logger as logger
from llm.data.tokenizer import build_tokenizer
from llm.utils.env.hf_dist_helper import (
    setup_distributed,
    get_world_size
)
from llm.utils.general.hf_build_utils import (
    build_batch_collator,
    build_dataloader,
    build_dataset,
    build_model,
    hack_model,
    build_augmentation
)
from llm.utils.general.hf_utils import (
    hf_inference,
    hf_inference_multimodal,
    load_from_ds,
    load_from_hf,
    save_hf_checkpoint,
    save_ds_checkpoints
)


class HFRunner(object):
    def __init__(self, args, cfg=None, training=True):
        self.args = args
        self.config = cfg
        self.training = training
        self.deepspeed = False
        self.dtype = torch.float16
        if 'deepspeed' in self.config:
            self.deepspeed = self.config['deepspeed'].get('enabled', False)
            self.dtype = self.get_dtype_from_ds(self.config['deepspeed']['config'])
        if 'runtime' not in self.config:
            self.config['runtime'] = {}
        self.gradient_accumulation_steps = self.config['runtime'].get('gradient_accumulation_steps', 1)
        self.start_iter = 0
        self.build()
        if not self.deepspeed:
            from llm.utils.general.grad_scaler import ShardedGradScaler
            self.scaler = ShardedGradScaler(enabled=True)
        if self.training:
            logger.info(f"Start_iter: {self.start_iter}")
            logger.info(f"Train_iters: {self.train_iters}")
            logger.info(f"Train_epoch_size: {self.train_epoch_size}")
            logger.info(f"Total epoch: {self.get_max_train_epoch()}")
            logger.info(f"Gradient_accumulation_steps: {self.gradient_accumulation_steps}")
            logger.info(f"Global_train_batch_size: {self.global_train_batch_size}")

    def get_dtype_from_ds(self, ds_confg):
        bf16 = False
        fp16 = False
        if 'bf16' in ds_confg:
            bf16 = ds_confg['bf16'].get('enabled', False)
        if 'fp16' in ds_confg:
            fp16 = ds_confg['fp16'].get('enabled', False)
        assert bf16 != fp16
        if bf16:
            return torch.bfloat16
        if fp16:
            return torch.float16

    def build(self):
        self.build_tokenizer()
        self.build_model()
        self.build_hooks()
        self.build_data()
        self.build_trainer()
        if self.deepspeed and self.training:
            self.deepspeed_init()
        self.load_checkpoints(self.config['loader'])

    def get_cur_train_epoch(self):
        epoch = (self.cur_iter // self.train_epoch_size) + 1
        return epoch

    def get_max_train_epoch(self):
        epoch = (max(self.train_iters - 1, 1)) // self.train_epoch_size + 1
        return epoch

    def build_optimzer(self):
        optimizer_cfg = self.config['trainer']['optimizer']
        self.optimizer = build_optimizer(optimizer_cfg, self.model)

    def build_lr_scheduler(self):
        lr_scheduler_cfg = self.config['trainer']['lr_scheduler']
        self.lr_scheduler = build_learning_rate_scheduler(lr_scheduler_cfg, self.optimizer)

    def build_tokenizer(self):
        self.tokenizer = build_tokenizer(self.config['tokenizer'])

    def build_model(self):
        self.model = build_model(self.config['model'])
        if self.config['runtime'].get('gradient_checkpointing', True):
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "base_model"):
                self.model.base_model.gradient_checkpointing_enable()
            if self.config['model'].get('peft_model_cfg', None) is not None:
                modules_to_save = self.config['model']['peft_model_cfg'].get('modules_to_save', [])
                if len(modules_to_save) == 0:
                    hack_model(self.model)
        if not self.deepspeed:
            self.mdoel = self.model.cuda()
            if self.training:
                self.model = DDP(self.model,
                                 broadcast_buffers=False,
                                 find_unused_parameters=False)

    def build_trainer(self):
        world_size = get_world_size()
        if self.training:
            self.train_iters = self.config['trainer']['train_iters']
            self.save_interval = self.config['saver'].get('save_interval', 100)
            self.build_optimzer()
            self.build_lr_scheduler()
            self.mirco_train_batch_size = self.data_loaders['train'].batch_sampler.batch_size
            self.train_epoch_size = self.data_loaders['train'].get_epoch_size()
            self.global_train_batch_size = self.mirco_train_batch_size * world_size
        else:
            if 'test' in self.data_loaders:
                self.mirco_test_batch_size = self.data_loaders['test'].batch_sampler.batch_size
                self.test_epoch_size = self.data_loaders['test'].get_epoch_size()
            else:
                self.mirco_test_batch_size = 1
                self.test_epoch_size = 1
            self.global_test_batch_size = self.mirco_test_batch_size * world_size
            self.global_train_batch_size = 1

    def build_hooks(self):
        cfg_hooks = self.config.get('hooks', [])
        self._hooks = build_hooks(self, cfg_hooks, is_train=self.training, add_log_if_not_exists=True)
        logger.info('build hooks done')

    def deepspeed_init(self):
        ds_config = self.config['deepspeed']['config']
        if ds_config.get('gradient_accumulation_steps', 'auto') == 'auto':
            ds_config['gradient_accumulation_steps'] = self.gradient_accumulation_steps
        self.gradient_accumulation_steps = ds_config['gradient_accumulation_steps']
        self.global_train_batch_size *= self.gradient_accumulation_steps
        self.train_epoch_size //= self.gradient_accumulation_steps
        if 'train_batch_size' not in ds_config or ds_config['train_batch_size'] == 'auto':
            ds_config['train_batch_size'] = self.global_train_batch_size
        if 'train_micro_batch_size_per_gpu' not in ds_config or ds_config['train_micro_batch_size_per_gpu'] == 'auto':
            ds_config['train_micro_batch_size_per_gpu'] = self.mirco_train_batch_size
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=self.config['deepspeed']['config'],
            args=None,
        )
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def load_checkpoints(self, load_cfg):
        if load_cfg.get('enabled', False):
            load_dir = load_cfg.get("load_path", None)
            mode = load_cfg.get('load_mode', 'hf')
            if not load_dir:
                logger.info("No weights need to be loaded.")
                return
            logger.info(f"Loading model from {load_dir}")
            if mode == 'huggingface':
                try:
                    if self.config['model'].get('mode', "from_pretrained") == "from_config":
                        load_from_hf(self, load_cfg)
                except:  # noqa
                    logger.warning("Loading failed by huggingface")
            elif mode == 'deepspeed':
                try:
                    load_from_ds(self, load_cfg)
                except:  # noqa
                    logger.warning("Loading failed by deepspeed")
            else:
                raise NotImplementedError

    def build_data(self):
        self.data_loaders = {}
        for data_type in self.config['data'].get('data_types', []):
            dataset_cfg = self.config['data'][data_type]['dataset']
            dataset = build_dataset(dataset_cfg, self.tokenizer)
            batch_collector_cfg = self.config['data'][data_type]['batch_collector']
            batch_collector_cfg['kwargs']['offset_label'] = False
            batch_collector = build_batch_collator(batch_collector_cfg, self.tokenizer)
            if data_type == 'val' or data_type == 'test':
                self.config['data'][data_type]['batch_sampler']['infinite'] = False
                self.config['data'][data_type]['batch_sampler']['kwargs']['sampler']['type'] = 'dist_test'
            data_loader = build_dataloader(self.config['data'][data_type], dataset, batch_collector)
            self.data_loaders[data_type] = data_loader

    def batch2device(self, batch):
        batch['input_ids'] = batch['input_ids'].to(device=torch.device('cuda'))
        batch['labels'] = batch['labels'].to(device=torch.device('cuda'))
        batch['attention_mask'] = batch['attention_mask'].to(device=torch.device('cuda'))
        return batch

    def get_batch(self, batch_type='train'):
        assert batch_type in self.data_loaders
        if not hasattr(self, 'data_iterators'):
            self.data_iterators = {}
        if batch_type not in self.data_iterators:
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
        else:
            iterator = self.data_iterators[batch_type]
        try:
            batch = next(iterator)
        except StopIteration as e:  # noqa
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
            batch = next(iterator)
        batch = self.batch2device(batch)
        return batch

    def _save(self, iteration):
        if (iteration + 1) % self.save_interval == 0:
            self.save_checkpoint(self.config.get('saver', {}), iteration + 1)

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
        save_hf_checkpoint(self, self.config['saver'], self.train_iters)
        self._hooks('after_train')

    def infer(self):
        self.model.eval()
        self.model.cuda()
        device = self.model.device

        assert 'infer_tokenization' in self.config, "infer_tokenization does not exist."
        self.config['infer_tokenization']['kwargs'].update({'tokenizer': self.tokenizer})
        sense_tokenization = build_augmentation(self.config["infer_tokenization"])
        sense_tokenization.parser.inference_mode = True
        model_type = self.config["infer_cfg"].get("model_type", "llm")
        if model_type == "llm":
            hf_inference(self.config["infer_cfg"],
                         self.model,
                         sense_tokenization,
                         device,
                         args=self.args)
        elif model_type == "multimodal":
            hf_inference_multimodal(self.config["infer_cfg"],
                                    self.model,
                                    sense_tokenization,
                                    device,
                                    args=self.args)
        else:
            raise NotImplementedError

    def save_checkpoint(self, save_cfg, global_step):
        if save_cfg.get('enabled', True):
            save_path = save_cfg.get('save_path', "checkpoints")
            assert save_path is not None, "Save path must be provided!!!"
            save_mode = save_cfg.get('save_mode', 'deepspeed')
            if save_mode == 'huggingface':
                save_hf_checkpoint(self, save_cfg, global_step)
            elif save_mode == 'deepspeed':
                save_ds_checkpoints(self, save_cfg, global_step)
            else:
                raise NotImplementedError


def main(args):
    cfg = load_yaml(args.config)
    cfg['runtime'] = cfg.setdefault('runtime', {})
    if not args.inference:
        runner = HFRunner(args, cfg, training=True)
        runner.train()
    else:
        runner = HFRunner(args, cfg, training=False)
        runner.infer()


if __name__ == "__main__":
    args = parse_args()
    setup_distributed(launcher=args.launcher, backend=args.distributed_backend, port=args.port)
    main(args)
