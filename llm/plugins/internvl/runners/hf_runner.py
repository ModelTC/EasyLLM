import torch
import deepspeed
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.utils.general.yaml_loader import load_yaml
from llm.utils.general.parser_helper import parse_args
# from llm.utils.model.optimizer_helper import build_optimizer
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
from llm.plugins.internvl.data.data_utils import (
    IMG_CONTEXT_TOKEN,
    IMG_START_TOKEN,
    IMG_END_TOKEN,
    BOX_START_TOKEN,
    BOX_END_TOKEN,
    REF_START_TOKEN,
    REF_END_TOKEN,
    QUAD_START_TOKEN,
    QUAD_END_TOKEN
)
from llm.runners.hf_runner import HFRunner
from llm.models.hf_models.sequence import init_sequence_parallel


class MLLMHFRunner(HFRunner):
    def build_env(self):
        from llm.utils.general.hf_utils import set_random_seed
        set_random_seed(self.config['runtime'].get('seed', 42))
        deepspeed.init_distributed(dist_backend='nccl')
        self.mpu = None

        from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
        # self.hf_deepspeed_config = HfTrainerDeepSpeedConfig("/mnt/afs_2/zhangfeizhao/mllm/internvl/open_source/workdir_el/hf/zero_stage3_config.json")
        self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.config['deepspeed']['config'])
        if self.config['deepspeed']['config']["zero_optimization"].get("mics_shard_size", None) is not None:
            from llm.utils.env import dist_env, initialize_model_parallel, is_unitialized
            if is_unitialized():
                initialize_model_parallel()
            else:
                pass
            self.mpu = dist_env

            import transformers
            from llm.plugins.internvl.runners.transformers_patch import from_pretrained, _from_config
            transformers.PreTrainedModel.from_pretrained = from_pretrained
            transformers.PreTrainedModel._from_config = _from_config

        sequence_parallel_world_size = self.config['runtime'].get('sp', 1)
        init_sequence_parallel(sequence_parallel_world_size)

    def build(self):
        self.build_env()
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

    def build_lr_scheduler(self):
        lr_scheduler_cfg = self.config['trainer']['lr_scheduler']
        self.lr_scheduler = build_learning_rate_scheduler(lr_scheduler_cfg, self.optimizer)

    def build_tokenizer(self):
        tokenizer = build_tokenizer(self.config['tokenizer'])
        self.tokenizer = tokenizer
        self.tokenizer.tokenizer_path = self.config["tokenizer"]["kwargs"]["tokenizer_name_or_path"]
        self.tokenizer.model_max_length = self.config["tokenization"]["kwargs"].get("max_seq_length", 4096)

        # token_list = [IMG_START_TOKEN, IMG_END_TOKEN,
        #               BOX_START_TOKEN, BOX_END_TOKEN,
        #               REF_START_TOKEN, REF_END_TOKEN,
        #               REL_START_TOKEN, REL_END_TOKEN,
        #               ] + [IMG_CONTEXT_TOKEN, ]  # ensure the last token is IMG_CONTEXT_TOKEN
        token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                      QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                      REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
        num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)
        self.num_new_tokens = num_new_tokens
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def build_model(self):
        model = build_model(self.config['model'])
        model.img_context_token_id = self.img_context_token_id
        force_image_size = self.config['runtime'].get('force_image_size', 448)
        patch_size = self.config['runtime'].get('patch_size', 14)
        down_sample_ratio = self.config['runtime'].get('down_sample_ratio', 0.5)
        model.num_image_token = int((force_image_size // patch_size) ** 2 * (down_sample_ratio ** 2))

        num_new_tokens = self.num_new_tokens
        if num_new_tokens > 0 or model.language_model.vocab_size != len(self.tokenizer):
            model.language_model.resize_token_embeddings(len(self.tokenizer))
            output_embeddings = model.language_model.get_output_embeddings().weight.data
            if num_new_tokens > 0:
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            model.config.llm_config.vocab_size = len(self.tokenizer)
            model.language_model.config.vocab_size = len(self.tokenizer)

        # if force_image_size != model.config.vision_config.image_size:

        model.language_model.config.use_cache = False
        if self.config['runtime'].get('vit_ckpt', False):
            model.vision_model.gradient_checkpointing = True
            model.vision_model.encoder.gradient_checkpointing = True
        if self.config['runtime'].get('llm_ckpt', False):
            model.language_model._set_gradient_checkpointing()

        def _freeze_params(module):
            for param in module.parameters():
                param.requires_grad = False

        if self.config['runtime'].get('freeze_vit', False):
            model.vision_model = model.vision_model.eval()
            _freeze_params(model.vision_model)
        if self.config['runtime'].get('freeze_llm', False):
            model.language_model = model.language_model.eval()
            _freeze_params(model.language_model)
        if self.config['runtime'].get('unfreeze_lm_head', False):
            try:
                model.language_model.lm_head.requires_grad = True
            except Exception:
                model.language_model.output.requires_grad = True
        if self.config['runtime'].get('freeze_mlp', False):
            _freeze_params(model.mlp1)

        self.model = model
        torch.cuda.empty_cache()
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
            mpu=self.mpu
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
        if batch.get("pixel_values", None) is not None:
            batch['pixel_values'] = batch["pixel_values"].to(device=torch.device("cuda"))
        if batch.get("image_flags", None) is not None:
            batch["image_flags"] = batch["image_flags"].to(device=torch.device("cuda"))

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
        if "position_ids" not in batch:
            batch["position_ids"] = None
        if "cu_seqlens" not in batch:
            batch["cu_seqlens"] = None

        # sp_group = get_sequence_parallel_group()
        # for key in batch.keys():
        #     if key in ('input_ids', 'labels', 'position_ids') and batch[key] is not None:
        #         batch[key] = split_for_sequence_parallel(batch[key], dim=1, sp_group=sp_group)
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
            torch.cuda.empty_cache()
            self.cur_iter = iteration // self.gradient_accumulation_steps
            batch = self.get_batch()
            self._hooks('before_train_iter', self.cur_iter, batch)
            with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
                if batch["position_ids"] is not None and batch["cu_seqlens"] is not None:
                    output = self.model(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        labels=batch['labels'],
                                        return_dict=True,
                                        use_cache=False,
                                        position_ids=batch["position_ids"],
                                        cu_seqlens=batch["cu_seqlens"],
                                        pixel_values=batch['pixel_values'].to(self.dtype),
                                        image_flags=batch['image_flags'])
                else:
                    output = self.model(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        labels=batch['labels'],
                                        pixel_values=batch['pixel_values'].to(self.dtype),
                                        image_flags=batch['image_flags'],
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
        if self.config['deepspeed']['config']["zero_optimization"]["stage"] == 3:
            state_dict = self.model._zero3_consolidated_16bit_state_dict()
            save_hf_checkpoint(self, self.config['saver'], self.train_iters, state_dict=state_dict)
        else:
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

    def save_checkpoint(self, save_cfg, global_step, start_dict=None):
        if save_cfg.get('enabled', True):
            ds_config = self.config['deepspeed']['config']
            if ds_config["zero_optimization"]["stage"] == 3:
                # if self.model.zero_gather_16bit_weights_on_model_save():
                state_dict = self.model._zero3_consolidated_16bit_state_dict()
            save_path = save_cfg.get('save_path', "checkpoints")
            assert save_path is not None, "Save path must be provided!!!"
            save_mode = save_cfg.get('save_mode', 'deepspeed')
            if save_mode == 'huggingface':
                save_hf_checkpoint(self, save_cfg, global_step, state_dict=state_dict)
            elif save_mode == 'deepspeed':
                save_ds_checkpoints(self, save_cfg, global_step)
            else:
                raise NotImplementedError


def main(args):
    cfg = load_yaml(args.config)
    cfg['runtime'] = cfg.setdefault('runtime', {})
    if not args.inference:
        runner = MLLMHFRunner(args, cfg, training=True)
        runner.train()
    else:
        runner = MLLMHFRunner(args, cfg, training=False)
        runner.infer()


if __name__ == "__main__":
    args = parse_args()
    setup_distributed(launcher=args.launcher, backend=args.distributed_backend, port=args.port)
    main(args)
