import time
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from llm.utils.general.parser_helper import parse_args
from llm.utils.general.yaml_loader import load_yaml
from llm.runners.base_llm_runner import BaseRunner
from llm.utils.model.optimizer_helper import build_optimizer
from llm.plugins.eva.models.eva import _EVA_MODELS
from llm.utils.env import (get_distributed_info, initialize_distributed,
                           setup_deepspeed_random_and_activation_checkpointing,
                           set_logging_verbosity,
                           set_random_seed)
from llm.utils.general.log_helper import default_logger as logger

from llm.plugins.eva.models import get_layer_info
from llm.models.mg_models import unwrap_model, Float16Module
from llm.utils.model.lr_helper import build_learning_rate_scheduler

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

_TRAIN_START_TIME = time.time()


class EVABaseRunner(BaseRunner):
    def build(self):
        super().build()
        self.profile_path = self.config['model']['kwargs'].get("profile_path", None)
        if self.training and self.config['trainer']['lr_scheduler']["kwargs"].get("training_steps", None) is None:
            if not isinstance(self.lr_scheduler.lr_lambdas, list):
                self.lr_scheduler.lr_lambdas.keywords['training_steps'] = self.total_train_iters
            else:
                for idx in range(len(self.lr_scheduler.lr_lambdas)):
                    self.lr_scheduler.lr_lambdas[idx].keywords['training_steps'] = self.total_train_iters

    def build_env(self, rank=None, local_rank=None):
        cfg_runtime = self.config['runtime']
        # get env info
        rank, local_rank, world_size, tensor_model_parallel_size, \
            pipeline_model_parallel_size = get_distributed_info(cfg_runtime, self.args.launcher, self.args.port)
        # initialize env
        # Pytorch distributed.
        initialize_distributed(rank, local_rank, world_size, tensor_model_parallel_size,
                               pipeline_model_parallel_size, cfg_runtime.get('distributed_backend', 'nccl'),
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

    def build_tokenizer(self):
        # self.tokenizer = build_tokenizer(self.config['tokenizer'])
        super().build_tokenizer()
        # add special token
        # self.tokenizer.pad_token_id = 0
        # if self.tokenizer.unk_token is None:
        #     self.tokenizer.add_special_tokens({"unk_token": UNK_TOKEN})

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
        num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)  # noqa

    def build_data_engine(self):
        # self.config['data']['train']['dataset']['kwargs']['tokenizer'] = self.tokenizer
        if self.config["data"]["train"]["dataset"]["type"] == "internvl":
            self.config['data']['train']['dataset']['kwargs']['num_image_token'] = int((self.config['runtime']['force_image_size'] // self.config['runtime']['patch_size']) ** 2 * (self.config['runtime']['down_sample_ratio'] ** 2))  # noqa
        elif "packed" in self.config["data"]["train"]["dataset"]["type"] and self.config["data"]["train"]["dataset"]['kwargs']["dataset"]["type"] == "internvl":
            self.config['data']['train']['dataset']['kwargs']["dataset"]['kwargs']['num_image_token'] = int((self.config['runtime']['force_image_size'] // self.config['runtime']['patch_size']) ** 2 * (self.config['runtime']['down_sample_ratio'] ** 2))  # noqa
        super().build_data_engine()

    def _freeze_params(self, module, is_eval=False):
        if is_eval:
            module = module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_params(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def build_model(self):
        force_image_size = self.config["runtime"].get("force_image_size", 224)
        if force_image_size != 224:
            self.config["model"]["kwargs"]["vision_image_size"] = force_image_size
        # setting img_context_token_id
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.config["model"]["kwargs"]["img_context_token_id"] = img_context_token_id
        super().build_model()
        model_type = self.config["model"]["type"]
        freeze_vit = self.config["runtime"].get("freeze_vit", False)
        freeze_llm = self.config["runtime"].get("freeze_llm", False)
        unfreeze_lm_head = self.config["runtime"].get("unfreeze_lm_head", False)
        freeze_mlp = self.config["runtime"].get("freeze_mlp", False)
        for name, child in self.model.named_children():
            if freeze_vit and name == "tied_modules":
                self._freeze_params(child, is_eval=True)
            elif freeze_vit and int(name) < _EVA_MODELS[model_type]["num_eva_layers"] + 2:  # freeze vit
                self._freeze_params(child, is_eval=True)
            elif freeze_llm and int(name) >= _EVA_MODELS[model_type]["num_eva_layers"] + 3 and int(name) < _EVA_MODELS[model_type]["num_eva_layers"] + _EVA_MODELS[model_type]["num_layers"] + 7:  # noqa
                self._freeze_params(child, is_eval=True)
            elif freeze_mlp and int(name) == _EVA_MODELS[model_type]["num_eva_layers"] + 2:
                self._freeze_params(child)
        for name, child in self.model.named_children():
            if name == "tied_modules":
                continue
            elif unfreeze_lm_head and int(name) == _EVA_MODELS[model_type]["num_eva_layers"] + _EVA_MODELS[model_type]["num_layers"] + 6:
                self._unfreeze_params(child)

    def build_trainer(self):
        unwrapped_model = unwrap_model(self.model, (torchDDP, Float16Module))
        if self.training:
            cfg_optim = self.config['trainer']['optimizer']
            optimizer = build_optimizer(cfg_optim, unwrapped_model, deepspeed=self.deepspeed)
            cfg_lr_scheduler = self.config['trainer']['lr_scheduler']
            cfg_lr_scheduler['kwargs']['max_lr'] = cfg_optim['kwargs']['lr']        # noqa
            if cfg_lr_scheduler['type'] == 'iter_base_annealing':
                cfg_lr_scheduler['kwargs']['global_batch_size'] = self.num_microbatches_calculator.global_batch_size
            lr_scheduler = build_learning_rate_scheduler(cfg_lr_scheduler, optimizer)
        else:
            optimizer = None
            lr_scheduler = None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


def main():
    args = parse_args()
    assert args.config is not None, 'please provide a config file'
    cfg = load_yaml(args.config)
    if args.profile_path is not None:
        cfg['model']['kwargs']['profile_path'] = args.profile_path
    if args.layer_profile:
        cfg['model']['kwargs']['layer_profile'] = args.layer_profile
    if args.pp_method is not None:
        cfg['model']['kwargs']['pp_partition_method'] = args.pp_method

    print(f'training cfg: {cfg}')
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
        runner = EVABaseRunner(args, cfg, training=False, base_type='infer')
        runner.generate()
    else:
        runner = EVABaseRunner(args, cfg, training=True, base_type='train')
        runner.train()


if __name__ == "__main__":
    main()
