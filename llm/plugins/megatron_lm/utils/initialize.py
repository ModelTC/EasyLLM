import torch

from megatron.training.initialize import (
    get_args,
    setup_logging,
    _initialize_distributed,
    _set_random_seed,
    _init_autoresume,
    _compile_dependencies,
    _initialize_tp_communicators
)
from megatron.core import mpu
from megatron.training.checkpointing import load_args_from_checkpoint


def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    # Parse arguments
    # args = parse_args(extra_args_provider, ignore_unknown_args)
    args = get_args()

    # Prep for checkpoint conversion.
    if args.ckpt_convert_format is not None:
        assert args.ckpt_convert_save is not None
        assert args.load is not None
        args.exit_on_missing_checkpoint = True

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoint-args requires --load argument"
        load_args_from_checkpoint(args)

    # if args.yaml_cfg is not None:
    #     args = validate_yaml(args, args_defaults)
    # else:
    #     validate_args(args, args_defaults)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    # set_global_variables(args, build_tokenizer=False)

    # set logging level
    setup_logging()

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            _initialize_tp_communicators()

        # No continuation function
        return None
