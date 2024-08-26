import argparse
import deepspeed

from llm.utils.general.parser_helper import (
    _add_training_args,
    _add_inference_args,
    _add_medusa_args,
    _add_distributed_args
)


def parse_args(ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_training_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_medusa_args(parser)
    parser = _add_distributed_args(parser)
    parser = deepspeed.add_config_arguments(parser)

    # args = parser.parse_args()
    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    return args
