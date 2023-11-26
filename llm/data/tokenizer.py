from transformers import AutoTokenizer
from transformers import LlamaTokenizer
from transformers import LlamaTokenizerFast
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.general.registry_factory import TOKENIZER_REGISTRY


def build_tokenizer(cfg_tokenizer):
    pad_vocab_size_to = cfg_tokenizer.pop('pad_vocab_size_to', None)
    type = cfg_tokenizer['type']
    tokenizer_name_or_path = cfg_tokenizer['kwargs'].pop('tokenizer_name_or_path')
    tokenizer = TOKENIZER_REGISTRY[type].from_pretrained(tokenizer_name_or_path, **cfg_tokenizer['kwargs'])
    if 'special_tokens' in cfg_tokenizer:
        special_tokens = cfg_tokenizer.get('special_tokens')
        tokenizer.add_special_tokens(special_tokens)
    # Add vocab size.
    padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                 pad_vocab_size_to)
    setattr(tokenizer, 'padded_vocab_size', padded_vocab_size)
    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, pad_vocab_size_to=None):
    """Apply the requested rules to change the size of the vocabulary"""
    if pad_vocab_size_to is not None:
        if pad_vocab_size_to < orig_vocab_size:
            raise ValueError(
                f"You asked to pad the vocabulary to {pad_vocab_size_to} when the initial vocabulary size is "
                f"{orig_vocab_size}. You can only pad to a higher value."
            )
        after = pad_vocab_size_to
    else:
        after = orig_vocab_size
    logger.info('> padded vocab (size: {}) with {} dummy tokens (new size: {})'.format(orig_vocab_size, after - orig_vocab_size, after))        # noqa
    return after


TOKENIZER_REGISTRY.register("AutoTokenizer", AutoTokenizer)
TOKENIZER_REGISTRY.register("LlamaTokenizer", LlamaTokenizer)
TOKENIZER_REGISTRY.register("LlamaTokenizerFast", LlamaTokenizerFast)
