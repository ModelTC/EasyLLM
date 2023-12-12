import torch

from llm.utils.env import dist_env

from ..base_modules.modules.enums import PositionEmbeddingType
from ..base_modules.modules.meg_module import MegatronModule
from ..base_modules.layers import VocabParallelEmbedding
from ..base_modules.utils import get_initializer_from_cfg, get_torch_dtype, get_position_embedding_type


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 embedding_dropout_prob,
                 num_tokentypes=0,
                 position_embedding_type=None,
                 max_position_embeddings=None,
                 params_dtype=None,
                 pretrain_causal_attention=False,
                 use_bnb_optimizer=False,
                 use_cpu_initialization=False,
                 initializer=None,
                 sequence_parallel=False):  # getattr
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = get_initializer_from_cfg(initializer)
        self.num_tokentypes = num_tokentypes
        self.position_embedding_type = get_position_embedding_type(position_embedding_type)
        self.pretrain_causal_attention = pretrain_causal_attention

        # Word embeddings (parallel).
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size, self.hidden_size,
            init_method=self.init_method,
            use_bnb_optimizer=use_bnb_optimizer, use_cpu_initialization=use_cpu_initialization,
            params_dtype=get_torch_dtype(params_dtype))
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            assert max_position_embeddings is not None
            self.position_embeddings = torch.nn.Embedding(
                max_position_embeddings, self.hidden_size)
            self._position_embeddings_key = 'position_embeddings'
            # Initialize the position embeddings.
            self.init_method(self.position_embeddings.weight)
        else:
            self.position_embeddings = None

        self.sequence_parallel = sequence_parallel
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings

        if self.position_embedding_type == PositionEmbeddingType.absolute:
            assert self.position_embeddings is not None
            embeddings = embeddings + self.position_embeddings(position_ids)
        else:
            assert self.position_embeddings is None

        # Dropout.
        if self.sequence_parallel:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose(0, 1).contiguous()
            embeddings = dist_env.scatter_to_sequence_parallel_region(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        if self.sequence_parallel:
            # set to [s b h] --> [b s h].
            embeddings = embeddings.transpose(0, 1).contiguous()

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            state_dict_[self._position_embeddings_key] \
                = self.position_embeddings.state_dict(
                    destination, prefix, keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(
                    destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self.position_embedding_type == PositionEmbeddingType.absolute:
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'position_embeddings' in key:
                        state_dict_[key.split('position_embeddings.')[1]] \
                            = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)


class EmbeddingPipe(Embedding):

    def forward(self, inputs, **kwargs):

        input_ids = inputs[0]
        position_ids = inputs[1]
        attention_mask = inputs[2]

        if len(inputs) == 4:
            cu_seqlens = inputs[3]

        embeddings = super().forward(input_ids,
                                     position_ids)

        if len(inputs) == 4:
            return embeddings, attention_mask, cu_seqlens, position_ids
        else:
            return embeddings

    @property
    def word_embeddings_weight(self):
        """Easy accessory for the DeepSpeed pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight
