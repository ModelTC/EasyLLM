import torch
from llm.utils.env import dist_env


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self,
                 share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings
        # self.args = args

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)

    def word_embeddings_weight(self):
        if dist_env.is_pipeline_first_stage(ignore_virtual=True):
            return self.language_model.embedding.word_embeddings.weight
        if dist_env.is_pipeline_last_stage(ignore_virtual=True):
            if not self.share_word_embeddings:
                raise Exception('word_embeddings_weight() called for last '
                                'stage, but share_word_embeddings is false')
            return self.word_embeddings.weight
        raise Exception('word_embeddings_weight() should be '
                        'called for first and last stage only')
