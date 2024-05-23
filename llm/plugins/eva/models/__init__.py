from .eva import _EVA_MODELS
from .eva import eva_custom, eva_mini, eva_1b_20b, eva_4b_20b, eva_8b_20b, eva_18b_20b  # noqa

from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)


_ALL_BASE_MODELS = {}
for key in _EVA_MODELS:
    _ALL_BASE_MODELS[key] = _EVA_MODELS[key]


def get_layer_info(cfg_model):
    model_type = cfg_model['type']
    model_kwargs = cfg_model.get('kwargs', {})
    if model_type in _ALL_BASE_MODELS:
        num_layers = _ALL_BASE_MODELS[model_type]['num_layers']
    else:
        assert model_kwargs.get('num_layers', None)
        num_layers = model_kwargs['num_layers']
    checkpoint_num_layers = model_kwargs.get('checkpoint_num_layers', 1)
    return num_layers, checkpoint_num_layers
