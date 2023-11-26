from .llama import (llama_custom,        # noqa
                    llama_7b,
                    llama_13b,
                    llama_20b,
                    llama_65b,
                    llama2_7b,
                    llama2_13b,
                    llama2_70b,
                    codellama_7b,
                    codellama_13b,
                    codellama_34b)        # noqa

from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        MODULE_ZOO_REGISTRY.register(var_name, var)

from .losses import *       # noqa
