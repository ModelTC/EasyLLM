from ..utils import recursive_getattr, recursive_setattr
from .layers import LoraVocabParallelEmbedding, LoraRowParallelLinear, LoraColumnParallelLinear


def convert_layer_to_lora(model, cfg_lora):
    target_modules = cfg_lora.get('target_modules', [])
    column_linear_target_modules = ["gate_proj", "up_proj", "q_proj", "k_proj", "v_proj"]
    row_linear_target_modules = ["down_proj", "o_proj"]
    vocab_target_modules = ["word_embeddings", "lm_head"]
    all_support_modules = column_linear_target_modules + row_linear_target_modules + vocab_target_modules
    for tar_mo in target_modules:
        if tar_mo not in all_support_modules:
            raise ValueError("the module {} is not support for lora now".format(tar_mo))

    word_embeding_replace_part = []
    lm_head_replace_part = []
    column_linear_replace_part = []
    row_linear_replace_part = []

    for name, module in model.named_modules():
        if ('word_embeddings' in name):
            if ('word_embeddings' in target_modules) and (name.split(".")[0] == "1"):
                word_embeding_replace_part.append(name)
            if ('lm_head' in target_modules) and (name.split(".")[0] != "1"):
                lm_head_replace_part.append(name)
        for cltm in column_linear_target_modules:
            if cltm in name:
                column_linear_replace_part.append(name)
        for rltm in row_linear_target_modules:
            if rltm in name:
                row_linear_replace_part.append(name)

    lora_params = {'lora_rank': cfg_lora['lora_rank'],
                   'lora_alpha': cfg_lora['lora_alpha'],
                   'lora_dropout': cfg_lora['lora_dropout'],
                   'lora_sync_tp_duplicated_parameters': cfg_lora.get('lora_sync_tp_duplicated_parameters', True)}

    all_replace_part = word_embeding_replace_part + lm_head_replace_part + \
        column_linear_replace_part + row_linear_replace_part
    for name in all_replace_part:
        module = recursive_getattr(model, name)
        if (name in word_embeding_replace_part) or (name in lm_head_replace_part):
            tmp = LoraVocabParallelEmbedding(
                module.num_embeddings, module.embedding_dim, module.init_method,
                module.use_bnb_optimizer, module.use_cpu_initialization,
                module.params_dtype, **lora_params)
        elif (name in column_linear_replace_part):
            tmp = LoraColumnParallelLinear(
                module.input_size, module.output_size, module.use_bias,
                module.gather_output, module.init_method, module.stride,
                module.keep_master_weight_for_test, module.skip_bias_add,
                module.use_cpu_initialization, module.params_dtype,
                module.sequence_parallel, **lora_params)
        elif (name in row_linear_replace_part):
            tmp = LoraRowParallelLinear(
                module.input_size, module.output_size, module.use_bias,
                module.input_is_parallel, module.init_method, module.stride,
                module.keep_master_weight_for_test, module.skip_bias_add,
                module.use_cpu_initialization, module.params_dtype,
                module.bias_tp_auto_sync, module.sequence_parallel, **lora_params)
        else:
            raise ValueError
        recursive_setattr(model, name, tmp)
    return model
