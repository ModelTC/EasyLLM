import torch

from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from deepspeed.runtime import utils as ds_utils

from llm.utils.env import dist_env
from llm.models.mg_models.base_modules.modules.meg_module import MegatronModule
from llm.plugins.internvl.models.mg_models.word_embedings import EmbeddingPipe
# from llm.models.mg_models.llama.lm_head import EmbedddingPipeNoTied
from llm.plugins.internvl.models.mg_models.lm_head import EmbedddingPipeNoTied
# from llm.models.mg_models.llama.transformer import ParallelTransformerLayerPipe
from llm.plugins.internvl.models.mg_models.transformer import ParallelTransformerLayerPipe
from llm.models.mg_models.base_modules.layers.fused_layer_norm import build_layer_norm

from llm.models.mg_models.base_modules.modules.fp16_module import float16_to_fp32, fp32_to_float16
# from .losses import get_cross_entropy
from .default_cfg import update_model_cfg
from llm.plugins.eva.models.utils import load_lora_ckpt_pretrained, load_ckpt_pretrained, save_lora_ckpt_pretrained
from llm.plugins.eva.models.utils import set_train_params, set_train_status
from llm.plugins.eva.models.utils import measure_time, dist_save_obj_to_json
from llm.utils.general.registry_factory import LOSS_REGISTRY

from .vision_embeddings import VisionEmbeddings
from .vision_transformer import ParallelVisionTransformerLayerPipe
from .vision_extract_feat import VisionExtractFeat
from llm.utils.general.log_helper import default_logger as logger
import torch.nn as nn
import torch.distributed as dist
import os
try:
    from pynvml import *
except BaseException:
    pass


def get_time():
    import time
    torch.cuda.synchronize()
    return time.time()


class EVAModelPipe(PipelineModule, MegatronModule):
    """
        LLaMA model.
        NOTE: The name of this class has to be kept as GPTModelPipe.
        I don't know why, but it is used in the code.
    """

    def __init__(
        self,
        num_vit_layers,
        num_layers,
        parallel_output=True,
        fp16: bool = False,
        bf16: bool = True,
        fp32_residual_connection: bool = False,
        pretrain_causal_attention: bool = False,
        checkpoint_activations: bool = True,
        checkpoint_num_layers: int = 1,
        pp_partition_method: str = 'type:transformer|embedding',
        sequence_parallel=False,
        dynamic_checkpoint=None,
        word_embedings_params=None,
        transformer_layer_params=None,
        layer_norm_params=None,
        lm_head_params=None,
        loss_params=None,
        vision_embedings_params=None,
        vision_transformer_layer_params=None,
        vision_extract_feat_params=None,
        profile_path=None,
        layer_profile=False
    ):

        self.parallel_output = parallel_output
        self.sequence_parallel = sequence_parallel
        # set model args
        self._set_model_kwargs(num_vit_layers, num_layers, checkpoint_activations)
        self.word_embedings_params = word_embedings_params
        self.transformer_layer_params = transformer_layer_params
        self.layer_norm_params = layer_norm_params
        self.lm_head_params = lm_head_params
        self.loss_params = loss_params
        self.vision_embedings_params = vision_embedings_params
        self.vision_transformer_layer_params = vision_transformer_layer_params
        self.vision_extract_feat_params = vision_extract_feat_params

        self.specs = self.build_specs(num_vit_layers, num_layers, fp16, bf16, fp32_residual_connection, pretrain_causal_attention)
        self.loss_fn = LOSS_REGISTRY.build(self.loss_params)
        self.img_size = self.vision_embedings_params['image_size']
        self.patch_size = self.vision_embedings_params['patch_size']

        if checkpoint_activations:
            interval = checkpoint_num_layers
        else:
            interval = 0
        topo = PipeModelDataParallelTopology(num_pp=dist_env.get_pipeline_model_parallel_world_size(),
                                             num_mp=dist_env.get_tensor_model_parallel_world_size(),
                                             num_dp=dist_env.get_data_parallel_world_size())

        # here one can extend the regex to include more layers to be counted towards partitioning,
        # e.g. 'type:transformer|embedding' will add up all the transformer blocks and also the first
        # and last embedding layers and then partition that transformers+2 layers - so to get a good
        # balance you may want to use less transformer layers
        #
        # caveat emptor: the current implementation of PP fails unless each stage has at least one
        # transformer layer
        if pp_partition_method is not None:
            partition_method = pp_partition_method
        else:
            partition_method = 'type:transformer'

        self.profile_path = profile_path
        self.layer_profile = layer_profile

        super().__init__(layers=self.specs,
                         loss_fn=self.loss_fn,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method=partition_method)

        self.size_map = None
        self.skip_checkpoint_layer_range = -1
        self.dc_profile = None
        if dynamic_checkpoint is not None:
            if dynamic_checkpoint['enabled']:
                self.size_map = dynamic_checkpoint['size_map']
                if 'dc_profile' in dynamic_checkpoint:
                    if dynamic_checkpoint['dc_profile'].get('enabled', True):
                        self.dc_profile = dynamic_checkpoint['dc_profile']
        pp_rank = dist_env.get_pipeline_model_parallel_rank()
        pp_size = dist_env.get_pipeline_model_parallel_world_size()
        self.checkpoint_list = []
        self.ckpt_module_set = ['ParallelTransformerLayerPipe', 'ParallelVisionTransformerLayerPipe']
        self.ckpt_module_exclude = ['EmbeddingPipe']
        for i in range(pp_size):
            if pp_rank == i:
                for j in range(self.parts[i], self.parts[i + 1]):
                    name = str(self._layer_specs[j])
                    flag = False
                    for k in self.ckpt_module_set:
                        if k in name:
                            flag = True
                    if flag:
                        self.checkpoint_list.append(j)
        if self.profile_path is not None:
            if dist_env.get_global_rank() == 0:
                os.system(f"mkdir -p {self.profile_path}")
                import json
                with open(os.path.join(self.profile_path, "pp_parts.txt"), "w") as f:
                    print(json.dumps(self.parts), file=f, flush=True)
        self.layer_profile_info = []
        self.pp_profile = []
        if self.dc_profile is not None:
            self.get_vit_llm_checkpoint_info(num_vit_layers, num_layers)
            self.layer_profile = False

    def get_vit_llm_checkpoint_info(self, num_vit_layer, num_llm_layer):
        # (to_bfloat16) + (vit embedding) + num_vit_layer + (mlp) + llm embedding + num_llm_layer + other 4 layer  # noqa
        self.pp_checkpoint_num = []
        self.pp_vit_llm_layer_num = []
        vit_layer_set = set()
        llm_layer_set = set()
        # get different stage vit and llm layer num
        for i in range(num_vit_layer + num_llm_layer + 8):
            if i < 2:
                continue
            elif i < 2 + num_vit_layer:
                vit_layer_set.add(i)
            elif i < 4 + num_vit_layer:
                continue
            elif i < 4 + num_llm_layer + num_vit_layer:
                llm_layer_set.add(i)
            else:
                continue
        for i in range(0, len(self.parts) - 1):
            vit_num = 0
            llm_num = 0
            for j in range(self.parts[i], self.parts[i + 1]):
                if j in vit_layer_set:
                    vit_num += 1
                if j in llm_layer_set:
                    llm_num += 1
            self.pp_vit_llm_layer_num.append([vit_num, llm_num])
            self.pp_checkpoint_num.append(vit_num + llm_num)

        self.pp_status = 0
        self.memory_cost_vit_llm = [0, 0]
        self.warmup_iter = self.dc_profile.get("warmup_iter", 5)
        global_batch_size = self.dc_profile['global_batch_size']
        self.micro_batches = global_batch_size // dist_env.get_data_parallel_world_size()
        self.min_free_memory = 1000 * 1024

    def _is_checkpointable(self, funcs):
        # return False
        for f in funcs:
            flag = True
            if f.__class__.__name__ in self.ckpt_module_set:
                if self.size_map is None or self.skip_checkpoint_layer_range < 0:
                    return flag
                if hasattr(f, "layer_number") or hasattr(f, "vision_layer_number"):
                    if hasattr(f, "layer_number"):
                        layer_number = f.layer_number
                    if hasattr(f, "vision_layer_number"):
                        layer_number = f.vision_layer_number
                    if layer_number in self.checkpoint_list:
                        index = self.checkpoint_list.index(layer_number)
                        flag &= index < self.skip_checkpoint_layer_range
                else:
                    if f.__class__.__name__ in self.ckpt_module_exclude:
                        flag = False
            else:
                flag = False
        return flag

    def get_checkpoint_range(self, seq_len):
        size_list = sorted(list(self.size_map.keys()))
        for item in size_list:
            if seq_len <= item:
                range_size = self.size_map[item]
                if isinstance(range_size, list):
                    pp_rank = dist_env.get_pipeline_model_parallel_rank()
                    range_size = range_size[pp_rank]
                return range_size
        return -1

    def get_layer_idx(self, start_idx):
        pp_rank = dist_env.get_pipeline_model_parallel_rank()
        return start_idx + self.parts[pp_rank]

    def save_profile(self):
        if self.profile_path is not None:
            if self.layer_profile:
                for item in self.layer_profile_info:
                    dist_save_obj_to_json(item, 'layer_time', self.profile_path)
            pp_rank = dist_env.get_pipeline_model_parallel_rank()
            if self.dc_profile is None:
                for info in self.pp_profile:
                    import json
                    with open(os.path.join(self.profile_path, f"pp_stage_{pp_rank}.txt"), "a") as f:
                        print(json.dumps(info), file=f, flush=True)

    def get_seq_len(self, forward_input):
        # first stage
        # if dist_env.get_pipeline_model_parallel_rank() == 0:
        #     seq_len = forward_input[0].shape[1] * forward_input[0].shape[0]
        # else:
        if len(forward_input[0].shape) <= 1:
            seq_len = 1
            return seq_len

        if len(forward_input) == 7 and len(forward_input[-1].shape) == 2:
            if len(forward_input[0].shape) == 4:
                seq_len = forward_input[0].shape[1] * forward_input[0].shape[2]
            elif len(forward_input[0].shape) == 3:
                seq_len = forward_input[0].shape[0] * forward_input[0].shape[1]
            else:
                seq_len = ((self.img_size // self.patch_size) ** 2 + 1) * forward_input[-2].shape[0]
        elif len(forward_input) == 6:
            if len(forward_input[0].shape) == 4:
                seq_len = forward_input[0].shape[1] * forward_input[0].shape[2]
            elif len(forward_input[0].shape) == 3:
                seq_len = forward_input[0].shape[0] * forward_input[0].shape[1]
            else:
                seq_len = ((self.img_size // self.patch_size) ** 2 + 1) * forward_input[-1].shape[0]
        elif len(forward_input) in (3, 4, 5):
            if self.sequence_parallel:
                seq_len = forward_input[0].shape[0] * forward_input[0].shape[1] * dist_env.get_tensor_model_parallel_world_size()
            else:
                seq_len = forward_input[0].shape[0] * forward_input[0].shape[1]

        return seq_len

    def get_vit_llm_nockpt_num(self, ckpt_num, pp_rank=0):
        vit_nockpt_num = 0
        llm_nockpt_num = 0
        vit_num, llm_num = self.pp_vit_llm_layer_num[pp_rank]
        if pp_rank == 0:
            if ckpt_num >= vit_num:
                llm_nockpt_num = max(0, vit_num + llm_num - ckpt_num)
            else:
                llm_nockpt_num = llm_num
                vit_nockpt_num = max(0, vit_num - ckpt_num)
        if pp_rank == - 1:
            if ckpt_num <= vit_num:
                llm_nockpt_num = llm_num
                vit_nockpt_num = max(0, vit_num - ckpt_num)
            else:
                llm_nockpt_num = max(0, vit_num + llm_num - ckpt_num)
        return vit_nockpt_num, llm_nockpt_num

    def get_vit_llm_memory_cost(self):
        tp_rank = dist_env.get_tensor_model_parallel_rank()
        dp_rank = dist_env.get_data_parallel_rank()
        pp_rank = dist_env.get_pipeline_model_parallel_rank()
        world_size = dist_env.get_pipeline_model_parallel_world_size()
        pp_profile_list = []
        import json
        for rank in [0, world_size - 1]:
            with open(os.path.join(self.profile_path, f"pp_stage_{rank}.txt")) as f:
                temp = []
                for line in f.readlines():
                    try:
                        temp.append(json.loads(line))
                    except: # noqa
                        pass
            pp_profile_list.append(temp)

        cost_dict1 = {}
        for item in pp_profile_list[0]:
            if item['ckpt_num'] not in cost_dict1:
                cost_dict1[item['ckpt_num']] = []
            if item['tp'] == tp_rank and item['dp'] == dp_rank:
                cost_dict1[item['ckpt_num']].append(item['free_memory'])

        keys = sorted(list(cost_dict1.keys()))
        cost_memory1 = float(cost_dict1[keys[-1]][-1] - cost_dict1[keys[0]][-1])

        vit_nockpt_num1, llm_nockpt_num1 = self.get_vit_llm_nockpt_num(keys[0], 0)

        cost_dict2 = {}
        for item in pp_profile_list[-1]:
            if item['ckpt_num'] not in cost_dict2:
                cost_dict2[item['ckpt_num']] = []
            if item['tp'] == tp_rank and item['dp'] == dp_rank:
                cost_dict2[item['ckpt_num']].append(item['free_memory'])

        keys = sorted(list(cost_dict2.keys()))
        cost_memory2 = float(cost_dict2[keys[-1]][-1] - cost_dict2[keys[0]][-1])
        vit_nockpt_num2, llm_nockpt_num2 = self.get_vit_llm_nockpt_num(keys[0], -1)

        cost_memory1 = torch.FloatTensor([cost_memory1]).cuda()
        dist.all_reduce(cost_memory1, group=dist_env.get_data_parallel_group(), op=dist.ReduceOp.AVG)
        dist.all_reduce(cost_memory1, group=dist_env.get_tensor_model_parallel_group(), op=dist.ReduceOp.AVG)
        cost_memory1 = max(int(cost_memory1.cpu().item()), 10)

        cost_memory2 = torch.FloatTensor([cost_memory2]).cuda()
        dist.all_reduce(cost_memory2, group=dist_env.get_data_parallel_group(), op=dist.ReduceOp.AVG)
        dist.all_reduce(cost_memory2, group=dist_env.get_tensor_model_parallel_group(), op=dist.ReduceOp.AVG)
        cost_memory2 = max(int(cost_memory2.cpu().item()), 10)

        from sympy import Symbol, solve
        x = Symbol('x')
        y = Symbol('y')

        if vit_nockpt_num2 == 0 and llm_nockpt_num2 == 0:
            cost_memory2 = 0

        if vit_nockpt_num1 == 0 and llm_nockpt_num1 == 0:
            cost_memory1 = 0
        
        print(pp_rank, "cost memory info", cost_memory1, cost_memory2, vit_nockpt_num1, llm_nockpt_num1, vit_nockpt_num2, llm_nockpt_num2)
        result = solve([vit_nockpt_num1 * x + llm_nockpt_num1 * y - cost_memory1, vit_nockpt_num2 * x + llm_nockpt_num2 * y - cost_memory2], [x, y])
        vit_cost = self.dc_profile.get('vit_memory_estimate', 2) * 1024
        llm_cost = self.dc_profile.get('llm_memory_estimate', 1) * 1024
        if x in result:
            vit_cost = float(result[x])
        if y in result:
            llm_cost = float(result[y])
        self.memory_cost_vit_llm = [vit_cost, llm_cost]
        if os.path.exists(os.path.join(self.profile_path, f"pp_stage_memory_cost_vit_llm.txt")):
            with open(os.path.join(self.profile_path, f"pp_stage_memory_cost_vit_llm.txt")) as f:
                self.memory_cost_vit_llm = json.loads(f.readlines()[0])
        else:
            with open(os.path.join(self.profile_path, f"pp_stage_memory_cost_vit_llm.txt"), "a") as f:
                print(json.dumps(self.memory_cost_vit_llm), file=f, flush=True)
        print(pp_rank, "vit llm memory cost", self.memory_cost_vit_llm)


    def ave_free_memory(self):
        free_memory_list = []
        for item in self.pp_profile:
            free_memory_list.append(item['free_memory'])
        free_memory = min(free_memory_list)

        gather_dp = [None for _ in range(dist_env.get_data_parallel_world_size())]
        dist.all_gather_object(gather_dp, free_memory, group=dist_env.get_data_parallel_group())

        gather_tp = [None for _ in range(dist_env.get_tensor_model_parallel_world_size())]
        dist.all_gather_object(gather_tp, gather_dp, group=dist_env.get_tensor_model_parallel_group())

        min_free_memory = 1024 * 1000
        for i in gather_tp:
            min_free_memory = min(min(i), min_free_memory)

        return min_free_memory

    def adjust_ckpt_num_warmup(self):
        rank = dist_env.get_pipeline_model_parallel_rank()
        world_size = dist_env.get_pipeline_model_parallel_world_size()
        ckpt_num = self.pp_checkpoint_num[rank]

        step_num = self.warmup_iter // 2 + 1

        if self.micro_offset < self.micro_batches * step_num:
            self.warmup_ckpt_num = ckpt_num
        elif self.micro_offset == self.micro_batches * step_num:
            vit_memory_estimate = self.dc_profile.get('vit_memory_estimate', 2)
            llm_memory_estimate = self.dc_profile.get('llm_memory_estimate', 1)
            free_memory = max(self.ave_free_memory() - self.dc_profile['min_memory'] * 1024, 0)
            if rank == 0:  # for vision
                ckpt_num = self.pp_checkpoint_num[rank] - free_memory // (1024 * vit_memory_estimate)
            if rank == world_size - 1:  # for llm
                ckpt_num = self.pp_checkpoint_num[rank] - free_memory // (1024 * llm_memory_estimate)
            self.warmup_ckpt_num = ckpt_num
        return max(self.warmup_ckpt_num, 1)
    
    def get_nockpt_num(self, free_memory):
        pp_rank = dist_env.get_pipeline_model_parallel_rank()
        vit_num, llm_num = self.pp_vit_llm_layer_num[pp_rank]
        if vit_num == 0:
            return (free_memory // self.memory_cost_vit_llm[1] - 1)
        if llm_num == 0:
            return (free_memory // self.memory_cost_vit_llm[0] - 1)
        if llm_num * self.memory_cost_vit_llm[1] > free_memory:
            return (free_memory // self.memory_cost_vit_llm[1] - 1)
        else:
            temp = free_memory - llm_num * self.memory_cost_vit_llm[1]
            output = temp // self.memory_cost_vit_llm[0] - 1 + llm_num
            return output

    def adjust_ckpt_layer_num(self):
        '''
        adjust ckpt layer from last stage to first stage
        '''
        # all_pp_status = self.gather_all_status()

        pp_rank = dist_env.get_pipeline_model_parallel_rank()
        world_size = dist_env.get_pipeline_model_parallel_world_size()
        predefine_memory = self.dc_profile.get('min_memory', 3) * 1024
        pp_profile_list = []
        import json
        for rank in range(world_size):
            with open(os.path.join(self.profile_path, f"pp_stage_{rank}.txt")) as f:
                temp = []
                for line in f.readlines():
                    try:
                        temp.append(json.loads(line))
                    except: # noqa
                        pass
            pp_profile_list.append(temp)
        free_memory_list = []
        for item in pp_profile_list[pp_rank]:
            if item['ckpt_num'] == self.pp_checkpoint_num[pp_rank]:
                free_memory_list.append(item['free_memory'])
        min_free_memory = min(free_memory_list)

        self.min_free_memory = min(self.min_free_memory, min_free_memory)
        if self.min_free_memory < predefine_memory:
            no_ckpt_num = 0
        else:
            no_ckpt_num = max(self.get_nockpt_num(self.min_free_memory - predefine_memory), 0)
        final_ckpt_num = max(self.pp_checkpoint_num[pp_rank] - no_ckpt_num, 0)
        final_ckpt_num = torch.FloatTensor([final_ckpt_num]).cuda()
        dist.all_reduce(final_ckpt_num, group=dist_env.get_data_parallel_group(), op=dist.ReduceOp.AVG)
        dist.all_reduce(final_ckpt_num, group=dist_env.get_tensor_model_parallel_group(), op=dist.ReduceOp.AVG)
        final_ckpt_num = int(final_ckpt_num.cpu().item())
        return final_ckpt_num

    def forward(self, forward_input):
        # We need to offset the seed by the microbatch ID. Save it in a local var to
        # ensure it is preserved in the closure. Otherwise checkpointed forward funcs
        # will see a different offset.
        pp_rank = dist_env.get_pipeline_model_parallel_rank()
        self.micro_offset += 1
        if self.dc_profile is not None:
            if self.micro_offset == self.warmup_iter * self.micro_batches:
                self.get_vit_llm_memory_cost()

        def exec_range_func(start, end):
            ''' Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            '''
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = (self.base_seed * local_micro_offset) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            ds_utils.set_random_seed(new_seed)

                    inputs = layer(inputs)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.forward_funcs))
            x = func(forward_input)
        else:
            num_layers = len(self.forward_funcs)
            x = forward_input
            if self.profile_path:
                global_rank = dist_env.get_global_rank()
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(global_rank % 8)
                st_time = get_time()
            if self.layer_profile:
                pp_rank = dist_env.get_pipeline_model_parallel_rank()
                # tp_rank = dist_env.get_tensor_model_parallel_rank()
                for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                    end_idx = min(start_idx + self.activation_checkpoint_interval, num_layers)
                    layer_idx = self.get_layer_idx(start_idx)
                    with measure_time(f'pp{pp_rank}_layer{layer_idx}', False) as stats:
                        funcs = self.forward_funcs[start_idx:end_idx]
                        if funcs[0].__class__.__name__ in self.ckpt_module_set:
                            seq_len = self.get_seq_len(x)
                            if self.size_map is not None:
                                self.skip_checkpoint_layer_range = self.get_checkpoint_range(seq_len)
                        else:
                            self.skip_checkpoint_layer_range = 0
                        # Since we either pass tensors or tuples of tensors without unpacking, we
                        # need to be careful not to double-wrap tensors with tuple.
                        if not isinstance(x, tuple):
                            x = (x, )
                        # if hasattr(funcs[0], "vision_layer_number") and funcs[0].vision_layer_number == 2:
                        #     import pdb;pdb.set_trace()
                        if self._is_checkpointable(funcs):
                            # print(funcs[0].vision_layer_number)
                            x = self.activation_checkpoint_func(exec_range_func(start_idx, end_idx), *x)
                        else:
                            x = exec_range_func(start_idx, end_idx)(*x)
                    # dist_save_obj_to_json({'layer_idx': layer_idx, 'time': stats['elapsed_time']}, 'layer_time', self.profile_path)
                    self.layer_profile_info.append({'layer_idx': layer_idx, 'time': stats['elapsed_time']})
            else:
                if self.dc_profile is not None:
                    if self.micro_offset <= self.warmup_iter * self.micro_batches:
                        step_num = self.warmup_iter // 2
                        if self.micro_offset > self.micro_batches * step_num:
                            self.skip_checkpoint_layer_range = self.adjust_ckpt_num_warmup()
                        else:
                            self.skip_checkpoint_layer_range = self.pp_checkpoint_num[pp_rank]
                    else:
                        self.skip_checkpoint_layer_range = self.adjust_ckpt_layer_num()
                    print(pp_rank, self.skip_checkpoint_layer_range, self.micro_offset)
                for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                    end_idx = min(start_idx + self.activation_checkpoint_interval, num_layers)

                    funcs = self.forward_funcs[start_idx:end_idx]
                    if self.dc_profile is None:
                        if funcs[0].__class__.__name__ in self.ckpt_module_set:
                            seq_len = self.get_seq_len(x)
                            if self.size_map is not None:
                                self.skip_checkpoint_layer_range = self.get_checkpoint_range(seq_len)
                        else:
                            self.skip_checkpoint_layer_range = 0
                    # Since we either pass tensors or tuples of tensors without unpacking, we
                    # need to be careful not to double-wrap tensors with tuple.
                    if not isinstance(x, tuple):
                        x = (x, )

                    # if hasattr(funcs[0], "vision_layer_number") and funcs[0].vision_layer_number == 2:
                    #     import pdb;pdb.set_trace()
                    if self._is_checkpointable(funcs):
                        # print(funcs[0].vision_layer_number)
                        x = self.activation_checkpoint_func(exec_range_func(start_idx, end_idx), *x)
                    else:
                        x = exec_range_func(start_idx, end_idx)(*x)
            if self.profile_path:
                end_time = get_time()
                cur_time = end_time - st_time
                if self.dc_profile is not None:
                    if self.micro_offset <= self.micro_batches * self.warmup_iter:
                        torch.cuda.empty_cache()
                        torch.cuda.memory.reset_peak_memory_stats()
                        import time
                        time.sleep(0.1)
                memory_info = nvmlDeviceGetMemoryInfo(handle)
                info = {}
                pp_rank = dist_env.get_pipeline_model_parallel_rank()
                tp_rank = dist_env.get_tensor_model_parallel_rank()
                dp_rank = dist_env.get_data_parallel_rank()
                info['time'] = float(cur_time)
                info['stage'] = int(pp_rank)
                info['tp'] = int(tp_rank)
                info['dp'] = int(dp_rank)
                info['ckpt_num'] = int(self.skip_checkpoint_layer_range)
                info['used_memory'] = memory_info.used // (1024**2)
                info['free_memory'] = memory_info.free // (1024**2)
                if self.dc_profile is not None:
                    if self.micro_offset > self.micro_batches:
                        self.pp_profile.append(info)
                        import json
                        with open(os.path.join(self.profile_path, f"pp_stage_{pp_rank}.txt"), "a") as f:
                            print(json.dumps(info), file=f, flush=True)
                else:
                    self.pp_profile.append(info)
        return x

    def _set_model_kwargs(self, num_vit_layers, num_layers, checkpoint_activations):
        self.model_kwargs = {"num_vit_layers": num_vit_layers, "num_layers": num_layers, "checkpoint_activations": checkpoint_activations}

    def _partition_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages with method {method}')

        method = method.lower()

        # Each stage gets a simple uniform number of layers.
        if method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
            if self.profile_path is not None:
                from .utils import reduce_list_data
                self.parts = reduce_list_data(self.parts)
                dist_save_obj_to_json({'parts': self.parts}, 'meta', self.profile_path)
        elif "manual" in method:
            self.parts = method.split("manual:")[1].split(',')
            self.parts = [int(item) for item in self.parts]
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            self.parts = ds_utils.partition_balanced(weights=binary_weights, num_parts=num_stages)
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    print(f'    {idx+start:2d}: {name}')
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')
        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])
        logger.info(self.parts)

    def build_specs(self, num_vit_layers, num_layers, fp16, bf16, fp32_residual_connection, pretrain_causal_attention):
        specs = []

        def _to_float16(inputs):
            if fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        specs.append(_to_float16)

        specs.append(LayerSpec(VisionEmbeddings, **self.vision_embedings_params))

        # dpr = [0.0 for _ in range(num_intern_layers)]
        for vision_layer_idx in range(num_vit_layers):
            self.vision_transformer_layer_params.update({'vision_layer_number': vision_layer_idx + 2})
            specs.append(LayerSpec(ParallelVisionTransformerLayerPipe, **self.vision_transformer_layer_params))

        specs.append(LayerSpec(VisionExtractFeat, **self.vision_extract_feat_params))

        self.word_embedings_params.update({"fp32_residual_connection": fp32_residual_connection})
        specs.append(LayerSpec(EmbeddingPipe, **self.word_embedings_params))

        for layer_idx in range((num_vit_layers + 4), (num_vit_layers + 4 + num_layers)):
            self.transformer_layer_params.update({'qkv_pack': True})
            self.transformer_layer_params.update({'layer_number': layer_idx})
            specs.append(LayerSpec(ParallelTransformerLayerPipe, **self.transformer_layer_params))

        # Undo data format change
        # def undo(x):
        #     if isinstance(x, tuple) or isinstance(x, list):
        #         x = x[0]
        #     return x.transpose(0, 1).contiguous()
        def undo(inputs):
            if len(inputs) == 3:
                x, loss_mask, labels = inputs[0], inputs[1], inputs[2]
                cu_seqlens = None
            elif len(inputs) == 5:
                x, cu_seqlens, position_ids, loss_mask, labels = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]  # noqa
            return x.transpose(0, 1).contiguous(), loss_mask, labels, cu_seqlens
        specs.append(undo)

        # Final layernorm after transformer layers
        """
        self.spec_layer_norm_params = {"layer_norm_params": self.layer_norm_params}
        class LayerNormSpec(nn.Module):
            def __init__(self, layer_norm_params):
                super().__init__()
                self.layer_norm = build_layer_norm(layer_norm_params, layer_spec=False)

            def forward(self, inputs):
                hidden_states, loss_mask, labels, cu_seqlens = inputs[0], inputs[1], inputs[2], inputs[3]
                output = self.layer_norm(hidden_states)
                return output, loss_mask, labels, cu_seqlens
        specs.append(LayerSpec(LayerNormSpec, **self.spec_layer_norm_params))
        """
        specs.append(build_layer_norm(self.layer_norm_params, layer_spec=True))

        specs.append(LayerSpec(EmbedddingPipeNoTied, **self.lm_head_params))

        # Convert to fp32 if needed
        if fp16 or bf16:
            specs.append(float16_to_fp32)

        return specs


_EVA_MODELS = {
    "eva_mini": {
        "num_layers": 48,  # 48
        "hidden_size": 768,  # 6144,
        "num_attention_heads": 48,  # 48,
        "num_kv_attention_heads": 8,
        "intermediate_size": 2048,  # 16384,
        "eps": 1e-5,
        # vision
        "num_vit_layers": 45,  # 45,
        "vision_hidden_size": 400,  # 3200,
        "vision_image_size": 448,
        "vision_patch_size": 14,
        "vision_intermediate_size": 1600,  # 12800,
        "vision_num_attention_heads": 25,  # 25,
        "vision_attention_dropout": 0.0,
        "vision_layer_norm_eps": 1e-06,
        "vit_select_layer": -1,
        "postnorm": False
    },
    "eva_1b_20b": {
        "num_layers": 48,  # 48
        "hidden_size": 6144,  # 6144,
        "num_attention_heads": 48,  # 48,
        "num_kv_attention_heads": 8,
        "intermediate_size": 16384,  # 16384,
        "eps": 1e-5,
        # vision
        "num_vit_layers": 40,  # 45,
        "vision_hidden_size": 1408,  # 3200,
        "vision_image_size": 224,  # 224,
        "vision_patch_size": 14,
        "vision_intermediate_size": 6144,  # 12800,
        "vision_num_attention_heads": 16,  # 25,
        "vision_attention_dropout": 0.0,
        "vision_layer_norm_eps": 1e-06,
        "vit_select_layer": -1,
        "postnorm": False,
        "qkv_bias": True
    },
    "eva_4b_20b": {
        "num_layers": 48,  # 48
        "hidden_size": 6144,  # 6144,
        "num_attention_heads": 48,  # 48,
        "num_kv_attention_heads": 8,
        "intermediate_size": 16384,  # 16384,
        "eps": 1e-5,
        # vision
        "num_vit_layers": 64,  # 45,
        "vision_hidden_size": 1792,  # 3200,
        "vision_image_size": 224,  # 224,
        "vision_patch_size": 14,
        "vision_intermediate_size": 15360,  # 12800,
        "vision_num_attention_heads": 16,  # 25,
        "vision_attention_dropout": 0.0,
        "vision_layer_norm_eps": 1e-06,
        "vit_select_layer": -1,
        "postnorm": True,
        "qkv_bias": True
    },
    "eva_8b_20b": {
        "num_layers": 48,  # 48
        "hidden_size": 6144,  # 6144,
        "num_attention_heads": 48,  # 48,
        "num_kv_attention_heads": 8,
        "intermediate_size": 16384,  # 16384,
        "eps": 1e-5,
        # vision
        "num_vit_layers": 32,  # 45,
        "vision_hidden_size": 4096,  # 3200,
        "vision_image_size": 448,  # 224,
        "vision_patch_size": 14,
        "vision_intermediate_size": 20480,  # 12800,
        "vision_num_attention_heads": 32,  # 25,
        "vision_attention_dropout": 0.0,
        "vision_layer_norm_eps": 1e-06,
        "vit_select_layer": -1,
        "postnorm": False,
        "qkv_bias": False
    },
    "eva_18b_20b": {
        "num_layers": 48,  # 48
        "hidden_size": 6144,  # 6144,
        "num_attention_heads": 48,  # 48,
        "num_kv_attention_heads": 8,
        "intermediate_size": 16384,  # 16384,
        "eps": 1e-5,
        # vision
        "num_vit_layers": 48,  # 45,
        "vision_hidden_size": 5120,  # 3200,
        "vision_image_size": 224,  # 224,
        "vision_patch_size": 14,
        "vision_intermediate_size": 25600,  # 12800,
        "vision_num_attention_heads": 40,  # 25,
        "vision_attention_dropout": 0.0,
        "vision_layer_norm_eps": 1e-06,
        "vit_select_layer": -1,
        "postnorm": True,
        "qkv_bias": False
    },
}


def eva_custom(**cfg_model):
    cfg_model = update_model_cfg(cfg_model)
    model = EVAModelPipe(**cfg_model)
    # set save and load
    model.load_lora_ckpt_pretrained = load_lora_ckpt_pretrained
    model.load_ckpt_pretrained = load_ckpt_pretrained
    model.save_lora_ckpt_pretrained = save_lora_ckpt_pretrained
    # set trainable
    model.set_train_status = set_train_status
    model.set_train_params = set_train_params
    return model


def eva_mini(**cfg_model):
    cfg_mini = _EVA_MODELS['eva_mini']
    cfg_model.update(cfg_mini)
    return eva_custom(**cfg_model)


def eva_1b_20b(**cfg_model):
    cfg_1b_20b = _EVA_MODELS['eva_1b_20b']
    cfg_model.update(cfg_1b_20b)
    return eva_custom(**cfg_model)


def eva_4b_20b(**cfg_model):
    cfg_4b_20b = _EVA_MODELS['eva_4b_20b']
    cfg_model.update(cfg_4b_20b)
    return eva_custom(**cfg_model)


def eva_8b_20b(**cfg_model):
    cfg_8b_20b = _EVA_MODELS['eva_8b_20b']
    cfg_model.update(cfg_8b_20b)
    return eva_custom(**cfg_model)


def eva_18b_20b(**cfg_model):
    cfg_18b_20b = _EVA_MODELS['eva_18b_20b']
    cfg_model.update(cfg_18b_20b)
    return eva_custom(**cfg_model)
