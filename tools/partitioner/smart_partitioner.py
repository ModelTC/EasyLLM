import numpy as np
import matplotlib.pyplot as plt
import os

from utils import get_avg_stats, load_stats
from utils import partition_balanced
import argparse

from comm import get_comm_times
import itertools

class Partitioner(object):
    def __init__(self, num_layer, num_pp, num_tp,
                 forward_times, comm_times=None, mem_usages=None):
        self.num_layer = num_layer
        self.num_pp = num_pp
        self.num_tp = num_tp
        self.forward_times = forward_times
        self.comm_times = comm_times
        self.mem_usages = mem_usages

        self._best_part_by_forward_time = None

    """
        根据forward_time生成最好的partition, 也就是我们最初的baseline
    """
    @property
    def best_part_by_forward_time(self):
        if self._best_part_by_forward_time is None:
            self._best_part_by_forward_time = partition_balanced(self.forward_times, self.num_pp)
        return self._best_part_by_forward_time
    
    """
        根据forward的方差来选择topk
    """
    def partiton_by_fwd_std(self, topk=50, radius=4, verbose=False):
        grids = self._make_grid(self.best_part_by_forward_time, radius)
        grids_fwd_times = self.get_grids_times(grids)
        sorted_idx, sorted_vars = self.sort_by_var(grids_fwd_times)
        sorted_grids = grids[sorted_idx]

        if verbose:
            plot_vars = 1 / (np.clip(np.round(sorted_vars, 4), a_min=1e-4, a_max=100000))
            plt.bar(list(range(len(sorted_vars))), plot_vars)
            plt.savefig("vars.png")
            plt.clf()
            for grid, var in zip(grids, sorted_vars):
                print(grid, var)
            
        return sorted_grids[:topk]

    def partiton_by_comm_plus_var(self, comm_topk=100, var_topk=50, radius=4, verbose=False):
        grids = self._make_grid(self.best_part_by_forward_time, radius)
        part_comm_times = self.get_grids_comm_times(grids)
        grids_fwd_times = self.get_grids_times(grids)
        vars = np.var(np.array(grids_fwd_times), axis=1).tolist()
        comm_fwd_times = [part_comm_times[i] + vars[i] for i in range(len(part_comm_times))]
        sorted_idx = np.argsort(comm_fwd_times)
        if comm_topk == -1:
            comm_sorted_grids = grids[sorted_idx][:]
        else:
            comm_sorted_grids = grids[sorted_idx][:comm_topk]
        grids_fwd_times = self.get_grids_times(comm_sorted_grids)
        sorted_idx, _ = self.sort_by_var(grids_fwd_times)

        return comm_sorted_grids[sorted_idx][:var_topk]

    """
        根据一个划分好的parts, 以radius来划分grid
    """
    def _make_grid(self, parts, radius=4):
        pp_num = len(parts)
        pp_bound = []
        for i in range(1, pp_num-1):
            pp_bound.append(parts[i])
        choice_grid = []
        for item in pp_bound:
            temp = []
            for i in range(item - radius, item + radius + 1):
                temp.append(i)
            choice_grid.append(temp)
        grids = list(itertools.product(*choice_grid))
        return np.array(grids)
    
    """
        得到任意一个partition的每个区间的times
    """
    def get_partition_fwd_time(self, parts):
        """
            parts: [0, 49, 66, 82, 101]
        """
        assert len(parts) == (self.num_pp + 1)
        fwd_times = []
        for pp_rank in range(self.num_pp):
            start = parts[pp_rank]
            end = parts[pp_rank+1]
            fwd_time = sum(self.forward_times[start:end])
            fwd_times.append(fwd_time)
        return np.array(fwd_times)

    """
        根据划分好的grid得到每个grid的每个区间forward times
    """
    def get_grids_times(self, grids):
        grids_fwd_times = []
        for grid in grids:
            tmp_parts = [0] + list(grid) + [self.num_layer]
            fwd_times = self.get_partition_fwd_time(tmp_parts)
            grids_fwd_times.append(fwd_times)
        return np.array(grids_fwd_times)
    
    def sort_by_var(self, list_fwd_times):
        vars = np.var(np.array(list_fwd_times), axis=1)
        sorted_idx = np.argsort(vars)
        return sorted_idx, vars[sorted_idx]
    
    def get_partition_comm_time(self, parts):
        """
            parts: [0, 49, 66, 82, 101]
        """
        assert len(parts) == (self.num_pp + 1)
        part_comm_times = 0
        for pp_rank in range(self.num_pp):
            endpoint = parts[pp_rank+1] - 1
            part_comm_times += self.comm_times[endpoint]
        return part_comm_times
    
    def get_grids_comm_times(self, grids):
        grids_comm_times = []
        for grid in grids:
            tmp_parts = [0] + list(grid) + [self.num_layer]
            comm_times = self.get_partition_comm_time(tmp_parts)
            grids_comm_times.append(comm_times)
        return np.array(grids_comm_times)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_pp",
        default=4, type=int,
        help="pp num")
    parser.add_argument(
        "--num_tp",
        default=4, type=int,
        help="tp num")
    parser.add_argument(
        "--num_layer",
        default=101, type=int,
        help="num layer")
    parser.add_argument(
        "--warmup_iter",
        default=1, type=int,
        help="warmup_iter")
    parser.add_argument(
        "--radius",
        default=2, type=int,
        help="radius")
    parser.add_argument(
        "--vit_bs",
        default=1, type=int,
        help="vit bs")
    parser.add_argument(
        "--vit_length",
        default=1, type=int,
        help="vit length")
    parser.add_argument(
        "--vit_hidden_size",
        default=1, type=int,
        help="vit hidden size")

    parser.add_argument(
        "--llm_bs",
        default=1, type=int,
        help="llm bs")
    
    parser.add_argument(
        "--llm_length",
        default=4096, type=int,
        help="llm length")

    parser.add_argument(
        "--llm_hidden_size",
        default=8192, type=int,
        help="llm length")
    
    parser.add_argument(
        "--vit_num_token",
        default=256, type=int,
        help="vit num token")

    parser.add_argument(
        "--vit_layer_num",
        default=45, type=int,
        help="vit_layer_num")

    parser.add_argument(
        "--llm_layer_num",
        default=48, type=int,
        help="llm_layer_num")
    parser.add_argument(
        "--topn",
        default=50, type=int,
        help="topn")

    parser.add_argument(
        "--input_path",
        default='', type=str,
        help="input_path",
    )
    parser.add_argument(
        "--output_path",
        default='', type=str,
        help="output_path",
    )
    args = parser.parse_args()
    # 初始化变量
    NUM_PP = args.num_pp
    NUM_TP = args.num_tp
    NUM_LAYER = args.num_layer
    warmup_iter = args.warmup_iter

    vit_bs = args.vit_bs
    vit_length = args.vit_length
    vit_hidden_size = args.vit_hidden_size
    llm_bs = args.llm_bs
    llm_length = args.llm_length
    llm_hidden_size = args.llm_hidden_size
    vit_layer_num = args.vit_layer_num
    llm_layer_num = args.llm_layer_num
    vit_num_token = args.vit_num_token

    comm_times = get_comm_times(vit_bs=vit_bs,
                                vit_length=vit_length,
                                vit_hidden_size=vit_hidden_size,
                                llm_bs=llm_bs,
                                llm_length=llm_length,
                                llm_hidden_size=llm_hidden_size,
                                vit_layer_num=vit_layer_num,
                                llm_layer_num=llm_layer_num,
                                vit_num_token=vit_num_token)
    
    # 读取存下来的forward time
    stats = load_stats(args.input_path, NUM_LAYER, NUM_PP, NUM_TP)
    avg_forward_time = get_avg_stats(stats, warmup_iter, verbose=False)

    # 初始化smart partioner
    partitioner = Partitioner(num_layer=NUM_LAYER, num_pp=NUM_PP, num_tp=NUM_TP,
                              forward_times=avg_forward_time,
                              comm_times=comm_times)

    print(partitioner.best_part_by_forward_time)
    print("fwd std based")
    radius = args.radius
    fwd_parts = partitioner.partiton_by_fwd_std(topk=args.topn, verbose=False, radius=radius)
    print(fwd_parts)
    print("commucation time based")
    comm_parts = partitioner.partiton_by_comm_plus_var(comm_topk=1000, var_topk=args.topn, radius=radius)
    print(comm_parts)

    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, "pp_method.txt")

    final_parts = fwd_parts.tolist()
    fwd_parts_set = set()
    for item in fwd_parts:
        fwd_parts_set.add(str(item))
    for item in comm_parts:
        if str(item) not in fwd_parts_set:
            final_parts.append(item)
    with open(out_path, 'w') as f:
        for item in final_parts:
            temp = [0]
            for i in item:
                temp.append(i)
            temp.append(args.num_layer)
            method = "manual:"
            for idx, i in enumerate(temp):
                if idx == len(temp) - 1:
                    method += f"{i}"
                else:
                    method += f"{i},"
            print(method, file=f)

        

    

        
