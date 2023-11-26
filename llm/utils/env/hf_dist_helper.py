import os
import functools
import pickle

# Import from third library
import torch
import torch.distributed as dist

MASTER_RANK = 0


def get_rank_from_env():
    rank_cands = ['SLURM_PROCID', 'MV2_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_RANK']
    for rank_name in rank_cands:
        if rank_name in os.environ:
            return int(os.environ[rank_name])
    return None


def get_dist_rank(*args, **kwargs):
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank(*args, **kwargs)


def get_dist_world_size(*args, **kwargs):
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(*args, **kwargs)


def get_world_size_from_env():
    ws_cands = ['SLURM_NTASKS', 'MV2_COMM_WORLD_SIZE', 'OMPI_COMM_WORLD_SIZE']
    for ws_name in ws_cands:
        if ws_name in os.environ:
            return int(os.environ[ws_name])
    return None


def get_rank(*args, **kwargs):
    rank = get_rank_from_env()
    if rank is not None:
        return rank
    return get_dist_rank(*args, **kwargs)


def get_world_size(*args, **kwargs):
    world_size = get_world_size_from_env()
    if world_size is not None:
        return world_size
    return get_dist_world_size(*args, **kwargs)


def get_local_rank(*args, **kwargs):
    rank = get_rank(*args, **kwargs)
    return rank % torch.cuda.device_count()


def all_reduce(*args, **kwargs):
    if get_world_size() <= 1:
        return
    return dist.all_reduce(*args, **kwargs)


def dist_barrier(*args, **kwargs):
    if get_world_size() <= 1:
        return
    dist.barrier(*args, **kwargs)


def allgather(*args, **kwargs):
    if get_world_size() <= 1:
        return
    return dist.all_gather(*args, **kwargs)


def gather(*args, **kwargs):
    if get_world_size() <= 1:
        return
    return dist.gather(*args, **kwargs)


def broadcast(*args, **kwargs):
    if get_world_size() <= 1:
        return
    return dist.broadcast(*args, **kwargs)


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    allgather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def _serialize_to_tensor(data, group):

    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")
    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        import logging
        logger = logging.getLogger('global')
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    allgather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather_pk(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if get_world_size(group=group) == 1:
        return [data]
    rank = get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        gather(tensor, tensor_list, dst=dst, group=group)
        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        gather(tensor, [], dst=dst, group=group)
        return []


def broadcast_object(obj, group=None):
    """make suare obj is picklable
    """
    if get_world_size() <= 1:
        return obj

    serialized_tensor = _serialize_to_tensor(obj, group=None).cuda()
    numel = torch.IntTensor([serialized_tensor.numel()]).cuda()
    broadcast(numel, MASTER_RANK)
    # serialized_tensor from storage is not resizable
    serialized_tensor = serialized_tensor.clone()
    serialized_tensor.resize_(numel)
    broadcast(serialized_tensor, MASTER_RANK)
    serialized_bytes = serialized_tensor.cpu().numpy().tobytes()
    deserialized_obj = pickle.loads(serialized_bytes)
    return deserialized_obj


def setup_distributed_torch():
    dist.init_process_group(backend='nccl')
    rank = get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)


def setup_distributed_slurm(backend='nccl', port='13333'):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    if backend == 'nccl':
        dist.init_process_group(backend='nccl')
    else:
        dist.init_process_group(backend='gloo', rank=proc_id, world_size=ntasks)
    rank = get_rank()
    device = rank % torch.cuda.device_count()
    os.environ['LOCAL_RANK'] = str(device)
    torch.cuda.set_device(device)


def setup_distributed_mpi(backend='nccl',
                          addr='localhost',
                          port="13333",
                          rank=None,
                          world_size=None):
    r"""
    Overview:
        Init the distributed training setting
    """
    assert backend in ['nccl', 'gloo'], backend
    os.environ['MASTER_ADDR'] = addr or os.environ.get('MASTER_ADDR', addr)
    os.environ['MASTER_PORT'] = str(port) or os.environ.get('MASTER_PORT', str(port))

    if rank is None:
        local_id = os.environ.get('OMPI_COMM_WORLD_RANK', None)
        if local_id is None:
            raise RuntimeError("please indicate rank explicitly in dist_init method")
        else:
            rank = int(local_id)
    if world_size is None:
        ntasks = os.environ.get('OMPI_COMM_WORLD_SIZE', None)
        if ntasks is None:
            raise RuntimeError("please indicate world_size explicitly in dist_init method")
        else:
            world_size = int(ntasks)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    os.environ['LOCAL_RANK'] = str(device)
    torch.cuda.set_device(device)


def setup_distributed(launcher='slurm', backend='nccl', port=13333):
    if launcher == 'torch':
        setup_distributed_torch()
    elif launcher == 'slurm':
        setup_distributed_slurm(backend, port)
    else:
        setup_distributed_mpi()
