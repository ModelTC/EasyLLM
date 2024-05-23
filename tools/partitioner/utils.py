
import json
from bisect import bisect_left

file_template = "layer_time_pp{}_tp{}.json"


def load_stats(dump_dir, NUM_LAYER, NUM_PP, NUM_TP):
    path_template = f"{dump_dir}/{file_template}"
    stats = [[] for _ in range(NUM_LAYER)]
    for pp_rank in range(NUM_PP):
        for tp_rank in range(NUM_TP):
            filename = path_template.format(pp_rank, tp_rank)
            print(f'read from {filename}')
            with open(filename, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    stats[data['layer_idx']].append(float(data['time']))
    return stats


def get_avg_stats(stats, warmup_iter=1, verbose=True):
    avg_stats = []
    for layer_idx in range(len(stats)):
        stat = stats[layer_idx][warmup_iter:]
        avg_stats.append(sum(stat) / len(stat))
        if verbose:
            print(f'layer {layer_idx}, time {avg_stats[layer_idx]}')
    return avg_stats


def prefix_sum_inc(weights):
    """ Compute an inclusive prefix sum.

    Example:
        >>> prefix_sum_inc([3,4,5])
        [3, 7, 12]
    """
    weights_ = [w for w in weights]
    for x in range(1, len(weights_)):
        weights_[x] += weights_[x - 1]
    return weights_


def _rb_partition_balanced(weights, num_parts, eps):
    total_weight = weights[-1]
    lower = total_weight / num_parts  # best case heaviest partition
    upper = total_weight  # worst case heaviest partition

    # Do a binary search for the best partitioning
    while upper > lower + eps:
        mid = lower + ((upper - lower) / 2)
        parts, success = _lprobe(weights, num_parts, mid)
        if success:
            upper = mid
        else:
            lower = mid + eps
    return upper


def _lprobe(weights, num_parts, bottleneck):
    num_items = len(weights)
    total_weight = weights[-1]

    # initialize partitioning
    parts = [0] * (num_parts + 1)
    for p in range(1, num_parts + 1):
        parts[p] = num_items

    bsum = bottleneck  # running sum of target weight for pth partition
    chunksize = num_items // num_parts
    step = chunksize
    for p in range(1, num_parts):
        # Jump to the next bucket
        while (step < num_items) and (weights[step] < bsum):
            step += chunksize

        # Find the end index of partition p
        parts[p] = bisect_left(weights, bsum, lo=step - chunksize, hi=min(step, num_items))
        # Nothing more to partition, return early
        if parts[p] == num_items:
            # See if the current partition is overweight.
            part_size = weights[-1] - weights[parts[p - 1]]
            return parts, part_size < bottleneck

        # Next partition target
        bsum = weights[parts[p] - 1] + bottleneck

    return parts, bsum >= total_weight


def partition_balanced(weights, num_parts, eps=1e-3):
    # num_items = len(weights)
    weights_ = prefix_sum_inc(weights)

    # Find the smallest bottleneck (weight of heaviest partition)
    bottleneck = _rb_partition_balanced(weights_, num_parts, eps=eps)

    # Now compute that partitioning
    parts, success = _lprobe(weights_, num_parts, bottleneck)
    assert success

    return parts


def print_time(parts):
    for pp_rank in range(NUM_PP):
        start = parts[pp_rank]
        end = parts[pp_rank + 1]
        forward_time = sum(avg_stats[start:end])
        print(f'pp rank {pp_rank}, forward time {forward_time}')


def compute_forward_time(dump_dir='./', NUM_PP=4, NUM_TP=2, NUM_LAYER=101, warmup_iter=1):
    stats = load_stats(dump_dir, NUM_LAYER, NUM_PP, NUM_TP)
    avg_stats = get_avg_stats(stats)
    return avg_stats


if __name__ == '__main__':
    NUM_PP = 4
    NUM_TP = 4
    NUM_LAYER = 101
    warmup_iter = 1
    stats = load_stats('./', NUM_LAYER, NUM_PP, NUM_TP)
    avg_stats = get_avg_stats(stats)
    parts = [0, 49, 66, 82, 101]
    print(parts)
    print_time(parts)

    parts = partition_balanced(avg_stats, NUM_PP)
    print(parts)
    print_time(parts)
