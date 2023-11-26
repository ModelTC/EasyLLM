import time

import torch

from llm.utils.env import dist_env
from llm.utils.general.microbatches import RampupBatchsizeNumMicroBatches


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(f'time/{name}-time', value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = ''
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if not len(string):
            return
        string = 'time (ms)' + string
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                    torch.distributed.get_world_size() - 1):
                print(string, flush=True)
        else:
            print(string, flush=True)


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if dist_env.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def get_train_iters(num_microbatches_calculator, train_iters=None, train_samples=None):
    global_batch_size = num_microbatches_calculator.global_batch_size
    # For iteration-based training, we don't need to do anything
    if train_iters is None:
        # Constant batch size with sample-based training.
        if not isinstance(num_microbatches_calculator, RampupBatchsizeNumMicroBatches):
            train_iters = train_samples // global_batch_size
        else:
            # Sample based training with rampup batch size.
            iterations = 0
            consumed_samples = 0
            # Rampup phase.
            while consumed_samples <= num_microbatches_calculator.ramup_samples:
                num_microbatches_calculator.update(consumed_samples, False)
                consumed_samples += num_microbatches_calculator.get_current_global_batch_size()
                iterations += 1
            # Reset
            num_microbatches_calculator.update(0, False)
            # Constant phase
            # Note that we throw away any partial last batch.
            iterations += (train_samples - consumed_samples) // global_batch_size
            train_iters = iterations
    return train_iters
