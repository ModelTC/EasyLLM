import time

import os
import threading
from torch.multiprocessing import Process, Manager, Lock  # noqa
import logging as logger
from .petrel_helper import PetrelHelper


def _to_cpu(ele, snapshot=None):
    if snapshot is None:
        snapshot = {}
    if hasattr(ele, 'cpu'):
        snapshot = ele.cpu()
    elif isinstance(ele, dict):
        snapshot = {}
        for k, v in ele.items():
            snapshot[k] = None
            snapshot[k] = _to_cpu(v, snapshot[k])
    elif isinstance(ele, list):
        snapshot = [None for _ in range(len(ele))]
        for idx, v in enumerate(ele):
            snapshot[idx] = _to_cpu(v, snapshot[idx])
    else:
        snapshot = ele
    return snapshot


class CheckpointManager(object):
    def __init__(self,
                 pid=None,
                 use_thread=False):
        self.ckpt_dict = None
        self.ckpt_path = None
        self.pid = os.getpid() if pid is None else pid
        self.lock = Lock()
        self.chk_process = None
        if use_thread:
            self.concur_fn = getattr(threading, 'Thread')
        else:
            self.concur_fn = globals()["Process"]
            logger.info("Parallel Function is {}".format(self.concur_fn))
        self.mp_manager = Manager
        self.saved_ckpt_files = self.mp_manager().list()
        return

    def update_ckpt_setting(self, ckpt_dict=None, ckpt_path=None):
        with self.lock:
            if ckpt_path:
                self.ckpt_path = ckpt_path
            if ckpt_dict:
                self.ckpt_dict = ckpt_dict
        return

    def cpu_snapshot(self, model_dicts):
        snapshot = {}
        s = time.time()
        for name, ref in model_dicts.items():
            snapshot[name] = {}
            snapshot[name] = _to_cpu(ref)
        logger.info("Time for CPU snapshot = {}s".format(time.time() - s))
        return snapshot

    def _save(self, snapshot, cur_ckpt_path):
        PetrelHelper.save(snapshot, cur_ckpt_path)
        return

    def async_save_ckpt(self, ckpt_dict, ckpt_path, sync=False):
        self.update_ckpt_setting(ckpt_dict, ckpt_path)
        if os.getpid() != self.pid:  # this cond should always met.
            logger.info(f'async save process pid: {os.getpid()}, ppid: {os.getppid()}, thread: {threading.current_thread().name}')  # noqa
            return

        if self.chk_process is not None:
            # There is an checkpoint underway. Wait
            if self.chk_process.is_alive():
                self.chk_process.join()

        # sync io move data from GPU to CPU
        cpu_snapshot = {}
        start = time.time()
        with self.lock:
            if self.ckpt_dict is not None:
                cpu_snapshot = self.cpu_snapshot(self.ckpt_dict)
                cur_ckpt_path = self.ckpt_path
                cpu_snapshot[cur_ckpt_path] = cpu_snapshot

        logger.info(f"cpu snapeshot {cpu_snapshot.keys()} time {time.time() - start}")
        self.chk_process = self.concur_fn(target=self._save, args=[cpu_snapshot, cur_ckpt_path])  # noqa
        self.chk_process.start()
        if sync:
            self.chk_process.join()
        return
