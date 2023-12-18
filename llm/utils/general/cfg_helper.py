import copy
import re
from .log_helper import default_logger as logger


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    res = pattern.match(num)
    if res:
        return True
    return False


def try_decode(val):
    """bool, int, float, or str"""
    if val.upper() == 'FALSE':
        return False
    elif val.upper() == 'TRUE':
        return True
    if val.isdigit():
        return int(val)
    if is_number(val):
        return float(val)
    return val


def merge_opts_into_cfg(opts, cfg):
    cfg = copy.deepcopy(cfg)
    if opts is None or len(opts) == 0:
        return cfg

    assert len(opts) % 2 == 0
    keys, values = opts[0::2], opts[1::2]
    for key, val in zip(keys, values):
        logger.info(f'replacing {key}')
        val = try_decode(val)
        cur_cfg = cfg
        # for hooks
        if '-' in key:
            # if len(key.split('-'))!=2:
            key_p, key_s = key.split('-')
            k_module, k_type = key_p.split('.')
            cur_cfg = cur_cfg[k_module]
            flag_exist = False
            for idx in range(len(cur_cfg)):
                if cur_cfg[idx]['type'] != k_type:
                    continue
                flag_exist = True
                cur_cfg_temp = cur_cfg[idx]
                key_s = key_s.split('.')
                for k in key_s[:-1]:
                    cur_cfg_temp = cur_cfg_temp.setdefault(k, {})
                cur_cfg_temp[key_s[-1]] = val
            if not flag_exist:
                _cur_cfg = {}
                cur_cfg_temp = _cur_cfg
                key_s = key_s.split('.')
                for k in key_s[:-1]:
                    cur_cfg_temp = cur_cfg_temp.setdefault(k, {})
                cur_cfg_temp[key_s[-1]] = val
                cur_cfg.append(_cur_cfg)
        else:
            key = key.split('.')
            for k in key[:-1]:
                cur_cfg = cur_cfg.setdefault(k, {})
            cur_cfg[key[-1]] = val
    return cfg
