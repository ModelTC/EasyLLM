from __future__ import division

# Standard Library
import logging
import sys
import torch.distributed as dist
logs = set()


log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


# LOGGER
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"

COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}

MASTER_RANK = 0


def is_master():
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def basicConfig(*args, **kwargs):
    return


# To prevent duplicate logs, we mask this baseConfig setting
logging.basicConfig = basicConfig
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('default').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        msg = record.msg
        levelname = record.levelname
        if self.use_color and levelname in COLORS and COLORS[levelname] != WHITE:
            if isinstance(msg, str):
                msg_color = COLOR_SEQ % (30 + COLORS[levelname]) + msg + RESET_SEQ
                record.msg = msg_color
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def init_log(name='global', level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    format_str = f'%(asctime)s-rk{MASTER_RANK}-%(filename)s#%(lineno)d:%(message)s'
    logger.addFilter(lambda record: is_master())
    formatter = ColoredFormatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


default_logger = init_log('global', logging.INFO)
