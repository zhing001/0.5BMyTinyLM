import torch.distributed as dist
from loguru import logger as logger

def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

class LoggerRank0:
    def info(self, *args, **kwargs):
        if not is_rank_0():
            return
        logger.info(*args, **kwargs)

logger_rank0 = LoggerRank0()