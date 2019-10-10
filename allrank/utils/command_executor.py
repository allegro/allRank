import os

from allrank.utils.ltr_logging import get_logger

logger = get_logger()


def execute_command(command):
    logger.info("will execute {}".format(command))
    result = os.system(command)
    logger.info("exit_code = {}".format(result))
    if result != 0:
        raise RuntimeError("non-zero exit-code: {} from command '{}'".format(result, command))
