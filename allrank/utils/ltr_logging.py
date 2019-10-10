import logging
import os
import sys


def init_logger(output_dir: str) -> logging.Logger:
    log_format = "[%(levelname)s] %(asctime)s - %(message)s"
    log_dateformat = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(format=log_format, datefmt=log_dateformat, stream=sys.stdout, level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)
