"""
@author: OpenNMT-py
@github: OpenNMT-py
@link: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/logging.py
"""
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from enum import Enum

import logging

logger = logging.getLogger()


class LogLevel(Enum):
    '''
    What the stdlib did not provide!
    '''
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    NOTSET = logging.NOTSET

    def __str__(self):
        return self.name

    @staticmethod
    def items():
        return list(map(lambda c: c, LogLevel))


def init_logger(log_file: Path,
                log_level: LogLevel = LogLevel.NOTSET,
                rotate=False) -> logging.Logger:
    if log_level not in LogLevel.items():
        raise ValueError(f"Unknown log level: {log_level}")
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(log_level.value)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.suffix == '':
        log_file.mkdir(parents=True, exist_ok=True)
        log_file /= f"tod_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
    if log_file.name and log_file.name != '':
        if rotate:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10)
        else:
            file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(log_level.value)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
