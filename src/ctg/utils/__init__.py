"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace

from .logging import logger


def pretty_print_list(_list) -> str:
    return str(_list)


def pretty_print_dict(_dict, spacing: int = 20, indent: int = 0) -> str:
    indent = ' '*indent
    for k, v in _dict.items():
        if v is None:
            logger.info(f'{indent}{k:{spacing}}| {"None":{spacing}}')
        elif isinstance(v, list):
            logger.info(
                f'{indent}{k:{spacing}}| {pretty_print_list(v):{spacing}}')
        elif isinstance(v, dict):
            logger.info(f'{indent}{k:{spacing}}| ')
            pretty_print_dict(v, spacing=spacing-4, indent=4)
        elif isinstance(v, Namespace):
            pretty_print_namespace(k, v)
        else:
            logger.info(f'{indent}{k:{spacing}}| {k:{spacing}}')


def pretty_print_namespace(name: str, args: Namespace) -> None:
    spacing = 30
    logger.info("="*spacing*2)
    logger.info(f"{name} args")
    logger.info("="*spacing*2)
    pretty_print_dict(args.__dict__, spacing=spacing)
