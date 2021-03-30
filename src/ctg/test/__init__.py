"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace

from .smcalflow import main as smcalflow_main
from ..utils.logging import logger


def main(args: Namespace) -> None:
    if args.test_command == 'smcalflow':
        smcalflow_main(args)
    else:
        logger.critical(f"Unknown test subcommand {args.test_command}. " +
                        "see 'python -m tod test --help' for options.")
