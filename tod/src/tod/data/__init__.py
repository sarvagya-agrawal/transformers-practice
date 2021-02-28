"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathie@gmail.com
"""
from ..utils.logging import logger
from argparse import Namespace


def main(args: Namespace):
    if args.data_command == 'prep-smcalflow':
        ...
    else:
        logger.critical(f"Unknown data subcommand {args.data_command}. " +
                        "see 'python -m tod data --help' for options.")
