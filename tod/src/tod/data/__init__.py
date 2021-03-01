"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathie@gmail.com
"""
from onmt.bin.build_vocab import build_vocab_main
from configargparse import Namespace

from ..utils.logging import logger
from .smcalflow.prep import main as prep_smcalflow_main


def main(args: Namespace):
    if args.data_command == 'prep-smcalflow':
        prep_smcalflow_main(args)
    if args.data_command == 'build-vocab':
        build_vocab_main(args)
    else:
        logger.critical(f"Unknown data subcommand {args.data_command}. " +
                        "see 'python -m tod data --help' for options.")
