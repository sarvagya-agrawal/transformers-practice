"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathie@gmail.com
"""
from argparse import ArgumentParser
from pathlib import Path

import logging

from .utils.logging import init_logger, args as logging_args
from .train import args as train_args, main as train_main

parser = ArgumentParser(description=__doc__)
subparser = parser.add_subparsers(dest='command')
logging_args(parser)
train_subparser = subparser.add_parser(
    'train', help='Train commands', parents=[parser])
train_args(train_subparser)

args = parser.parse_args()
if args.verbose:
    args.log_level = logging.DEBUG
logger = init_logger(Path('logs/tod.log'), level=args.log_level)

if str(args.command) == 'train':
    train_main(args)
else:
    logger.critical(f"Unknown subcommand {args.command}")
