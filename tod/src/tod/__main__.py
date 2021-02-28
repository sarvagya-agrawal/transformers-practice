"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathie@gmail.com
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from .utils.logging import LogLevel, init_logger
from .train import main as train_main
from .args import train_args, logging_args

parser = ArgumentParser(description="TOD")
parser.add_argument(
    '--logs', help="Set output for log outputs",
    default=f"logs/tod_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",)
subparser = parser.add_subparsers(dest='command')
logging_args(parser)
train_subparser = subparser.add_parser(
    'train', help="run 'python -m tod train --help' for train arguments",
    parents=[parser], add_help=False)
train_args(train_subparser)

args = parser.parse_args()
if args.verbose:
    args.log_level = LogLevel.DEBUG
logger = init_logger(Path(args.logs),
                     log_level=args.log_level)
if str(args.command) == 'train':
    train_main(args)
else:
    logger.critical(f"Unknown subcommand {args.command}")
