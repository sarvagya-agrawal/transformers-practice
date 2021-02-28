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
from .args import train_args, logging_args, data_args
from .data import main as data_main

parser = ArgumentParser(description="TOD")
parser.add_argument(
    '--logs', help="Set output for log outputs",
    default=f"logs/tod_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",)
sub_parser = parser.add_subparsers(dest='command')
logging_args(parser)
train_sub_parser = sub_parser.add_parser(
    'train', help="run 'python -m tod train --help' for train arguments")
train_args(train_sub_parser)

data_sub_parser = sub_parser.add_parser(
    'data', help="run 'python -m tod data --help' for data arguments")
data_args(data_sub_parser)

args, subcommand_args = parser.parse_known_args()
if args.verbose:
    args.log_level = LogLevel.DEBUG
logger = init_logger(Path(args.logs),
                     log_level=args.log_level)
if str(args.command) == 'train':
    train_main(args)
elif str(args.command) == 'data':
    data_main(args)
else:
    logger.critical(f"Unknown subcommand {args.command}." +
                    "See 'python -m tod --help' for options.")
