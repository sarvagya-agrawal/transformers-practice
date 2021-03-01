"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathie@gmail.com
"""
from datetime import datetime
from pathlib import Path

from configargparse import ArgumentParser

from .utils.logging import LogLevel, init_logger
from .train import main as train_main
from .args import train_args, logging_args, data_args
from .data import main as data_main

parser = ArgumentParser(description="TOD")
parent_parser = ArgumentParser()
parent_parser.add_argument('--config', '-c', is_config_file=True,
                           help='Config file path')
parent_parser.add_argument(
    '--logs', help="Set output for log outputs",
    default=f"logs/tod_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",)
sub_parser = parser.add_subparsers(dest='command')
logging_args(parent_parser)
train_sub_parser = sub_parser.add_parser(
    'train', help="run 'python -m tod train --help' for train arguments",
    parents=[parent_parser], add_help=False)
train_args(parent_parser, train_sub_parser)

data_sub_parser = sub_parser.add_parser(
    'data', help="run 'python -m tod data --help' for data arguments",
    parents=[parent_parser], add_help=False)
data_args(parent_parser, data_sub_parser)

args, unknown = parser.parse_known_args()
if args.verbose:
    args.log_level = LogLevel.DEBUG
logger = init_logger(Path(args.logs),
                     log_level=args.log_level)
if str(args.command) == 'train':
    train_main(args)
elif str(args.command) == 'data':
    data_main(args)
else:
    logger.critical(f"Unknown subcommand {args.command}. " +
                    "See 'python -m tod --help' for options.")
