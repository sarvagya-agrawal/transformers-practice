"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from datetime import datetime
from pathlib import Path


from argparse import ArgumentParser

import torch

from .args import train_args, logging_args, data_args, test_args,\
    decode_args, split_args, general_args
from .utils.io import config_file_parser
from .decode import main as decode_main
from .utils.logging import init_logger
from .train import main as train_main
from .test import main as test_main
from .data import main as data_main

torch.multiprocessing.set_sharing_strategy('file_system')

parser = ArgumentParser(description="TOD Base Parser")
# parent_parser = ArgumentParser(
#     config_file_parser_class=YAMLConfigFileParser)
parent_parser = ArgumentParser(description='TOD Parent Parser')
# parent_parser.add_argument('--config', '-c', is_config_file=True,
#                            help='Config file path', type=yaml.safe_load)
parent_parser.add_argument('--config', '-c',
                           help='Config file path', default=None)
general_args(parser, parent_parser)
parent_parser.add_argument(
    '--logs', help="Set output for log outputs",
    default=f"logs/tod_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log",)
sub_parser = parser.add_subparsers(dest='command')
logging_args(parent_parser)
train_sub_parser = sub_parser.add_parser(
    'train', help="run 'python -m tod train --help' for train arguments",
    parents=[parent_parser], add_help=False)
train_args(parent_parser, train_sub_parser)

decode_sub_parser = sub_parser.add_parser(
    'decode', parents=[parent_parser], add_help=False,
    help="run 'python -m tod decode --help' for decode arguments",)
decode_args(parent_parser, decode_sub_parser)

test_sub_parser = sub_parser.add_parser(
    'test', help="run 'python -m tod test --help' for test arguments",
    parents=[parent_parser], add_help=False)
test_args(parent_parser, test_sub_parser)

data_sub_parser = sub_parser.add_parser(
    'data', help="run 'python -m tod data --help' for data arguments",
    parents=[parent_parser], add_help=False)
# config_file_parser_class=YAMLConfigFileParser)
data_args(parent_parser, data_sub_parser)

# args, unknown = parser.parse_known_args()
args = parser.parse_args()
if args.config is not None:
    args = config_file_parser(args)
logger = init_logger(Path(args.logs),
                     log_level=args.log_level)
if str(args.command) == 'train':
    args = split_args(train_sub_parser._action_groups,
                      ['train', 'data', 'io'],
                      args)
    train_main(args)
elif str(args.command) == 'decode':
    args = split_args(decode_sub_parser._action_groups,
                      ['decode', 'data', 'io'],
                      args)
    decode_main(args)
elif str(args.command) == 'test':
    test_main(args)
elif str(args.command) == 'data':
    data_main(args)
else:
    logger.critical(f"Unknown subcommand {args.command}. " +
                    "See 'python -m tod --help' for options.")
