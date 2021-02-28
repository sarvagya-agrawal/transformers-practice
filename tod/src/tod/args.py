"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathie@gmail.com
"""
from argparse import ArgumentParser, _SubParsersAction

from .utils.logging import LogLevel
from .data.args import prep_smcalflow_args


def train_args(sub_parser: _SubParsersAction) -> None:
    sub_parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")


def logging_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        dest='verbose',
        help="Set verbose. In effect, set --log-level to INFO.")
    parser.set_defaults(verbose=False)
    parser.add_argument('--log-level', type=LogLevel.__getitem__,
                        default=LogLevel.INFO,
                        choices=LogLevel.__members__.values(),
                        dest='log_level',
                        help="Log level.")


def data_args(sub_parser: _SubParsersAction) -> None:
    data_subp = sub_parser.add_subparsers(dest='data_command')
    smcalflow_prep = data_subp.add_parser(
        'prep-smcalflow',
        help="run 'python -m tod data prep-smcalflow --help' for arguments",
    )
    prep_smcalflow_args(smcalflow_prep)
