"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathie@gmail.com
"""
from configargparse import ArgumentParser


from onmt.opts import model_opts, _add_train_general_opts, \
    _add_train_dynamic_data
from .utils.logging import LogLevel
from .data.args import prep_smcalflow_args, build_vocab_args


def train_args(parent_parser: ArgumentParser,
               sub_parser: ArgumentParser) -> None:
    group = sub_parser.add_argument_group('Train')
    group.add_argument(
        '--framework-choice', default='onmt', choices=['onmt', 'custom'],
        type=str, help="Define framework choice.")
    build_vocab_args(sub_parser, build_vocab_only=False)
    model_opts(sub_parser)
    _add_train_general_opts(sub_parser)
    _add_train_dynamic_data(sub_parser)


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


def data_args(parent_parser: ArgumentParser,
              sub_parser: ArgumentParser) -> None:
    data_subp = sub_parser.add_subparsers(dest='data_command')
    smcalflow_prep = data_subp.add_parser(
        'prep-smcalflow',
        help="run 'python -m tod data prep-smcalflow --help' for arguments",
        parents=[parent_parser], add_help=False
    )
    prep_smcalflow_args(smcalflow_prep)
    build_vocab = data_subp.add_parser(
        'build-vocab',
        help="run 'python -m tod data build-vocab  --help' for arguments",
        parents=[parent_parser], add_help=False
    )
    build_vocab_args(build_vocab, build_vocab_only=True)
