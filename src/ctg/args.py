"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import _ArgumentGroup
from typing import List

from argparse import ArgumentParser, Namespace


from .data.args import prep_smcalflow_args  # , build_vocab_args
from .test.args import test_smcalflow_args
from .utils.logging import LogLevel
from .train.args import train_args as _train_args


def split_args(groups: _ArgumentGroup,
               group_titles: List[str],
               args: Namespace) -> Namespace:
    split_args = Namespace()
    for group in groups:
        if group.title in group_titles:
            sub_ns = Namespace()
            sub_ns.__dict__ = {a.dest: getattr(
                args, a.dest, None) for a in group._group_actions}
            split_args.__dict__[group.title] = sub_ns
        else:
            other_args = {a.dest: getattr(
                args, a.dest, None) for a in group._group_actions}
            for k, v in other_args.items():
                split_args.__dict__[k] = v
    return split_args


def train_args(parent_parser: ArgumentParser,
               sub_parser: ArgumentParser) -> None:
    # group = sub_parser.add_argument_group('Train')
    _train_args(sub_parser)
    # group.add_argument(
    #     '--framework-choice', default='onmt', choices=['onmt', 'custom'],
    #     type=str, help="Define framework choice.")
    # build_vocab_args(sub_parser, build_vocab_only=False)
    # model_opts(sub_parser)
    # _add_train_general_opts(sub_parser)
    # _add_train_dynamic_data(sub_parser)


def decode_args(parent_parser: ArgumentParser,
                sub_parser: ArgumentParser) -> None:
    group = sub_parser.add_argument_group('Translate')
    # group.add_argument(
    #     '--framework-choice', default='onmt', choices=['onmt', 'custom'],
    #     type=str, help="Define framework choice.")
    # translate_opts(sub_parser)


def test_args(parent_parser: ArgumentParser,
              sub_parser: ArgumentParser) -> None:
    test_subp = sub_parser.add_subparsers(dest='test_command')
    smcalflow_test = test_subp.add_parser(
        'smcalflow',
        help="run 'python -m tod test smcalflow --help' for arguments",
        parents=[parent_parser], add_help=False
    )
    test_smcalflow_args(smcalflow_test)


def logging_args(parser: ArgumentParser) -> None:
    group = parser.add_argument_group("Logs")
    # group.add_argument(
    #     '-v', '--verbose', action='store_true',
    #     dest='verbose',
    #     help="Set verbose. In effect, set --log-level to INFO.")
    # group.set_defaults(verbose=False)
    group.add_argument('--log-level', type=LogLevel.__getitem__,
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
    # build_vocab = data_subp.add_parser(
    #     'build-vocab',
    #     help="run 'python -m tod data build-vocab  --help' for arguments",
    #     parents=[parent_parser], add_help=False
    # )
    # build_vocab_args(build_vocab, build_vocab_only=True)
