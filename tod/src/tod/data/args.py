"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathie@gmail.com
"""
from onmt.opts import _add_dynamic_corpus_opts, _add_dynamic_fields_opts,\
    _add_dynamic_transform_opts, _add_reproducibility_opts
from configargparse import ArgumentParser


def build_vocab_args(sub_parser: ArgumentParser,
                     build_vocab_only: bool) -> None:
    # dynamic_prepare_opts(sub_parser, build_vocab_only=True)
    _add_dynamic_corpus_opts(sub_parser, build_vocab_only=build_vocab_only)
    _add_dynamic_fields_opts(sub_parser, build_vocab_only=build_vocab_only)
    _add_dynamic_transform_opts(sub_parser)

    if build_vocab_only:
        _add_reproducibility_opts(sub_parser)


def prep_smcalflow_args(sub_parser: ArgumentParser) -> None:
    sub_parser.add_argument(
        "--data",
        required=True,
        help="the jsonl file containing the dialogue data with" +
             "dataflow programs",
    )
    sub_parser.add_argument(
        "--output",
        required=True,
        help="the output folder containing the final processed data"
    )
    sub_parser.add_argument(
        "--context",
        required=True,
        type=int,
        help="number of previous turns to be included in the source sequence",
    )
    sub_parser.add_argument(
        "--include-program",
        default=False,
        action="store_true",
        help="if True, include the gold program for the context turn parts",
    )
    sub_parser.add_argument(
        "--include-agent-utterance",
        default=False,
        action="store_true",
        help="if True, include the gold agent utterance for" +
             "the context turn parts",
    )
    sub_parser.add_argument(
        "--include-described-entities",
        default=False,
        action="store_true",
        help="if True, include the described entities field for the" +
             "context turn parts",
    )
    sub_parser.add_argument(
        "--subsets",
        nargs="+",
        default=['train', 'valid'],
        help="list of output file basenames for the extracted text data" +
             "for OpenNMT",
    )
