"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathie@gmail.com
"""
from argparse import _SubParsersAction


def prep_smcalflow_args(sub_parser: _SubParsersAction) -> None:
    sub_parser.add_argument(
        "--data",
        help="the jsonl file containing the dialogue data with" +
             "dataflow programs",
    )
    sub_parser.add_argument(
        "--context",
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
