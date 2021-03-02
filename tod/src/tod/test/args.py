"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from configargparse import ArgumentParser


def test_smcalflow_args(sub_parser: ArgumentParser) -> None:
    sub_parser.add_argument(
        "--data",
        help="the jsonl file containing the dialogue data with" +
        " dataflow programs",
    )
    sub_parser.add_argument("--datum-ids", help="datum ID file")
    sub_parser.add_argument("--src", help="source sequence file")
    sub_parser.add_argument(
        "--tgt", help="target sequence reference file")
    sub_parser.add_argument(
        "--nbest-txt", help="onmt_translate output file")
    sub_parser.add_argument(
        "--nbest", type=int, help="number of hypos per datum")
    sub_parser.add_argument(
        "--output", help="the basename of output files")
    sub_parser.add_argument(
        "--leaderboard",
        default=False,
        action="store_true",
        help="if set, use the isCorrectLeaderboard field instead of " +
        "isCorrect field in the prediction report",
    )
    sub_parser.add_argument("--scores-json", help="output scores json file")
