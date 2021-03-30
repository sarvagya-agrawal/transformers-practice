"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace

from .decode import main as decode_main


def main(args: Namespace) -> None:
    decode_main(args)
