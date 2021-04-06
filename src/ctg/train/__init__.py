"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace

from onmt.bin.train import train as onmt_train
from .train import main as train_main


def main(args: Namespace):
    if args.framework_choice == 'onmt':
        onmt_train(args)
    train_main(args)
