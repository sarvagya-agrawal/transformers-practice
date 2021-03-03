"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from configargparse import Namespace

from onmt.bin.train import train as onmt_train
from .train import main as custom_train


def main(args: Namespace):
    if args.framework_choice == 'onmt':
        onmt_train(args)
    if args.framework_choice == 'custom':
        custom_train(args)
