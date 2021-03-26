"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from configargparse import Namespace

# from onmt.bin.translate import translate as onmt_translate


def main(args: Namespace) -> None:
    # if args.framework_choice == 'onmt':
    #     onmt_translate(args)
    if args.framework_choice == 'custom':
        ...
