"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import ArgumentParser

from ..data import TASKS


def train_args(sub_parser: ArgumentParser) -> None:
    train_group = sub_parser.add_argument_group('train')
    train_group.add_argument("--model", type=str)
    train_group.add_argument("--model-pretrained", type=bool, default=True)
    train_group.add_argument("--tokenizer", type=str)
    train_group.add_argument("--tokenizer-pretrained", type=bool, default=True)
    train_group.add_argument("--hf-model-config", type=str, default=None)
    train_group.add_argument(
        "--hf-model-config-pretrained", type=str, default=True)
    # train_group.add_argument("--gpu", nargs="+")
    train_group.add_argument("--optimizer", type=str)
    train_group.add_argument("--scheduler", type=str)
    train_group.add_argument("--batch-size", type=int)
    train_group.add_argument("--batch-size-eval", type=int)
    train_group.add_argument("--max-src-length", type=int)
    train_group.add_argument("--max-tgt-length", type=int)
    train_group.add_argument("--max-epochs", type=int)
    train_group.add_argument("--num-trials", type=int, default=1)
    train_group.add_argument("--loss", type=str)
    train_group.add_argument("--mask", type=bool, default=False)
    train_group.add_argument("--clip-grad", default=1.0, type=float)
    train_group.add_argument("--optimizer-kwargs", default={})
    train_group.add_argument("--scheduler-kwargs", default={})
    train_group.add_argument("--gradient-accumulation-steps", type=int,
                             default=1,)
    train_group.add_argument("--ray-tune", type=bool, default=False)
    train_group.add_argument("--ray-tune-samples", type=int, default=20)

    data_group = sub_parser.add_argument_group('data')
    data_group.add_argument("--data-name", type=str, default=None)
    data_group.add_argument("--data-config", type=str, default=None)
    data_group.add_argument("--train-src", type=str)
    data_group.add_argument("--val-src", type=str, default=None)
    data_group.add_argument("--vocab", type=str)
    data_group.add_argument("--task", type=str, choices=TASKS)
    data_group.add_argument("--overwrite-cache", type=bool)
    data_group.add_argument("--num-workers", type=int, default=4)
    data_group.add_argument("--max-train-samples", type=int, default=-1)
    data_group.add_argument("--max-val-samples", type=int, default=-1)
    data_group.add_argument("--prefix", type=str, default='')
    data_group.add_argument("--tokenizer-files", default=None, nargs="+")
    data_group.add_argument("--pad-to-max-length", type=bool, default=False)
    data_group.add_argument("--ignore-pad-for-loss", type=bool, default=True)
    data_group.add_argument("--val-split", type=float, default=10)
    data_group.add_argument("--block-size", type=int, default=None)

    io_group = sub_parser.add_argument_group('io')
    io_group.add_argument("--output", type=str)
    io_group.add_argument("--checkpoint", type=str)
    io_group.add_argument("--cache-dir", type=str)
    io_group.add_argument("--save-freq", type=int, default=25)
    io_group.add_argument("--resume", default=None, type=str)
