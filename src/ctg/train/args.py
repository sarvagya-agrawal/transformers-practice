"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import _ArgumentGroup

from argparse import ArgumentParser, Namespace

from ..models import MODELS, TOKENIZERS
from ..optim import OPTIMIZERS, SCHEDULERS
from ..data import DATASETS, TASKS


def train_args(sub_parser: ArgumentParser) -> None:
    train_group = sub_parser.add_argument_group('train')
    train_group.add_argument("--model", type=str, choices=MODELS)
    train_group.add_argument("--pretrained", type=bool, default=True)
    train_group.add_argument("--tokenizer", type=str, choices=TOKENIZERS)
    train_group.add_argument("--gpu", nargs="+")
    train_group.add_argument("--optimizer", type=str, choices=OPTIMIZERS)
    train_group.add_argument("--scheduler", type=str, choices=SCHEDULERS)
    train_group.add_argument("--batch-size", type=int)
    train_group.add_argument("--max-length", type=int)
    train_group.add_argument("--max-epochs", type=int)
    train_group.add_argument("--num-workers", type=int, default=4)
    train_group.add_argument("--num-trials", type=int, default=3)
    train_group.add_argument("--loss", type=str)
    train_group.add_argument("--mask", type=bool, default=False)
    train_group.add_argument("--clip-grad", default=1.0, type=float)
    train_group.add_argument("--optimizer-kwargs")
    train_group.add_argument("--scheduler-kwargs")

    data_group = sub_parser.add_argument_group('data')
    data_group.add_argument("--name", type=str, choices=DATASETS)
    data_group.add_argument("--train-src", type=str)
    data_group.add_argument("--train-tgt", type=str, default=None)
    data_group.add_argument("--val-src", type=str)
    data_group.add_argument("--val-tgt", type=str, default=None)
    data_group.add_argument("--task", type=str, choices=TASKS)
    data_group.add_argument("--overwrite-cache", type=str)

    io_group = sub_parser.add_argument_group('io')
    io_group.add_argument("--output", type=str)
    io_group.add_argument("--checkpoint", type=str)
    io_group.add_argument("--cache-dir", type=str)
    io_group.add_argument("--save-freq", type=int, default=25)
