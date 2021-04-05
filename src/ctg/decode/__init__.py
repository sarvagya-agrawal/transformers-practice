"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace

from .decode import main as decode_main
from ..data.utils import load_data
from ..models import get_model

DECODE_METHODS = ['greedy', 'beam']


def main(args: Namespace) -> None:
    model = get_model(args.train.model,
                      cache_dir=args.io.cache_dir,
                      pretrained=args.train.pretrained)
