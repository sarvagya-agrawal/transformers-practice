"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace

from ..data.utils import load_data


def main(args: Namespace) -> None:
    data_loader = load_data(src_fname=args.data.src_fname,
                            tokenizer=tokenizer,
                            max_length=args.max_length,
                            batch_size=args.batch_size,
                            cache_dir=args.io.cache_dir,
                            split='decode')

    training_agent.train()
