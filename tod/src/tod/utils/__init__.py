"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace
import yaml


def config_file_parser(args: Namespace) -> Namespace:
    with open(args.config, 'r') as f:
        config = config = yaml.load(f, Loader=yaml.Loader)
    for k, v in args.__dict__.items():
        if k.replace('_', '-') in config:
            args.__dict__[k] = config[k.replace('_', '-')]
    return args
