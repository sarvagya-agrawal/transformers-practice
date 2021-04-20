from argparse import ArgumentParser
from pathlib import Path
import json

parser = ArgumentParser()
parser.add_argument('--src')
parser.add_argument('--tgt')
parser.add_argument('--output')


def get_lines(fname: str):
    return [line.strip() for line in Path(fname).open('r').readlines()]


if __name__ == "__main__":
    args = parser.parse_args()
    src_lines = get_lines(args.src)
    tgt_lines = get_lines(args.tgt)
    data = {'data': [{'source': src, 'target': tgt}
                     for src, tgt in zip(src_lines, tgt_lines)]}
    with Path(args.output).open('w') as f:
        f.write(json.dumps(data))
