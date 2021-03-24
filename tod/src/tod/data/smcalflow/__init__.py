"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from ..utils import read_lines
from pathlib import Path


def load_data(root: Path):
    src_fname = root / 'train.dataflow_dialogues.jsonl'
    tgt_fname = root / 'valid.dataflow_dialogues.jsonl'
