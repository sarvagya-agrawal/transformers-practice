"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import List, Union
from pathlib import PosixPath


def read_lines(filename: Union[str, PosixPath]) -> List[str]:
    """Read file and split into lines"""
    return open(filename).read().strip().split('\n')
