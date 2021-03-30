"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from pathlib import Path


def create_dir(dname: Path) -> None:
    dname.mkdir(exists_ok=True, parents=True)
    return dname
