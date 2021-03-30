from pathlib import Path
from typing import List

from tod.data.utils import read_lines
from support import fail_if


def test_read_lines() -> None:
    filename = '/tmp/test.txt'
    with open(filename, 'w') as f:
        for i in range(10):
            f.write(f'The winner is number {i}\n')

    def test_lines(lines: List[str]) -> None:
        fail_if(len(lines) != 10)
        for i, line in enumerate(lines):
            fail_if(line != f'The winner is number {i}')
            fail_if('\n' in line)

    lines = read_lines(filename)
    test_lines(lines)
    lines = read_lines(Path(filename))
    test_lines(lines)
