"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import Union

from transformers import PreTrainedModel as HFPretrained
from configargparse import Namespace
import torch


class TrainingAgent:
    def __init__(self,
                 model: Union[torch.nn.Module, HFPretrained],
                 args: Namespace,) -> None:
        self.model = model
        self.args = args

    def reset(self) -> None:
        if len(self.args.gpu) > 1:
            self.model = torch.nn.DataParallel(self.model)

    def train(self) -> None:
        ...t


def main(args: Namespace) -> None:
    ...
