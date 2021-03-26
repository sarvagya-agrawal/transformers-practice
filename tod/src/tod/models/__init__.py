"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from transformers import Tokenizer, BertTokenizer, OpenAIGPTTokenizer

import torch


def get_tokenizer(tokenizer_name: str, lower_case: bool = True) -> Tokenizer:
    toks = ['bert-base-uncased']
    if tokenizer_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=lower_case)
    elif tokenizer_name == 'openai-gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=lower_case)
    else:
        raise ValueError(f'Unknown tokenizer. Must be one of {toks}')
    return tokenizer


def get_model(model_name: str) -> torch.nn.Module:
    ...
