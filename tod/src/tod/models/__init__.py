"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from pathlib import Path

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertTokenizer, OpenAIGPTTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from configargparse import Namespace

import torch

TOKENIZERS = ['bert-base-uncased', 'openai-gpt']
MODELS = ['gpt2', 'gpt2-small']


def get_tokenizer(tokenizer_name: str,
                  cache_dir: Path,
                  lower_case: bool = True) -> PreTrainedTokenizerBase:
    if tokenizer_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=lower_case, cache_dir=cache_dir)
    elif tokenizer_name == 'openai-gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=lower_case, cache_dir=cache_dir)
    else:
        raise ValueError(f'Unknown tokenizer. Must be one of {TOKENIZERS}')
    return tokenizer


def get_model(model_name: str,
              cache_dir: Path,
              weights: Path = None) -> torch.nn.Module:
    if model_name == 'gpt2':
        _model = GPT2LMHeadModel
        _config = GPT2Config
    if model_name == 'gpt2-small':
        _model = GPT2LMHeadModel
        _config = None
    if weights is not None:
        model = _model.from_pretrained(
            weights,
            from_pt=True,  # or from_tf
            config=_config,
            cache_dir=cache_dir)
    else:
        model = _model(
            config=_config,
            cache_dir=cache_dir)
    return model
