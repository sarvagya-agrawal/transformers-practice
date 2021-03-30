"""
@author: Sarvagya Agrawal
"""
from pathlib import Path

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
# from configargparse import Namespace

import torch

TOKENIZERS = ['gpt2', 'distilgpt2']
MODELS = ['gpt2', 'distilgpt2']


def get_tokenizer(tokenizer_name: str,
                  cache_dir: Path,
                  lower_case: bool = True) -> PreTrainedTokenizerBase:
    if tokenizer_name == 'distilgpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir)
    elif tokenizer_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir)
    else:
        raise ValueError(f'Unknown tokenizer. Must be one of {TOKENIZERS}')
    return tokenizer


def get_model(model_name: str,
              cache_dir: Path,
              pretrained: bool = True) -> torch.nn.Module:
    if model_name == 'gpt2':
        _model = GPT2LMHeadModel
        _config = GPT2Config()
    if model_name == 'distilgpt2':
        _model = GPT2LMHeadModel
        _config = GPT2Config.from_pretrained("distilgpt2", cache_dir = cache_dir)
   if pretrained:
        model = _model.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            config=_config)
    else:
        model = _model(
            config=_config)
    return model
