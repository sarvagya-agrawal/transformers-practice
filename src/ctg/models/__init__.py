"""
@author: Mathieu Tuli, Sarvagya Agrawal
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from pathlib import Path

from transformers import BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import GPT2Config, GPT2LMHeadModel

import torch

TOKENIZERS = ['bert-base-uncased', 'openai-gpt', 'gpt2']
MODELS = ['gpt2', 'distilgpt2']


def get_tokenizer(tokenizer_name: str,
                  cache_dir: Path,
                  lower_case: bool = True) -> PreTrainedTokenizerBase:
    if tokenizer_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir)
    elif tokenizer_name == 'openai-gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(
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
        _config = GPT2Config
    if pretrained:
        model = _model.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            config=_config.from_pretrained(model_name))
    else:
        model = _model(
            config=_config(model_name))
    return model
