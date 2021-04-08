"""
@author: Mathieu Tuli, Sarvagya Agrawal
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from pathlib import Path

# BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer, AlbertTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import (
    GPT2Config, GPT2LMHeadModel, BertLMHeadModel,
    AutoTokenizer, BertConfig,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Config,
    EncoderDecoderModel,
    EncoderDecoderConfig
)

import torch

from .configs import CONFIG_DICTS

TOKENIZERS = set(['bert-base-uncased', 'openai-gpt',
                  'gpt2', 'bert-base-uncased', 't5-small'])
MODELS = set(['gpt2', 'distilgpt2', 'bert-base-uncased', 't5-small'])


def get_tokenizer(tokenizer_name: str,
                  cache_dir: Path,
                  dataset: str,
                  task: str,
                  lower_case: bool = True) -> PreTrainedTokenizerBase:
    # if tokenizer_name == 'bert-base-uncased':
    #     tokenizer = BertTokenizer.from_pretrained(
    #         tokenizer_name, cache_dir=cache_dir)
    # elif tokenizer_name == 'openai-gpt':
    #     tokenizer = OpenAIGPTTokenizer.from_pretrained(
    #         tokenizer_name, cache_dir=cache_dir)
    # elif tokenizer_name == 'gpt2':
    #     tokenizer = GPT2Tokenizer.from_pretrained(
    #         tokenizer_name, cache_dir=cache_dir)
    # elif tokenizer_name == 'albert-base-v2':
    #     tokenizer = AlbertTokenizer.from_pretrained(
    #         tokenizer_name, cache_dir=cache_dir)
    if tokenizer_name in TOKENIZERS:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                  cache_dir=cache_dir)
    else:
        raise ValueError(f'Unknown tokenizer. Must be one of {TOKENIZERS}')
    if dataset == 'multiwoz2.1':
        if task == 'clm':
            tokenizer.add_special_tokens(
                {'bos_token': '<|endoftext|>'})
            tokenizer.add_special_tokens(
                {'eos_token': '<|endoftext|>'})
    if tokenizer._pad_token is None:
        tokenizer.add_special_tokens(
            {'pad_token': '[PAD]'})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens(
            {'bos_token': '[BOS]'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens(
            {'eos_token': '[EOS]'})
        # tokenizer.pad_token = 0.
    return tokenizer


def get_model(model_name: str,
              cache_dir: Path,
              pretrained: bool = True,
              weights: str = None,
              task: str = 'clm') -> torch.nn.Module:
    if model_name == 'gpt2':
        _model = GPT2LMHeadModel
        _config = GPT2Config()
    elif model_name == 'distilgpt2':
        _model = GPT2LMHeadModel
        _config = GPT2Config
    elif model_name == 'bert-base-uncased':
        if task == 'nmt':
            if weights is not None:
                _model = EncoderDecoderModel
                _config = EncoderDecoderConfig
            else:
                _config = EncoderDecoderConfig.from_encoder_decoder_configs(
                    BertConfig(), BertConfig(),
                    cache_dir=cache_dir)
                model = EncoderDecoderModel(config=_config)
                model.config.decoder.is_decoder = True
                model.config.decoder.add_cross_attention = True
            return model
        elif task == 'clm':
            _model = BertLMHeadModel
            _config = BertConfig
    elif model_name == 't5-small' and task == 'nmt':
        _model = T5ForConditionalGeneration
        _config = T5Config
    else:
        raise ValueError("Unknown model and/or task")
    if pretrained:
        model_name = weights if weights is not None else model_name
        model = _model.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            config=_config.from_pretrained(model_name) if _config is not
            None else None)
    else:
        model = _model(
            config=_config(**CONFIG_DICTS[model_name]))
    if task == 'nmt' and model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `decoder_start_token_id` is correctly defined")
    return model
