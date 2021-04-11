"""
@author: Mathieu Tuli, Sarvagya Agrawal
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from pathlib import Path
from typing import List

import time

# BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer, AlbertTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import (
    GPT2Config, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, T5Config,
    EncoderDecoderModel, EncoderDecoderConfig,
    BertConfig, BertModel, BertForMaskedLM, BertLMHeadModel,
    PreTrainedTokenizerFast
)
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE

import torch

from .configs import CONFIG_DICTS

TOKENIZERS = set(['bert-base-uncased', 'openai-gpt',
                  'gpt2', 'bert-base-uncased', 't5-small', 'BPE'])
MODELS = set(['gpt2', 'distilgpt2', 'bert-base-uncased', 't5-small'])


def get_tokenizer(tokenizer_name: str,
                  cache_dir: Path,
                  dataset: str,
                  task: str,
                  pretrained: True,
                  datasets: List[str] = None,
                  lower_case: bool = True) -> PreTrainedTokenizerBase:
    if pretrained:
        if tokenizer_name in TOKENIZERS:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                      cache_dir=cache_dir)
        else:
            raise ValueError(f'Unknown tokenizer. Must be one of {TOKENIZERS}')
    else:
        if tokenizer_name == 'BPE':
            # if not (Path(cache_dir) / 'BPE_custom.json').exists():
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            trainer = BpeTrainer(
                special_tokens=["[UNK]", "[BOS]", "[EOS]",
                                "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.train(datasets, trainer)
            stamp = time.time()
            tokenizer.save(cache_dir + "/" + f'BPE_custom_{stamp}.json')
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=cache_dir + "/" + f'BPE_custom_{stamp}.json')

    if dataset == 'multiwoz2.1':
        if task == 'clm':
            tokenizer.add_special_tokens(
                {'bos_token': '<|endoftext|>'})
            tokenizer.add_special_tokens(
                {'eos_token': '<|endoftext|>'})
    if tokenizer._pad_token is None:
        tokenizer.add_special_tokens(
            {'pad_token': '[PAD]'})
    if tokenizer._bos_token is None:
        tokenizer.add_special_tokens(
            {'bos_token': '[BOS]'})
    if tokenizer._eos_token is None:
        tokenizer.add_special_tokens(
            {'eos_token': '[EOS]'})
    if tokenizer._mask_token is None:
        tokenizer.add_special_tokens(
            {'eos_token': '[MASK]'})
    return tokenizer


def get_model(model_name: str,
              cache_dir: Path,
              tokenizer_len: int,
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
                if pretrained:
                    encoder_config = BertConfig()  # vocab_size=tokenizer_len)
                    encoder = BertModel.from_pretrained(
                        model_name, config=encoder_config,
                        cache_dir=cache_dir)
                    decoder_config = BertConfig()  # vocab_size=tokenizer_len,
                    # is_decoder=True,
                    # add_cross_attention=True)
                    decoder = BertLMHeadModel.from_pretrained(
                        model_name,
                        config=decoder_config,
                        cache_dir=cache_dir)
                    decoder.is_decoder = True
                    decoder.add_cross_atention = True
                    encoder.resize_token_embeddings(tokenizer_len)
                    decoder.resize_token_embeddings(tokenizer_len)
                    model = EncoderDecoderModel(
                        encoder=encoder, decoder=decoder)
                else:
                    pass
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
    model.resize_token_embeddings(tokenizer_len)
    return model
