"""
@author: Mathieu Tuli, Sarvagya Agrawal
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
# BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer, AlbertTokenizer
from pathlib import Path
from transformers import AutoConfig, \
    AutoTokenizer, AutoModelForSeq2SeqLM, \
    AutoModelForCausalLM


def get_model_tokenizer(
        model_name,
        tokenizer_name: str,
        task: str,
        model_pretrained: bool = False,
        tokenizer_pretrained: bool = True,
        hf_model_config: str = None,
        hf_model_config_pretrained: bool = False,
        cache_dir: Path = None):
    if hf_model_config is not None:
        config = AutoConfig.from_pretrained(hf_model_config) if \
            hf_model_config_pretrained else \
            AutoConfig(hf_model_config)
    else:
        config = AutoConfig.from_pretrained(model_name) if \
            hf_model_config_pretrained else \
            AutoConfig(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True) if \
        tokenizer_pretrained else None

    if task == 'nmt':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, config=config) if model_pretrained \
            else AutoModelForSeq2SeqLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model, config=config) if model_pretrained \
            else AutoModelForCausalLM.from_config(config)

    if task == 'nmt':
        if model.config.decoder_start_token_id is None:
            raise ValueError("Decoder start token id is None")
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer
