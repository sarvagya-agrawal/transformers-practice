"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace
from pathlib import Path
from typing import List

import torch
import tqdm

from ..models import get_model, get_tokenizer
from ..data.utils import load_data
from ..utils.io import create_dir

DECODE_METHODS = ['greedy', 'beam']


def get_decoder(method: str) -> callable:
    if method == 'greedy':
        def greedy(model: torch.nn.Module, inputs: torch.Tensor,
                   attention_mask: torch.Tensor,
                   max_length: int, break_tokens: List[str]):
            # outputs = []
            # while decoded_token not in break_tokens:
            #     output = model.generate(
            #         inputs, max_length=max_length)[0]
            #     next_token = torch.argmax(output[0, -1, :]).item()
            # with torch.no_grad():
            outputs = model.generate(inputs, attention_mask=attention_mask,
                                     max_length=max_length)
            return outputs
        return greedy
    elif method == 'beam':
        def beam_search(model: torch.nn.Module,
                        inputs: torch.Tensor,
                        attention_mask: torch.Tensor,
                        beam_depth: int = -1,
                        beam_width: int = -1):
            if not beam_width > 0 or not beam_depth > 0:
                raise ValueError("Beam depth and width must be > 0.")
            # with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=beam_depth,
                num_beam=beam_width, early_stopping=True,
                num_return_sequences=beam_width)
            return outputs
        return beam_search
    else:
        raise NotImplementedError(
            f"Unkown decoding method {method}")


def main(args: Namespace) -> None:
    model = get_model(args.decode.model,
                      weights=args.decode.weights,
                      cache_dir=args.io.cache_dir,
                      pretrained=True)
    model.eval()
    device = torch.device('cuda', args.gpu)
    model.to(device)
    tokenizer = get_tokenizer(tokenizer_name=args.decode.tokenizer,
                              cache_dir=args.io.cache_dir,
                              dataset=args.data.name,
                              task=args.data.task)
    break_tokens = tokenizer.encode(tokenizer._eos_token)
    decoder = get_decoder(method=args.decode.method)
    data_loader, sampler = load_data(
        src_fname=Path(args.data.src),
        tokenizer=tokenizer,
        max_length=args.decode.max_length,
        max_samples=args.decode.max_samples,
        batch_size=args.decode.batch_size,
        cache_dir=args.io.cache_dir,
        overwrite_cache=args.data.overwrite_cache,
        num_workers=args.data.num_workers,
        split='test',
        distributed=False)
    output_dir = create_dir(Path(args.io.output))
    output_filename = output_dir / 'decoded.txt'
    output_filename.unlink(missing_ok=True)
    for step, batch in enumerate(
            tqdm.tqdm(data_loader,
                      desc='Decoding')):
        inputs = batch['input_ids'].to(
            device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(
            device, non_blocking=True)
        if args.decode.method == 'greedy':
            outputs = decoder(model, inputs,
                              attention_mask,
                              max_length=args.decode.max_length,
                              break_tokens=break_tokens)
        elif args.decode.method == 'beam':
            outputs = decoder(model, inputs,
                              attention_mask,
                              beam_depth=args.decode.beam_depth,
                              beam_width=args.decode.beam_width)
        text = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        for i, line in enumerate(text):
            output_filename.open('a+').write(line + '\n')
