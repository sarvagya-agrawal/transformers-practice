"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace
from pathlib import Path

import torch
import tqdm

from ..models import get_model, get_tokenizer
from ..data.utils import load_data

DECODE_METHODS = ['greedy', 'beam']


def get_decoder(method: str) -> callable:
    if method == 'greedy':
        def greedy(model: torch.nn.Module, inputs: torch.Tensor,
                   attention_mask: torch.Tensor,
                   max_length: int, break_tokens: list[str]):
            # outputs = []
            # while decoded_token not in break_tokens:
            #     output = model.generate(
            #         inputs, max_length=max_length)[0]
            #     next_token = torch.argmax(output[0, -1, :]).item()
            return model.generate(inputs, max_length=max_length)
        return greedy
    elif method == 'beam':
        def beam_search(model: torch.nn.Module,
                        inputs: torch.Tensor,
                        attention_mask: torch.Tensor,
                        beam_depth: int = -1,
                        beam_width: int = -1):
            if not beam_width > 0 or not beam_depth > 0:
                raise ValueError("Beam depth and width must be > 0.")
            return model.generate(inputs, max_length=beam_depth,
                                  num_beam=beam_width, early_stopping=True,
                                  num_return_sequences=beam_width)
        return beam_search
    else:
        raise NotImplementedError(
            f"Unkown decoding method {method}")


def main(args: Namespace) -> None:
    model = get_model(args.decoder.model,
                      cache_dir=args.io.cache_dir,
                      pretrained=True)
    model.eval()
    device = torch.device('cuda', args.decode.gpu)
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
        batch_size=1,
        cache_dir=args.io.cache_dir,
        tgt_fname=None,
        overwrite_cache=args.data.overwrite_cache,
        num_workers=args.decode.num_workers,
        split='test',
        distributed=False)
    output_filename = Path(args.io.output) / 'decoded.txt'
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
        outputs_text = tokenizer.decode(outputs)
        output_filename.open('a+').write(outputs_text + '\n')
