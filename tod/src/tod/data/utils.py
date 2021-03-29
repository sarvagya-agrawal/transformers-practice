"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from typing import List, Union, Dict, Tuple
from pathlib import PosixPath

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers import LineByLineTextDataset
from torch.utils.data import DataLoader, \
    RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


def load_data(src_fname: PosixPath,
              tokenizer: PreTrainedTokenizerBase,
              max_length: int,
              batch_size: int,
              cache_dir: PosixPath,
              tgt_fname: PosixPath = None,
              overwrite_cache: bool = False,
              num_workers: int = 4,
              split='train') -> None:
    if src_fname.suffix not in set(['.txt']) or not src_fname.exists():
        raise ValueError(f"Unknown src file {src_fname}. Files must be",
                         "line-by-line .txt files")
    if tgt_fname is not None and (tgt_fname.suffix not in set(['.txt']) or
                                  not tgt_fname.exists()):
        raise ValueError(f"Unknown tgt file {tgt_fname}. Files must be",
                         "line-by-line .txt files")
    if tgt_fname is not None:
        pass
    else:
        dataset = load_dataset('text',
                               data_files={split: str(src_fname)},
                               cache_dir=cache_dir, split=split)

    def tokenize(examples: List[str]) -> Tuple[List[int], None]:
        examples["text"] = [line for line in examples["text"] if len(line) > 0
                            and not line.isspace()]
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        return tokenizer(
            examples['text'],
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            truncation=True,)

    # def collate(examples):
    #     print(examples)
    #     if tokenizer._pad_token is None:
    #         return pad_sequence(examples, batch_first=True)
    #     return pad_sequence(
    #         examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=['text'],
        num_proc=num_workers,
        load_from_cache_file=not overwrite_cache
    )
    sampler = RandomSampler(dataset) if split == 'train'\
        else SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,)
    # collate_fn=collate)


def read_lines(filename: Union[str, PosixPath]) -> List[str]:
    """Read file and split into lines"""
    lines = open(filename).read().strip().split('\n')
    return [line for line in lines if (len(line) > 0 and not line.isspace())]


def vocabulary_indices(vocabulary: List[str]) -> Dict[str, int]:
    return {word: i for i, word in enumerate(sorted(list(vocabulary)))}
