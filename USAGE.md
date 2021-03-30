# Task Oriented Dialogue Experiments

## General Training Pipeline
### Dataset Organization
The data is split into
- `src.train.txt`
- `tgt.train.txt`
- `src.val.txt`
- `tgt.val.txt`

Within each file, each line is a single training datum, and lines in the `src` and `tgt` files are aligned.

## Usage
###
```Python
python -m tod train onmt ...
```
```Python
python -m tod train custom ...
```
```Python
python -m tod test smcalflow ...
```
```Python
python -m tod data prep-smcalflow ...
```
```Python
python -m tod translate onmt ...
```
```Python
python -m tod translate custom ...
```
