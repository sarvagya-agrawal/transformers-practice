# Controllable Text Generation
Codebase for controllable text generation research, curated for CSC2516 project.

See `scripts/configs/*.yaml` for the various yaml configurations for each model trained.

See `scripts/experiments/*.sh` for various scripts to run various of the experiments, including text pre-processing.

@credits to the SMCalFlow and SimpleTOD papers/repos for some of their code, which we based our experiments off of.

For generating the smcalflow dataset, download the jsonl files from [smcaflow](https://microsoft.github.io/task_oriented_dialogue_as_dataflow_synthesis/). Then run the relevant scripts from [their repo](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis).

For generating the MultiWOZ dataset, clone the [SimpleTOD](https://github.com/salesforce/simpletod) repo and run `create_dataset.sh`. They have a simple wrapper around the original MultiWOZ processing that simply converts each turn to a string form. But this won't work for our case. Using the output files `resources/gpt2/{train,val,test}.history_belief`, we have a script to convert and further process the data for our case. See `scripts/experiments/convert_multiwoz.py`. The `v2` version allows for dynamic context generation.

In each case, the output will be either `{train,valid,test}.{src,tgt,src_tok}`. In the case of SMCalflow, use the `.src_tok` files, and for MultiWOZ, well there's only the `.src` files. Then, we can change the config files in `scripts/configs` accordingly to point to the proper `src` and `tgt` files, and to train, simply run

```
onmt_build_vocab --config *config.yaml
onmt_train --config *config.yaml
```

We used to have a wrapper around the onmt but found it to be messy, so resolved to using their default trainer which works fine. The `ctg` package itself does have the ability to train a transformer from scratch using huggingface and a custom train loop, but we found better results with OpenNMT, so ended up using it for the report.

Once trained, you can generate a report of exact level matching as follows:

```Python
onmt_translate \
    --model *model-.pt* \
    --max_length 512 \
    --src {valid,test}.src \
    --replace_unk \
    --n_best 5 \
    --batch_size 28 \
    --beam_size 10 \
    --gpu 0 \
    --report_time \
    --output {FOLDER}/{valid,test}.nbest

python -m ctg test smcalflow
  --data *jsonl files* \
  --datum-ids *datum-id-files* \
  --src {valid,test}.{src,src_tok} \
  --tgt {valid,test}.tgt \
  --nbest-txt {valid,test}.nbest.txt \
  --nbest 5 \
  --output ? \
  --scores-json *output-scores-name
```

Note that jsonl files an datum-id files can be generated for MultiWOZ using the follow command:
```
#!/bin/sh
raw_trade_dialogues_dir="output/trade_dialogues"
mkdir -p "${raw_trade_dialogues_dir}"
python -m dataflow.multiwoz.trade_dst.create_data \
    --use_multiwoz_2_1 \
    --output_dir ${raw_trade_dialogues_dir}

# patch TRADE dialogues
patched_trade_dialogues_dir="output/patched_trade_dialogues"
mkdir -p "${patched_trade_dialogues_dir}"
for subset in "train" "dev" "test"; do
    python -m dataflow.multiwoz.patch_trade_dialogues \
        --trade_data_file ${raw_trade_dialogues_dir}/${subset}_dials.json \
        --outbase ${patched_trade_dialogues_dir}/${subset}
done

```
using the smcalflow repo.

Note that to replicate our experiments, you must do the following:
1. Train BLSTM config on the basic SMCalFlow and MWOZ dataset
2. Train Transformer config on the basic SMCalFlow and MWOZ dataset
3. Preprocess smcalflow to have context {2,4,10} ({1,3,9} in paper) and train the blstm
4. Preprocess multiwoz to remove taxi domain (see commented out code in `scripts/experiments/convert_multiwoz.py` and train BLSTM/Transformer
5. Train the transformer on the CCN/DailyMail dataset (see this [repo](https://github.com/becxer/cnn-dailymail/) for example. There are many to use.), then train on MultiWOZ, using same config (but increase train steps).
