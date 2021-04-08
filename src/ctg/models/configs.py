"""
@author: Mathieu Tuli, Sarvagya Agrawal
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
CONFIG_DICTS = {
    'distilgpt2': {
        "_name_or_path": "distilgpt2",
        "_num_labels": 1,
        "activation_function": "gelu_new",
        "architectures": [
            "GPT2LMHeadModel"
        ],
        "attn_pdrop": 0.1,
        "bos_token_id": 50256,
        "embd_pdrop": 0.1,
        "eos_token_id": 50256,
        "gradient_checkpointing": False,
        "id2label": {
            "0": "LABEL_0"
        },
        "initializer_range": 0.02,
        "label2id": {
            "LABEL_0": 0
        },
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_inner": None,
        "n_layer": 6,
        "n_positions": 1024,
        "resid_pdrop": 0.1,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {
            "text-generation": {
                "do_sample": True,
                "max_length": 50
            }
        },
        "transformers_version": "4.5.0.dev0",
        "use_cache": True,
        "vocab_size": 50257
    },
    't5-small':  {
        "_name_or_path": "t5-small",
        "architectures": [
            "T5WithLMHeadModel"
        ],
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 512,
        "decoder_start_token_id": 0,
        "dropout_rate": 0.1,
        "eos_token_id": 1,
        "feed_forward_proj": "relu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "layer_norm_epsilon": 1e-06,
        "model_type": "t5",
        "n_positions": 512,
        "num_decoder_layers": 6,
        "num_heads": 8,
        "num_layers": 6,
        "output_past": True,
        "pad_token_id": 0,
        "relative_attention_num_buckets": 32,
        "task_specific_params": {
            "summarization": {
                "early_stopping": True,
                "length_penalty": 2.0,
                "max_length": 200,
                "min_length": 30,
                "no_repeat_ngram_size": 3,
                "num_beams": 4,
                "prefix": "summarize: "
            },
            "translation_en_to_de": {
                "early_stopping": True,
                "max_length": 300,
                "num_beams": 4,
                "prefix": "translate English to German: "
            },
            "translation_en_to_fr": {
                "early_stopping": True,
                "max_length": 300,
                "num_beams": 4,
                "prefix": "translate English to French: "
            },
            "translation_en_to_ro": {
                "early_stopping": True,
                "max_length": 300,
                "num_beams": 4,
                "prefix": "translate English to Romanian: "
            }
        },
        "transformers_version": "4.5.0.dev0",
        "use_cache": True,
        "vocab_size": 32128
    }
}
