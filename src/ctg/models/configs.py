"""
@author: Mathieu Tuli, Sarvagya Agrawal
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
CONFIG_DICTS = {
    'distlgpt2': {
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
    }
}
