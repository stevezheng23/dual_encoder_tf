import codecs
import json

import numpy as np
import tensorflow as tf

__all__ = ["load_hyperparams", "override_hyperparams", "create_default_hyperparams"]

def load_hyperparams(config_file):
    """load hyperparameters from config file"""
    if tf.gfile.Exists(config_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(config_file, "rb")) as file:
            hyperparams = create_default_hyperparams()
            hyperparams_override = json.load(file)
            hyperparams = override_hyperparams(hyperparams, hyperparams_override)
            
            return hyperparams
    else:
        raise FileNotFoundError("config file not found")

def override_hyperparams(hyperparams,
                         hyperparams_override):
    if (hyperparams is None or hyperparams_override is None):
        raise ValueError("objects not exist")
    
    if (type(hyperparams) != type(hyperparams_override)):
        raise TypeError("objects have different type")
    
    if isinstance(hyperparams, dict):
        for key in hyperparams_override.keys():
            if (key not in hyperparams or hyperparams[key] is None):
                hyperparams[key] = hyperparams_override[key]
            else:
                hyperparams[key] = override_hyperparams(hyperparams[key], hyperparams_override[key])
    else:
        hyperparams = hyperparams_override
    
    return hyperparams

def create_default_hyperparams():
    hyperparams = {
        "data": {
            "src_train_file": "",
            "trg_train_file": "",
            "src_eval_file": "",
            "trg_eval_file": "",
            "src_char_vocab_file": "",
            "trg_char_vocab_file": "",
            "src_subword_vocab_file": "",
            "trg_subword_vocab_file": "",
            "src_word_vocab_file": "",
            "trg_word_vocab_file": "",
            "src_embedding_file": "",
            "trg_embedding_file": "",
            "src_full_embedding_file": "",
            "trg_full_embedding_file": "",
            "src_vocab_size": 30000,
            "trg_vocab_size": 30000,
            "src_max_length": 50,
            "trg_max_length": 50,
            "share_vocab": False,
            "log_output_dir": "output/log",
            "result_output_dir": "output/result"
        },
        "train": {
            "random_seed": 100,
            "batch_size": 128,
            "eval_batch_size": 256,
            "eval_metric": "precision",
            "num_epoch": 3,
            "ckpt_output_dir": "output/checkpoint",
            "summary_output_dir": "output/summary",
            "step_per_stat": 10,
            "step_per_ckpt": 100,
            "step_per_eval": 100,
            "clip_norm": 5.0,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "adam_beta_1": 0.9,
                "adam_beta_2": 0.999,
                "adam_epsilon": 1e-08
            }
        },
        "model": {
            "type": "default",
            "scope": "dual_encoder",
            "src_encoder": {
                "char_embed_layer": None,
                "subword_embed_layer": None,
                "word_embed_layer": {
                    "layer_type": "sequence",
                    "sub_layers": [
                        {
                            "layer_type": "pretrain",
                            "unit_dim": 300,
                            "is_trainable": False
                        }
                    ]
                },
                "semantic_embed_layer": {
                    "layer_type": "sequence",
                    "sub_layers": [
                        {
                            "layer_type": "bi_rnn",
                            "unit_dim": 512,
                            "unit_type": "lstm",
                            "activation": "tanh",
                            "dropout": 0.1,
                            "is_trainable": True
                        }
                    ]
                }
            },
            "trg_encoder": {
                "char_embed_layer": None,
                "subword_embed_layer": None,
                "word_embed_layer": {
                    "layer_type": "sequence",
                    "sub_layers": [
                        {
                            "layer_type": "pretrain",
                            "unit_dim": 300,
                            "is_trainable": False
                        }
                    ]
                },
                "semantic_embed_layer": {
                    "layer_type": "sequence",
                    "sub_layers": [
                        {
                            "layer_type": "bi_rnn",
                            "unit_dim": 512,
                            "unit_type": "lstm",
                            "activation": "tanh",
                            "dropout": 0.1,
                            "is_trainable": True
                        }
                    ]
                }
            },
            "weight_sharing": {
                "share_char_embed": False,
                "share_subword_embed": False,
                "share_word_embed": False,
                "share_semantic_embed": False
            },
            "task_runner": {
                "task_type": "similarity",
                "loss_type": "log_loss"
            }
        },
        "device": {
            "num_gpus": 1,
            "default_gpu_id": 0,
            "log_device_placement": False,
            "allow_soft_placement": True,
            "allow_growth": False,
            "per_process_gpu_memory_fraction": 0.95
        }
    }
    
    return hyperparams
