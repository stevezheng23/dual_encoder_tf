import numpy as np
import tensorflow as tf

__all__ = ["create_default_hyperparams"]

def create_default_hyperparams():
    """create default hyperparameters"""   
    hyperparams = tf.contrib.training.HParams(
        """create data hyperparameters"""
        data=tf.contrib.training.HParams(
            src_train_file="",
            trg_train_file="",
            src_eval_file="",
            trg_eval_file="",
            src_char_vocab_file="",
            trg_char_vocab_file="",
            src_subword_vocab_file="",
            trg_subword_vocab_file="",
            src_word_vocab_file="",
            trg_word_vocab_file="",
            src_embedding_file="",
            trg_embedding_file="",
            src_full_embedding_file="",
            trg_full_embedding_file="",
            src_vocab_size=30000,
            trg_vocab_size=30000,
            src_max_length=50,
            trg_max_length=50,
            share_vocab=False,
            sos="<sos>",
            eos="<eos>",
            pad="<pad>",
            unk="<unk>",
            log_output_dir="",
            result_output_dir=""
        ),
        """create train hyperparameters"""
        train=tf.contrib.training.HParams(
            random_seed=0,
            batch_size=128,
            eval_batch_size=1024,
            eval_metric="precision",
            num_epoch=20,
            ckpt_output_dir="",
            summary_output_dir="",
            step_per_stat=10,
            step_per_ckpt=100,
            step_per_eval=100,
            clip_norm=5.0,
            """create optimizer hyperparameters"""
            optimizer=tf.contrib.training.HParams(
                type="adam",
                learning_rate=0.001,
                decay_mode="exponential_decay",
                decay_rate=0.95,
                decay_step=1000,
                decay_start_step=10000,
                momentum_beta=0.9,
                rmsprop_beta=0.999,
                rmsprop_epsilon=1e-8,
                adadelta_rho=0.95,
                adadelta_epsilon=1e-8,
                adagrad_init_accumulator=0.1,
                adam_beta_1=0.9,
                adam_beta_2=0.999,
                adam_epsilon=1e-08
            )
        ),
        """create model hyperparameters"""
        model=tf.contrib.training.HParams(
            type="default",
            scope="dual_encoder",
            """create source encoder hyperparameters"""
            src_encoder=tf.contrib.training.HParams(
                """create source char-level embed layer hyperparameters"""
                char_embed_layer=[
                    tf.contrib.training.HParams(
                        layer_type="conv",
                        embed_dim=100,
                        window_size=3,
                        activation="relu",
                        dropout=0.1,
                        enable_update=True
                    ),
                    tf.contrib.training.HParams(
                        layer_type="pool",
                        pooling_type="max"
                    )
                ],
                """create source subword-level embed layer hyperparameters"""
                subword_embed_layer=[
                    tf.contrib.training.HParams(
                        layer_type="conv",
                        embed_dim=100,
                        window_size=3,
                        activation="relu",
                        dropout=0.1,
                        enable_update=True
                    ),
                    tf.contrib.training.HParams(
                        layer_type="pool",
                        pooling_type="max"
                    )
                ],
                """create source word-level embed layer hyperparameters"""
                word_embed_layer=tf.contrib.training.HParams(
                    pretrained_embedding=True,
                    embed_dim=300
                ),
                """create source CoVe embed layer hyperparameters"""
                cove_embed_layer=tf.contrib.training.HParams(
                    pretrained_embedding=True,
                    embed_dim=300,
                    layer_type="bi_rnn",
                    num_layer=1,
                    unit_dim=512,
                    unit_type="lstm",
                    activation="tanh",
                    residual_connect=False,
                    forget_bias=1.0,
                    dropout=0.1
                ),
                """create source semantic embed layer hyperparameters"""
                semantic_embed_layer=[
                    tf.contrib.training.HParams(
                        layer_type="bi_rnn",
                        unit_dim=512,
                        unit_type="lstm",
                        activation="tanh",
                        residual_connect=False,
                        forget_bias=1.0,
                        dropout=0.1,
                        enable_update=True
                    ),
                    tf.contrib.training.HParams(
                        layer_type="bi_rnn",
                        unit_dim=512,
                        unit_type="lstm",
                        activation="tanh",
                        residual_connect=False,
                        forget_bias=1.0,
                        dropout=0.1,
                        enable_update=True
                    )
                ]
            ),
            """create target encoder hyperparameters"""
            trg_encoder=tf.contrib.training.HParams(
                """create target char-level embed layer hyperparameters"""
                char_embed_layer=[
                    tf.contrib.training.HParams(
                        layer_type="conv",
                        embed_dim=100,
                        window_size=3,
                        activation="relu",
                        dropout=0.1,
                        enable_update=True
                    ),
                    tf.contrib.training.HParams(
                        layer_type="pool",
                        pooling_type="max"
                    )
                ],
                """create target subword-level embed layer hyperparameters"""
                subword_embed_layer=[
                    tf.contrib.training.HParams(
                        layer_type="conv",
                        embed_dim=100,
                        window_size=3,
                        activation="relu",
                        dropout=0.1,
                        enable_update=True
                    ),
                    tf.contrib.training.HParams(
                        layer_type="pool",
                        pooling_type="max"
                    )
                ],
                """create target word-level embed layer hyperparameters"""
                word_embed_layer=tf.contrib.training.HParams(
                    pretrained_embedding=True,
                    embed_dim=300
                ),
                """create target CoVe embed layer hyperparameters"""
                cove_embed_layer=tf.contrib.training.HParams(
                    pretrained_embedding=True,
                    embed_dim=300,
                    encoder_type="bi_rnn",
                    num_layer=1,
                    unit_dim=512,
                    unit_type="lstm",
                    activation="tanh",
                    residual_connect=False,
                    forget_bias=1.0,
                    dropout=0.1
                ),
                """create target semantic embed layer hyperparameters"""
                semantic_embed_layer=[
                    tf.contrib.training.HParams(
                        layer_type="bi_rnn",
                        unit_dim=512,
                        unit_type="lstm",
                        activation="tanh",
                        residual_connect=False,
                        forget_bias=1.0,
                        dropout=0.1,
                        enable_update=True
                    ),
                    tf.contrib.training.HParams(
                        layer_type="bi_rnn",
                        unit_dim=512,
                        unit_type="lstm",
                        activation="tanh",
                        residual_connect=False,
                        forget_bias=1.0,
                        dropout=0.1,
                        enable_update=True
                    )
                ]
            ),
            """create weight sharing hyperparameters"""
            weight_sharing=tf.contrib.training.HParams(
                share_char_embed=False,
                share_subword_embed=False,
                share_word_embed=False,
                share_cove_embed=False,
                share_semantic_embed=False
            ),
            """create task runner hyperparameters"""
            task_runner=tf.contrib.training.HParams(
                task_type="similarity",
                loss_type="log_loss"
            )
        ),
        """create device hyperparameters"""
        device=tf.contrib.training.HParams(
            num_gpus=1,
            default_gpu_id=0,
            log_device_placement=False,
            allow_soft_placement=False,
            allow_growth=False,
            per_process_gpu_memory_fraction=0.95
        )
    )
    
    return hyperparams
