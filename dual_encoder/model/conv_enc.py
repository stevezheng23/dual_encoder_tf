import collections
import functools
import os.path
import operator

import numpy as np
import tensorflow as tf

from functools import reduce

from util.default_util import *
from util.dual_encoder_util import *
from util.layer_util import *

from model.base_model import *

__all__ = ["ConvolutionEncoder"]

class ConvolutionEncoder(BaseModel):
    """convolution encoder model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 external_data,
                 mode="train",
                 scope="conv_enc"):
        """initialize convolution encoder model"""
        super(ConvolutionEncoder, self).__init__(logger=logger, hyperparams=hyperparams,
            data_pipeline=data_pipeline, external_data=external_data, mode=mode, scope=scope)
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                initializer=tf.zeros_initializer, trainable=False)
            
            """get batch input from data pipeline"""
            input_src_word = self.data_pipeline.input_src_word
            input_src_word_mask = self.data_pipeline.input_src_word_mask
            input_src_char = self.data_pipeline.input_src_char
            input_src_char_mask = self.data_pipeline.input_src_char_mask
            input_trg_word = self.data_pipeline.input_trg_word
            input_trg_word_mask = self.data_pipeline.input_trg_word_mask
            input_trg_char = self.data_pipeline.input_trg_char
            input_trg_char_mask = self.data_pipeline.input_trg_char_mask
            label = self.data_pipeline.input_label
            label_mask = self.data_pipeline.input_label_mask
            
            if self.enable_neg_sampling == True:
                self.indice_list = self._neg_sampling_indice(self.batch_size, self.neg_num, self.random_seed)
                self.label = tf.reshape(tf.convert_to_tensor(self.indice_list, dtype=tf.float32),
                    shape=[self.batch_size, self.neg_num+1, 1])
                self.label_mask = self.label
            
            """build graph for convolution encoder"""
            self.logger.log_print("# build graph")
            predict, predict_mask = self._build_graph(input_src_word, input_src_word_mask, input_src_char,
                input_src_char_mask, input_trg_word, input_trg_word_mask, input_trg_char, input_trg_char_mask)
            self.predict = predict
            self.predict_mask = predict_mask
            
            if self.hyperparams.train_ema_enable == True:
                self.ema = self._get_exponential_moving_average(self.global_step)
                self.variable_list = self.ema.variables_to_restore(tf.trainable_variables())
            else:
                self.variable_list = tf.global_variables()
            
            if self.mode == "infer":
                """get infer answer"""
                self.infer_predict = tf.nn.sigmoid(self.predict)
                self.infer_predict_mask = self.predict_mask
                
                """create infer summary"""
                self.infer_summary = self._get_infer_summary()
            
            if self.mode == "train":
                """compute optimization loss"""
                self.logger.log_print("# setup loss computation mechanism")
                self.train_loss = self._compute_loss(label, label_mask, self.predict, self.predict_mask)
                
                if self.hyperparams.train_regularization_enable == True:
                    regularization_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    regularization_loss = tf.contrib.layers.apply_regularization(self.regularizer, regularization_variables)
                    self.train_loss = self.train_loss + regularization_loss
                
                """apply learning rate warm-up & decay"""
                self.logger.log_print("# setup initial learning rate mechanism")
                self.initial_learning_rate = tf.constant(self.hyperparams.train_optimizer_learning_rate)
                
                if self.hyperparams.train_optimizer_warmup_enable == True:
                    self.logger.log_print("# setup learning rate warm-up mechanism")
                    self.warmup_learning_rate = self._apply_learning_rate_warmup(self.initial_learning_rate)
                else:
                    self.warmup_learning_rate = self.initial_learning_rate
                
                if self.hyperparams.train_optimizer_decay_enable == True:
                    self.logger.log_print("# setup learning rate decay mechanism")
                    self.decayed_learning_rate = self._apply_learning_rate_decay(self.warmup_learning_rate)
                else:
                    self.decayed_learning_rate = self.warmup_learning_rate
                
                self.learning_rate = self.decayed_learning_rate
                
                """initialize optimizer"""
                self.logger.log_print("# setup training optimizer")
                self.optimizer = self._initialize_optimizer(self.learning_rate)
                
                """minimize optimization loss"""
                self.logger.log_print("# setup loss minimization mechanism")
                self.opt_op, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                if self.hyperparams.train_ema_enable == True:
                    with tf.control_dependencies([self.opt_op]):
                        self.update_op = self.ema.apply(tf.trainable_variables())
                else:
                    self.update_op = self.opt_op
                
                """create train summary"""
                self.train_summary = self._get_train_summary()
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            
            self.ckpt_debug_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "debug")
            self.ckpt_epoch_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "epoch")
            
            if not tf.gfile.Exists(self.ckpt_debug_dir):
                tf.gfile.MakeDirs(self.ckpt_debug_dir)
            
            if not tf.gfile.Exists(self.ckpt_epoch_dir):
                tf.gfile.MakeDirs(self.ckpt_epoch_dir)
            
            self.ckpt_debug_name = os.path.join(self.ckpt_debug_dir, "model_debug_ckpt")
            self.ckpt_epoch_name = os.path.join(self.ckpt_epoch_dir, "model_epoch_ckpt")
            
            if self.mode == "infer":
                self.ckpt_debug_saver = tf.train.Saver(self.variable_list)
                self.ckpt_epoch_saver = tf.train.Saver(self.variable_list, max_to_keep=self.hyperparams.train_num_epoch)  
            
            if self.mode == "train":
                self.ckpt_debug_saver = tf.train.Saver()
                self.ckpt_epoch_saver = tf.train.Saver(max_to_keep=self.hyperparams.train_num_epoch)   
    
    def _build_representation_layer(self,
                                    input_src_word,
                                    input_src_word_mask,
                                    input_src_char,
                                    input_src_char_mask,
                                    input_trg_word,
                                    input_trg_word_mask,
                                    input_trg_char,
                                    input_trg_char_mask):
        """build representation layer for convolution encoder model"""
        src_word_vocab_size = self.hyperparams.data_src_word_vocab_size
        src_word_embed_dim = self.hyperparams.model_representation_src_word_embed_dim
        src_word_dropout = self.hyperparams.model_representation_src_word_dropout if self.mode == "train" else 0.0
        src_word_embed_pretrained = self.hyperparams.model_representation_src_word_embed_pretrained
        src_word_feat_trainable = self.hyperparams.model_representation_src_word_feat_trainable
        src_word_feat_enable = self.hyperparams.model_representation_src_word_feat_enable
        src_char_vocab_size = self.hyperparams.data_src_char_vocab_size
        src_char_embed_dim = self.hyperparams.model_representation_src_char_embed_dim
        src_char_unit_dim = self.hyperparams.model_representation_src_char_unit_dim
        src_char_window_size = self.hyperparams.model_representation_src_char_window_size
        src_char_hidden_activation = self.hyperparams.model_representation_src_char_hidden_activation
        src_char_dropout = self.hyperparams.model_representation_src_char_dropout if self.mode == "train" else 0.0
        src_char_pooling_type = self.hyperparams.model_representation_src_char_pooling_type
        src_char_feat_trainable = self.hyperparams.model_representation_src_char_feat_trainable
        src_char_feat_enable = self.hyperparams.model_representation_src_char_feat_enable
        src_fusion_type = self.hyperparams.model_representation_src_fusion_type
        src_fusion_num_layer = self.hyperparams.model_representation_src_fusion_num_layer
        src_fusion_unit_dim = self.hyperparams.model_representation_src_fusion_unit_dim
        src_fusion_hidden_activation = self.hyperparams.model_representation_src_fusion_hidden_activation
        src_fusion_dropout = self.hyperparams.model_representation_src_fusion_dropout if self.mode == "train" else 0.0
        src_fusion_trainable = self.hyperparams.model_representation_src_fusion_trainable
        trg_word_vocab_size = self.hyperparams.data_trg_word_vocab_size
        trg_word_embed_dim = self.hyperparams.model_representation_trg_word_embed_dim
        trg_word_dropout = self.hyperparams.model_representation_trg_word_dropout if self.mode == "train" else 0.0
        trg_word_embed_pretrained = self.hyperparams.model_representation_trg_word_embed_pretrained
        trg_word_feat_trainable = self.hyperparams.model_representation_trg_word_feat_trainable
        trg_word_feat_enable = self.hyperparams.model_representation_trg_word_feat_enable
        trg_char_vocab_size = self.hyperparams.data_trg_char_vocab_size
        trg_char_embed_dim = self.hyperparams.model_representation_trg_char_embed_dim
        trg_char_unit_dim = self.hyperparams.model_representation_trg_char_unit_dim
        trg_char_window_size = self.hyperparams.model_representation_trg_char_window_size
        trg_char_hidden_activation = self.hyperparams.model_representation_trg_char_hidden_activation
        trg_char_dropout = self.hyperparams.model_representation_trg_char_dropout if self.mode == "train" else 0.0
        trg_char_pooling_type = self.hyperparams.model_representation_trg_char_pooling_type
        trg_char_feat_trainable = self.hyperparams.model_representation_trg_char_feat_trainable
        trg_char_feat_enable = self.hyperparams.model_representation_trg_char_feat_enable
        trg_fusion_type = self.hyperparams.model_representation_trg_fusion_type
        trg_fusion_num_layer = self.hyperparams.model_representation_trg_fusion_num_layer
        trg_fusion_unit_dim = self.hyperparams.model_representation_trg_fusion_unit_dim
        trg_fusion_hidden_activation = self.hyperparams.model_representation_trg_fusion_hidden_activation
        trg_fusion_dropout = self.hyperparams.model_representation_trg_fusion_dropout if self.mode == "train" else 0.0
        trg_fusion_trainable = self.hyperparams.model_representation_trg_fusion_trainable
        share_representation = self.hyperparams.model_share_representation
        
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            input_src_feat_list = []
            input_src_feat_mask_list = []
            input_trg_feat_list = []
            input_trg_feat_mask_list = []
            
            if src_word_feat_enable == True:
                self.logger.log_print("# build word-level source representation layer")
                src_word_feat_layer = WordFeat(vocab_size=src_word_vocab_size, embed_dim=src_word_embed_dim,
                    dropout=src_word_dropout, pretrained=src_word_embed_pretrained, embedding=self.src_word_embedding,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer, 
                    random_seed=self.random_seed, trainable=src_word_feat_trainable)
                
                (input_src_word_feat,
                    input_src_word_feat_mask) = src_word_feat_layer(input_src_word, input_src_word_mask)
                input_src_feat_list.append(input_src_word_feat)
                input_src_feat_mask_list.append(input_src_word_feat_mask)
                
                src_word_unit_dim = src_word_embed_dim
                self.src_word_embedding_placeholder = src_word_feat_layer.get_embedding_placeholder()
            else:
                src_word_unit_dim = 0
                self.src_word_embedding_placeholder = None
            
            if src_char_feat_enable == True:
                self.logger.log_print("# build char-level source representation layer")
                src_char_feat_layer = CharFeat(vocab_size=src_char_vocab_size,
                    embed_dim=src_char_embed_dim, unit_dim=src_char_unit_dim, window_size=src_char_window_size,
                    activation=src_char_hidden_activation, pooling_type=src_char_pooling_type, dropout=src_char_dropout,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=src_char_feat_trainable)
                
                (input_src_char_feat,
                    input_src_char_feat_mask) = src_char_feat_layer(input_src_char, input_src_char_mask)
                
                input_src_feat_list.append(input_src_char_feat)
                input_src_feat_mask_list.append(input_src_char_feat_mask)
            else:
                src_char_unit_dim = 0
            
            self.logger.log_print("# build source representation fusion layer")
            src_feat_unit_dim = src_word_unit_dim + src_char_unit_dim
            src_feat_fusion_layer = FusionModule(input_unit_dim=src_feat_unit_dim, output_unit_dim=src_fusion_unit_dim,
                fusion_type=src_fusion_type, num_layer=src_fusion_num_layer, activation=src_fusion_hidden_activation,
                dropout=src_fusion_dropout, num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id,
                regularizer=self.regularizer, random_seed=self.random_seed, trainable=src_fusion_trainable)
            
            (input_src_feat,
                input_src_feat_mask) = src_feat_fusion_layer(input_src_feat_list, input_src_feat_mask_list)
            
            if trg_word_feat_enable == True:
                self.logger.log_print("# build word-level target representation layer")
                if share_representation == True:
                    trg_word_feat_layer = src_word_feat_layer
                else:
                    trg_word_feat_layer = WordFeat(vocab_size=trg_word_vocab_size, embed_dim=trg_word_embed_dim,
                        dropout=trg_word_dropout, pretrained=trg_word_embed_pretrained, embedding=self.trg_word_embedding,
                        num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer, 
                        random_seed=self.random_seed, trainable=trg_word_feat_trainable)
                
                (input_trg_word_feat,
                    input_trg_word_feat_mask) = trg_word_feat_layer(input_trg_word, input_trg_word_mask)
                input_trg_feat_list.append(input_trg_word_feat)
                input_trg_feat_mask_list.append(input_trg_word_feat_mask)
                
                trg_word_unit_dim = trg_word_embed_dim
                self.trg_word_embedding_placeholder = trg_word_feat_layer.get_embedding_placeholder()
            else:
                trg_word_unit_dim = 0
                self.trg_word_embedding_placeholder = None
            
            if trg_char_feat_enable == True:
                self.logger.log_print("# build char-level target representation layer")
                if share_representation == True:
                    trg_char_feat_layer = src_char_feat_layer
                else:
                    trg_char_feat_layer = CharFeat(vocab_size=trg_char_vocab_size,
                        embed_dim=trg_char_embed_dim, unit_dim=trg_char_unit_dim, window_size=trg_char_window_size,
                        activation=trg_char_hidden_activation, pooling_type=trg_char_pooling_type, dropout=trg_char_dropout,
                        num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                        random_seed=self.random_seed, trainable=trg_char_feat_trainable)
                
                (input_trg_char_feat,
                    input_trg_char_feat_mask) = trg_char_feat_layer(input_trg_char, input_trg_char_mask)
                
                input_trg_feat_list.append(input_trg_char_feat)
                input_trg_feat_mask_list.append(input_trg_char_feat_mask)
            else:
                trg_char_unit_dim = 0
            
            self.logger.log_print("# build target representation fusion layer")
            if share_representation == True:
                trg_feat_fusion_layer = src_feat_fusion_layer
            else:
                trg_feat_unit_dim = trg_word_unit_dim + trg_char_unit_dim
                trg_feat_fusion_layer = FusionModule(input_unit_dim=trg_feat_unit_dim, output_unit_dim=trg_fusion_unit_dim,
                    fusion_type=trg_fusion_type, num_layer=trg_fusion_num_layer, activation=trg_fusion_hidden_activation,
                    dropout=trg_fusion_dropout, num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id,
                    regularizer=self.regularizer, random_seed=self.random_seed, trainable=trg_fusion_trainable)
            
            (input_trg_feat,
                input_trg_feat_mask) = trg_feat_fusion_layer(input_trg_feat_list, input_trg_feat_mask_list)
        
        return input_src_feat, input_src_feat_mask, input_trg_feat, input_trg_feat_mask
    
    def _build_understanding_layer(self,
                                   input_src_feat,
                                   input_src_feat_mask,
                                   input_trg_feat,
                                   input_trg_feat_mask):
        """build understanding layer for convolution encoder model"""
        src_representation_unit_dim = self.hyperparams.model_representation_src_fusion_unit_dim
        src_understanding_num_layer = self.hyperparams.model_understanding_src_num_layer
        src_understanding_num_conv = self.hyperparams.model_understanding_src_num_conv
        src_understanding_unit_dim = self.hyperparams.model_understanding_src_unit_dim
        src_understanding_window_size = self.hyperparams.model_understanding_src_window_size
        src_understanding_hidden_activation = self.hyperparams.model_understanding_src_hidden_activation
        src_understanding_dropout = self.hyperparams.model_understanding_src_dropout if self.mode == "train" else 0.0
        src_understanding_layer_dropout = self.hyperparams.model_understanding_src_layer_dropout if self.mode == "train" else 0.0
        src_understanding_trainable = self.hyperparams.model_understanding_src_trainable
        trg_representation_unit_dim = self.hyperparams.model_representation_trg_fusion_unit_dim
        trg_understanding_num_layer = self.hyperparams.model_understanding_trg_num_layer
        trg_understanding_num_conv = self.hyperparams.model_understanding_trg_num_conv
        trg_understanding_unit_dim = self.hyperparams.model_understanding_trg_unit_dim
        trg_understanding_window_size = self.hyperparams.model_understanding_trg_window_size
        trg_understanding_hidden_activation = self.hyperparams.model_understanding_trg_hidden_activation
        trg_understanding_dropout = self.hyperparams.model_understanding_trg_dropout if self.mode == "train" else 0.0
        trg_understanding_layer_dropout = self.hyperparams.model_understanding_trg_layer_dropout if self.mode == "train" else 0.0
        trg_understanding_trainable = self.hyperparams.model_understanding_trg_trainable
        share_understanding = self.hyperparams.model_share_understanding
        
        with tf.variable_scope("understanding", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("source", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build source understanding layer")
                src_fusion_layer = FusionModule(input_unit_dim=src_representation_unit_dim,
                    output_unit_dim=src_understanding_unit_dim, fusion_type="concate", num_layer=1, activation=None, dropout=0.0,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=src_understanding_trainable)
                src_understanding_layer = StackedConvolutionBlock(num_layer=src_understanding_num_layer,
                    num_conv=src_understanding_num_conv, unit_dim=src_understanding_unit_dim,
                    window_size=src_understanding_window_size, activation=src_understanding_hidden_activation,
                    dropout=src_understanding_dropout, layer_dropout=src_understanding_layer_dropout,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=src_understanding_trainable)
                
                (input_src_fusion,
                    input_src_fusion_mask) = src_fusion_layer([input_src_feat], [input_src_feat_mask])
                (input_src_understanding_list,
                    input_src_understanding_mask_list) = src_understanding_layer(input_src_fusion, input_src_fusion_mask)
                input_src_understanding = input_src_understanding_list[-1]
                input_src_understanding_mask = input_src_understanding_mask_list[-1]
            
            with tf.variable_scope("target", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build target understanding layer")
                if share_understanding == True:
                    trg_fusion_layer = src_fusion_layer
                    trg_understanding_layer = src_understanding_layer
                else:
                    trg_fusion_layer = FusionModule(input_unit_dim=trg_representation_unit_dim,
                        output_unit_dim=trg_understanding_unit_dim, fusion_type="concate", num_layer=1, activation=None, dropout=0.0,
                        num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                        random_seed=self.random_seed, trainable=trg_understanding_trainable)
                    trg_understanding_layer = StackedConvolutionBlock(num_layer=trg_understanding_num_layer,
                        num_conv=trg_understanding_num_conv, unit_dim=trg_understanding_unit_dim,
                        window_size=trg_understanding_window_size, activation=trg_understanding_hidden_activation,
                        dropout=trg_understanding_dropout, layer_dropout=trg_understanding_layer_dropout,
                        num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                        random_seed=self.random_seed, trainable=trg_understanding_trainable)
                
                (input_trg_fusion,
                    input_trg_fusion_mask) = trg_fusion_layer([input_trg_feat], [input_trg_feat_mask])
                (input_trg_understanding_list,
                    input_trg_understanding_mask_list) = trg_understanding_layer(input_trg_fusion, input_trg_fusion_mask)
                input_trg_understanding = input_trg_understanding_list[-1]
                input_trg_understanding_mask = input_trg_understanding_mask_list[-1]
            
            if self.enable_negative_sampling == True:
                (input_src_understanding, input_src_understanding_mask, input_trg_understanding,
                    input_trg_understanding_mask) = self.negative_sampling(input_src_understanding,
                        input_src_understanding_mask, input_trg_understanding, input_trg_understanding_mask,
                        self.batch_size, self.neg_num, self.random_seed, self.indice_list)
        
        return input_src_understanding, input_src_understanding_mask, input_trg_understanding, input_trg_understanding_mask
    
    def _build_interaction_layer(self,
                                 input_src_understanding,
                                 input_src_understanding_mask,
                                 input_trg_understanding,
                                 input_trg_understanding_mask):
        """build interaction layer for convolution encoder model"""
        src_understanding_unit_dim = self.hyperparams.model_understanding_src_unit_dim
        trg_understanding_unit_dim = self.hyperparams.model_understanding_trg_unit_dim
        src2trg_interaction_attention_dim = self.hyperparams.model_interaction_src2trg_attention_dim
        src2trg_interaction_score_type = self.hyperparams.model_interaction_src2trg_score_type
        src2trg_interaction_dropout = self.hyperparams.model_interaction_src2trg_dropout if self.mode == "train" else 0.0
        src2trg_interaction_att_dropout = self.hyperparams.model_interaction_src2trg_attention_dropout if self.mode == "train" else 0.0
        src2trg_interaction_trainable = self.hyperparams.model_interaction_src2trg_trainable
        src2trg_interaction_enable = self.hyperparams.model_interaction_src2trg_enable
        src_fusion_type = self.hyperparams.model_interaction_src_fusion_type
        src_fusion_num_layer = self.hyperparams.model_interaction_src_fusion_num_layer
        src_fusion_unit_dim = self.hyperparams.model_interaction_src_fusion_unit_dim
        src_fusion_hidden_activation = self.hyperparams.model_interaction_src_fusion_hidden_activation
        src_fusion_dropout = self.hyperparams.model_interaction_src_fusion_dropout if self.mode == "train" else 0.0
        src_fusion_trainable = self.hyperparams.model_interaction_src_fusion_trainable
        trg2src_interaction_attention_dim = self.hyperparams.model_interaction_trg2src_attention_dim
        trg2src_interaction_score_type = self.hyperparams.model_interaction_trg2src_score_type
        trg2src_interaction_dropout = self.hyperparams.model_interaction_trg2src_dropout if self.mode == "train" else 0.0
        trg2src_interaction_att_dropout = self.hyperparams.model_interaction_trg2src_attention_dropout if self.mode == "train" else 0.0
        trg2src_interaction_trainable = self.hyperparams.model_interaction_trg2src_trainable
        trg2src_interaction_enable = self.hyperparams.model_interaction_trg2src_enable
        trg_fusion_type = self.hyperparams.model_interaction_trg_fusion_type
        trg_fusion_num_layer = self.hyperparams.model_interaction_trg_fusion_num_layer
        trg_fusion_unit_dim = self.hyperparams.model_interaction_trg_fusion_unit_dim
        trg_fusion_hidden_activation = self.hyperparams.model_interaction_trg_fusion_hidden_activation
        trg_fusion_dropout = self.hyperparams.model_interaction_trg_fusion_dropout if self.mode == "train" else 0.0
        trg_fusion_trainable = self.hyperparams.model_interaction_trg_fusion_trainable
        share_interaction = self.hyperparams.model_share_interaction
        
        with tf.variable_scope("interaction", reuse=tf.AUTO_REUSE):
            attention_matrix = None
            with tf.variable_scope("source2target", reuse=tf.AUTO_REUSE):
                input_src_interaction_list = [input_src_understanding]
                input_src_interaction_mask_list = [input_src_understanding_mask]
                src_interaction_unit_dim = src_understanding_unit_dim
                
                if src2trg_interaction_enable == True:
                    self.logger.log_print("# build source2target interaction layer")
                    src2trg_interaction_layer = create_attention_layer("att",
                        src_understanding_unit_dim, trg_understanding_unit_dim,
                        src2trg_interaction_attention_dim, -1, src2trg_interaction_score_type,
                        src2trg_interaction_dropout, src2trg_interaction_att_dropout, 0.0,
                        False, False, False, None, self.num_gpus, self.default_gpu_id,
                        self.regularizer, self.random_seed, src2trg_interaction_trainable)
                    
                    (input_src2trg_interaction, input_src2trg_interaction_mask,
                        _, _)= src2trg_interaction_layer(input_src_understanding, input_trg_understanding,
                            input_src_understanding_mask, input_trg_understanding_mask)
                    
                    if share_interaction == True:
                        attention_matrix = src2trg_interaction_layer.get_attention_matrix()
                    
                    input_src_interaction_list.append(input_src2trg_interaction)
                    input_src_interaction_mask_list.append(input_src2trg_interaction_mask)
                    src_interaction_unit_dim += trg_understanding_unit_dim
                
                self.logger.log_print("# build source interaction fusion layer")
                src_fusion_layer = FusionModule(input_unit_dim=src_interaction_unit_dim,
                    output_unit_dim=src_fusion_unit_dim, fusion_type=src_fusion_type,
                    num_layer=src_fusion_num_layer, activation=src_fusion_hidden_activation,
                    dropout=src_fusion_dropout, num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id,
                    regularizer=self.regularizer, random_seed=self.random_seed, trainable=src_fusion_trainable)
                
                (input_src_interaction,
                    input_src_interaction_mask) = src_fusion_layer(input_src_interaction_list,
                        input_src_interaction_mask_list)
            
            with tf.variable_scope("target2source", reuse=tf.AUTO_REUSE):
                input_trg_interaction_list = [input_trg_understanding]
                input_trg_interaction_mask_list = [input_trg_understanding_mask]
                trg_interaction_unit_dim = trg_understanding_unit_dim
                
                if trg2src_interaction_enable == True:
                    self.logger.log_print("# build target2source interaction layer")
                    trg2src_interaction_layer = create_attention_layer("att",
                        trg_understanding_unit_dim, src_understanding_unit_dim,
                        trg2src_interaction_attention_dim, -1, trg2src_interaction_score_type,
                        trg2src_interaction_dropout, trg2src_interaction_att_dropout, 0.0,
                        False, False, False, attention_matrix, self.num_gpus, self.default_gpu_id,
                        self.regularizer, self.random_seed, trg2src_interaction_trainable)
                    
                    (input_trg2src_interaction, input_trg2src_interaction_mask,
                        _, _) = trg2src_interaction_layer(input_trg_understanding, input_src_understanding,
                            input_trg_understanding_mask, input_src_understanding_mask)
                    
                    input_trg_interaction_list.append(input_trg2src_interaction)
                    input_trg_interaction_mask_list.append(input_trg2src_interaction_mask)
                    trg_interaction_unit_dim += src_understanding_unit_dim
                
                self.logger.log_print("# build target interaction fusion layer")
                if share_representation == True:
                    trg_fusion_layer = src_fusion_layer
                else:
                    trg_fusion_layer = FusionModule(input_unit_dim=trg_interaction_unit_dim,
                        output_unit_dim=trg_fusion_unit_dim, fusion_type=trg_fusion_type,
                        num_layer=trg_fusion_num_layer, activation=trg_fusion_hidden_activation,
                        dropout=trg_fusion_dropout, num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id,
                        regularizer=self.regularizer, random_seed=self.random_seed, trainable=trg_fusion_trainable)
                
                (input_trg_interaction,
                    input_trg_interaction_mask) = trg_fusion_layer(input_trg_interaction_list,
                        input_trg_interaction_mask_list)
        
        return input_src_interaction, input_src_interaction_mask, input_trg_interaction, input_trg_interaction_mask
    
    def _build_matching_layer(self,
                              input_src_interaction,
                              input_src_interaction_mask,
                              input_trg_interaction,
                              input_trg_interaction_mask):
        """build matching layer for convolution encoder model"""
        matching_score_type = self.hyperparams.model_matching_score_type
        matching_pooling_type = self.hyperparams.model_matching_pooling_type
        matching_num_layer = self.hyperparams.model_matching_num_layer
        matching_unit_dim = self.hyperparams.model_matching_unit_dim
        matching_hidden_activation = self.hyperparams.model_matching_hidden_activation
        matching_dropout = self.hyperparams.model_matching_dropout if self.mode == "train" else 0.0
        matching_trainable = self.hyperparams.model_matching_trainable
        
        with tf.variable_scope("matching", reuse=tf.AUTO_REUSE):
            if matching_score_type == "cosine":
                score_layer = CosineScore(pooling_type=matching_pooling_type,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id)
            elif matching_score_type == "dense":
                score_layer = DenseScore(pooling_type=matching_pooling_type, num_layer=matching_num_layer,
                    unit_dim=matching_unit_dim, activation=matching_hidden_activation, dropout=matching_dropout,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=matching_trainable)
            else:
                raise ValueError("unsupported score type {0}".format(matching_score_type))
            
            input_matching, input_matching_mask = score_layer(input_src_interaction,
                input_src_interaction_mask, input_trg_interaction, input_trg_interaction_mask)
        
        return input_matching, input_matching_mask
    
    def _build_graph(self,
                     input_src_word,
                     input_src_word_mask,
                     input_src_char,
                     input_src_char_mask,
                     input_trg_word,
                     input_trg_word_mask,
                     input_trg_char,
                     input_trg_char_mask):
        """build graph for convolution encoder model"""
        with tf.variable_scope("graph", reuse=tf.AUTO_REUSE):
            (input_src_feat, input_src_feat_mask, input_trg_feat,
                input_trg_feat_mask) = self._build_representation_layer(input_src_word,
                    input_src_word_mask, input_src_char, input_src_char_mask, input_trg_word,
                    input_trg_word_mask, input_trg_char, input_trg_char_mask)
            
            (input_src_understanding, input_src_understanding_mask, input_trg_understanding,
                input_trg_understanding_mask) = self._build_understanding_layer(input_src_feat,
                    input_src_feat_mask, input_trg_feat, input_trg_feat_mask)
            
            (input_src_interaction, input_src_interaction_mask, input_trg_interaction,
                input_trg_interaction_mask) = self._build_interaction_layer(input_src_understanding,
                    input_src_understanding_mask, input_trg_understanding, input_trg_understanding_mask)
            
            input_matching, input_matching_mask = self._build_matching_layer(input_src_understanding,
                input_src_understanding_mask, input_trg_understanding, input_trg_understanding_mask, input_src_interaction,
                input_src_interaction_mask, input_trg_interaction, input_trg_interaction_mask)
            
            predict = input_matching
            predict_mask = input_matching_mask
            
        return predict, predict_mask
    
    def save(self,
             sess,
             global_step,
             save_mode):
        """save checkpoint"""
        if save_mode == "debug":
            self.ckpt_debug_saver.save(sess, self.ckpt_debug_name, global_step=global_step)
        elif save_mode == "epoch":
            self.ckpt_epoch_saver.save(sess, self.ckpt_epoch_name, global_step=global_step)
        else:
            raise ValueError("unsupported save mode {0}".format(save_mode))
    
    def restore(self,
                sess,
                ckpt_file,
                ckpt_type):
        """restore from checkpoint"""
        if ckpt_file is None:
            raise FileNotFoundError("checkpoint file doesn't exist")
        
        if ckpt_type == "debug":
            self.ckpt_debug_saver.restore(sess, ckpt_file)
        elif ckpt_type == "epoch":
            self.ckpt_epoch_saver.restore(sess, ckpt_file)
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
    
    def get_latest_ckpt(self,
                        ckpt_type):
        """get the latest checkpoint"""
        if ckpt_type == "debug":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_debug_dir)
            if ckpt_file is None:
                raise FileNotFoundError("latest checkpoint file doesn't exist")
            
            return ckpt_file
        elif ckpt_type == "epoch":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_epoch_dir)
            if ckpt_file is None:
                raise FileNotFoundError("latest checkpoint file doesn't exist")
            
            return ckpt_file
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
    
    def get_ckpt_list(self,
                      ckpt_type):
        """get checkpoint list"""
        if ckpt_type == "debug":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_debug_dir)
            if ckpt_state is None:
                raise FileNotFoundError("checkpoint files doesn't exist")
            
            return ckpt_state.all_model_checkpoint_paths
        elif ckpt_type == "epoch":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_epoch_dir)
            if ckpt_state is None:
                raise FileNotFoundError("checkpoint files doesn't exist")
            
            return ckpt_state.all_model_checkpoint_paths
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))

class ConvolutionBlock(object):
    """convolution-block layer"""
    def __init__(self,
                 num_conv,
                 unit_dim,
                 window_size,
                 activation,
                 dropout,
                 layer_dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="conv_block"):
        """initialize convolution-block layer"""
        self.num_conv = num_conv
        self.unit_dim = unit_dim
        self.window_size = window_size
        self.activation = activation
        self.dropout = dropout
        self.sublayer_index, self.num_sublayer, self.layer_dropout = layer_dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.position_layer = create_position_layer("sin_pos", 0, 0, 1, 10000,
                self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            conv_layer_dropout = [self.layer_dropout * float(i + self.sublayer_index) / self.num_sublayer for i in range(self.num_conv)]
            self.conv_layer = create_convolution_layer("multi_sep_1d", self.num_conv, self.unit_dim,
                self.unit_dim, 1, self.window_size, 1, "SAME", self.activation, [self.dropout] * self.num_conv, conv_layer_dropout,
                True, True, True, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            dense_layer_dropout = [self.layer_dropout * float(self.num_conv + self.sublayer_index) / self.num_sublayer]
            self.dense_layer = create_dense_layer("double", 1, self.unit_dim, 1, self.activation, [self.dropout],
                dense_layer_dropout, True, True, True, num_gpus, default_gpu_id, self.regularizer, self.random_seed, self.trainable)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call convolution-block layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_pos, input_pos_mask = self.position_layer(input_data, input_mask)
            input_conv, input_conv_mask = self.conv_layer(input_pos, input_pos_mask)
            output_block, output_block_mask = self.dense_layer(input_conv, input_conv_mask)
        
        return output_block, output_block_mask

class StackedConvolutionBlock(object):
    """stacked convolution-block layer"""
    def __init__(self,
                 num_layer,
                 num_conv,
                 unit_dim,
                 window_size,
                 activation,
                 dropout,
                 layer_dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="stacked_conv_block"):
        """initialize stacked convolution-block layer"""
        self.num_layer = num_layer
        self.num_conv = num_conv
        self.unit_dim = unit_dim
        self.window_size = window_size
        self.activation = activation
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.block_layer_list = []
            num_sublayer = (self.num_conv + 1) * self.num_layer
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_dropout = ((self.num_conv + 1) * i + 1, num_sublayer, self.layer_dropout)
                block_layer = ConvolutionBlock(num_conv=self.num_conv, unit_dim=self.unit_dim, window_size=self.window_size
                    activation=self.activation, dropout=self.dropout, layer_dropout=layer_dropout,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=self.trainable, scope=layer_scope)
                self.block_layer_list.append(block_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked convolution-block layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            output_block_list = []
            output_block_mask_list = []
            input_block = input_data
            input_block_mask = input_mask
            for block_layer in self.block_layer_list:
                output_block, output_block_mask = block_layer(input_block, input_block_mask)
                output_block_list.append(output_block)
                output_block_mask_list.append(output_block_mask)
                input_block = output_block
                input_block_mask = output_block_mask
        
        return output_block_list, output_block_mask_list
