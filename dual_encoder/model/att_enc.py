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

__all__ = ["AttentionEncoder"]

class AttentionEncoder(BaseModel):
    """attention encoder model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 external_data,
                 mode="train",
                 scope="att_enc"):
        """initialize attention encoder model"""
        super(AttentionEncoder, self).__init__(logger=logger, hyperparams=hyperparams,
            data_pipeline=data_pipeline, external_data=external_data, mode=mode, scope=scope)
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                initializer=tf.zeros_initializer, trainable=False)
            
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
        """build representation layer for attention encoder model"""
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
        """build understanding layer for sequence encoder model"""
        src_representation_unit_dim = self.hyperparams.model_representation_src_fusion_unit_dim
        src_understanding_num_layer = self.hyperparams.model_understanding_src_num_layer
        src_understanding_num_head = self.hyperparams.model_understanding_src_num_head
        src_understanding_unit_dim = self.hyperparams.model_understanding_src_unit_dim
        src_understanding_hidden_activation = self.hyperparams.model_understanding_src_hidden_activation
        src_understanding_dropout = self.hyperparams.model_understanding_src_dropout if self.mode == "train" else 0.0
        src_understanding_att_dropout = self.hyperparams.model_understanding_src_attention_dropout if self.mode == "train" else 0.0
        src_understanding_layer_dropout = self.hyperparams.model_understanding_src_layer_dropout if self.mode == "train" else 0.0
        src_understanding_trainable = self.hyperparams.model_understanding_src_trainable
        trg_representation_unit_dim = self.hyperparams.model_representation_trg_fusion_unit_dim
        trg_understanding_num_layer = self.hyperparams.model_understanding_trg_num_layer
        trg_understanding_num_head = self.hyperparams.model_understanding_trg_num_head
        trg_understanding_unit_dim = self.hyperparams.model_understanding_trg_unit_dim
        trg_understanding_hidden_activation = self.hyperparams.model_understanding_trg_hidden_activation
        trg_understanding_dropout = self.hyperparams.model_understanding_trg_dropout if self.mode == "train" else 0.0
        trg_understanding_att_dropout = self.hyperparams.model_understanding_trg_attention_dropout if self.mode == "train" else 0.0
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
                src_understanding_layer = StackedAttentionBlock(num_layer=src_understanding_num_layer,
                    num_head=src_understanding_num_head, unit_dim=src_understanding_unit_dim,
                    activation=src_understanding_hidden_activation, dropout=src_understanding_dropout,
                    att_dropout=src_understanding_att_dropout, layer_dropout=src_understanding_layer_dropout,
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
                    trg_understanding_layer = StackedAttentionBlock(num_layer=trg_understanding_num_layer,
                        num_head=trg_understanding_num_head, unit_dim=trg_understanding_unit_dim,
                        activation=trg_understanding_hidden_activation, dropout=trg_understanding_dropout,
                        att_dropout=trg_understanding_att_dropout, layer_dropout=trg_understanding_layer_dropout,
                        num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                        random_seed=self.random_seed, trainable=trg_understanding_trainable)
                
                (input_trg_fusion,
                    input_trg_fusion_mask) = trg_fusion_layer([input_trg_feat], [input_trg_feat_mask])
                (input_trg_understanding_list,
                    input_trg_understanding_mask_list) = trg_understanding_layer(input_trg_fusion, input_trg_fusion_mask)
                input_trg_understanding = input_trg_understanding_list[-1]
                input_trg_understanding_mask = input_trg_understanding_mask_list[-1]
        
        return input_src_understanding, input_src_understanding_mask, input_trg_understanding, input_trg_understanding_mask
    
    def _build_interaction_layer(self,
                                 input_src_understanding,
                                 input_src_understanding_mask,
                                 input_trg_understanding,
                                 input_trg_understanding_mask):
        """build interaction layer for sequence encoder model"""
        pass
    
    def _build_matching_layer(self,
                              input_src_understanding,
                              input_src_understanding_mask,
                              input_trg_understanding,
                              input_trg_understanding_mask,
                              input_src2trg_interaction,
                              input_src2trg_interaction_mask,
                              input_trg2src_interaction,
                              input_trg2src_interaction_mask):
        """build matching layer for sequence encoder model"""
        pass
    
    def _build_graph(self,
                     input_src_word,
                     input_src_word_mask,
                     input_src_char,
                     input_src_char_mask,
                     input_trg_word,
                     input_trg_word_mask,
                     input_trg_char,
                     input_trg_char_mask):
        """build graph for sequence encoder model"""
        pass
    
    def _compute_loss(self,
                      label,
                      label_mask,
                      predict,
                      predict_mask):
        """compute optimization loss"""
        pass
    
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

class AttentionBlock(object):
    """attention-block layer"""
    def __init__(self,
                 num_head,
                 unit_dim,
                 activation,
                 dropout,
                 att_dropout,
                 layer_dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="att_block"):
        """initialize attention-block layer"""
        self.num_head = num_head
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.att_dropout = att_dropout
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
            
            att_layer_dropout = self.layer_dropout * float(self.sublayer_index) / self.num_sublayer
            self.attention_layer = create_attention_layer("multi_head_att", self.unit_dim, self.unit_dim,
                self.unit_dim, self.num_head, "scaled_dot", self.dropout, self.att_dropout, att_layer_dropout,
                True, True, True, None, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            dense_layer_dropout = [self.layer_dropout * float(self.sublayer_index + 1) / self.num_sublayer]
            self.dense_layer = create_dense_layer("double", 1, self.unit_dim, 1, self.activation, [self.dropout],
                dense_layer_dropout, True, True, True, num_gpus, default_gpu_id, self.regularizer, self.random_seed, self.trainable)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call attention-block layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_pos, input_pos_mask = self.position_layer(input_data, input_mask)
            input_att, input_att_mask = self.attention_layer(input_pos, input_pos, input_pos_mask, input_pos_mask)
            output_block, output_block_mask = self.dense_layer(input_att, input_att_mask)
        
        return output_block, output_block_mask

class StackedAttentionBlock(object):
    """stacked attention-block layer"""
    def __init__(self,
                 num_layer,
                 num_head,
                 unit_dim,
                 activation,
                 dropout,
                 att_dropout,
                 layer_dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="stacked_att_block"):
        """initialize stacked attention-block layer"""
        self.num_layer = num_layer
        self.num_head = num_head
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.layer_dropout = layer_dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.block_layer_list = []
            num_sublayer = 2 * self.num_layer
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_dropout = (2 * i + 1, num_sublayer, self.layer_dropout)
                block_layer = AttentionBlock(num_head=self.num_head, unit_dim=self.unit_dim, activation=self.activation,
                    dropout=self.dropout, att_dropout=self.att_dropout, layer_dropout=layer_dropout,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=self.trainable, scope=layer_scope)
                self.block_layer_list.append(block_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked attention-block layer"""
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

class WordFeat(object):
    """word-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 dropout,
                 pretrained,
                 embedding=None,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="word_feat"):
        """initialize word-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.pretrained = pretrained
        self.embedding = embedding
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size, self.embed_dim, self.pretrained, self.embedding,
                self.num_gpus, self.default_gpu_id, None, self.random_seed, self.trainable)
            
            self.dropout_layer = create_dropout_layer(self.dropout, self.num_gpus, self.default_gpu_id, self.random_seed)
    
    def __call__(self,
                 input_word,
                 input_word_mask):
        """call word-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_word_embedding_mask = input_word_mask
            input_word_embedding = tf.squeeze(self.embedding_layer(input_word), axis=-2)
            
            (input_word_dropout,
                input_word_dropout_mask) = self.dropout_layer(input_word_embedding, input_word_embedding_mask)
            
            input_word_feat = input_word_dropout
            input_word_feat_mask = input_word_dropout_mask
        
        return input_word_feat, input_word_feat_mask
    
    def get_embedding_placeholder(self):
        """get word-level embedding placeholder"""
        return self.embedding_layer.get_embedding_placeholder()

class CharFeat(object):
    """char-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 unit_dim,
                 window_size,
                 activation,
                 pooling_type,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="char_feat"):
        """initialize char-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.unit_dim = unit_dim
        self.window_size = window_size
        self.activation = activation
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size, self.embed_dim, False, None,
                self.num_gpus, self.default_gpu_id, None, self.random_seed, self.trainable)
            
            self.conv_layer = create_convolution_layer("stacked_multi_1d", 1, self.embed_dim, self.unit_dim,
                self.window_size, 1, "SAME", self.activation, [0.0], None, False, False, True,
                self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            self.dropout_layer = create_dropout_layer(self.dropout, self.num_gpus, self.default_gpu_id, self.random_seed)
            
            self.pooling_layer = create_pooling_layer(self.pooling_type, 1, 1, self.num_gpus, self.default_gpu_id)
    
    def __call__(self,
                 input_char,
                 input_char_mask):
        """call char-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_char_embedding_mask = tf.expand_dims(input_char_mask, axis=-1)
            input_char_embedding = self.embedding_layer(input_char)
            
            (input_char_dropout,
                input_char_dropout_mask) = self.dropout_layer(input_char_embedding, input_char_embedding_mask)
            
            (input_char_conv,
                input_char_conv_mask) = self.conv_layer(input_char_dropout, input_char_dropout_mask)
            
            (input_char_pool,
                input_char_pool_mask) = self.pooling_layer(input_char_conv, input_char_conv_mask)
            
            input_char_feat = input_char_pool
            input_char_feat_mask = input_char_pool_mask
        
        return input_char_feat, input_char_feat_mask
