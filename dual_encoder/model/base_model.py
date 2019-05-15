import collections
import os.path

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.dual_encoder_util import *
from util.layer_util import *

__all__ = ["TrainResult", "InferResult", "BaseModel", "FusionModule",
           "CosineScore", "DenseScore", "WordFeat", "CharFeat"]

class TrainResult(collections.namedtuple("TrainResult",
    ("loss", "learning_rate", "global_step", "batch_size", "summary"))):
    pass

class InferResult(collections.namedtuple("InferResult",
    ("predict", "batch_size", "summary"))):
    pass

class BaseModel(object):
    """dual encoder base model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 external_data,
                 mode="train",
                 scope="base"):
        """initialize dual encoder base model"""
        self.logger = logger
        self.hyperparams = hyperparams
        self.data_pipeline = data_pipeline
        self.mode = mode
        self.scope = scope
        
        self.update_op = None
        self.train_loss = None
        self.learning_rate = None
        self.global_step = None
        self.train_summary = None
        self.infer_summary = None
        
        self.src_word_embed = external_data["src_word_embed"] if external_data is not None and "src_word_embed" in external_data else None
        self.trg_word_embed = external_data["trg_word_embed"] if external_data is not None and "trg_word_embed" in external_data else None
        self.src_word_embed_placeholder = None
        self.trg_word_embed_placeholder = None
        
        self.batch_size = tf.size(tf.reduce_max(self.data_pipeline.input_label_mask, axis=-2))
        self.max_batch_size = self.data_pipeline.batch_size
        self.neg_num = self.hyperparams.train_neg_num
        self.enable_negative_sampling = self.hyperparams.train_loss_type == "neg_sampling" and self.mode == "train"
        
        self.num_gpus = self.hyperparams.device_num_gpus
        self.default_gpu_id = self.hyperparams.device_default_gpu_id
        self.logger.log_print("# {0} gpus are used with default gpu id set as {1}"
            .format(self.num_gpus, self.default_gpu_id))
        
        if self.hyperparams.train_regularization_enable == True:
            self.regularizer = create_weight_regularizer(self.hyperparams.train_regularization_type,
                self.hyperparams.train_regularization_scale)
        else:
            self.regularizer = None
        
        self.random_seed = self.hyperparams.train_random_seed if self.hyperparams.train_enable_debugging else None
    
    def _get_exponential_moving_average(self,
                                        num_steps):
        decay_rate = self.hyperparams.train_ema_decay_rate
        enable_debias = self.hyperparams.train_ema_enable_debias
        enable_dynamic_decay = self.hyperparams.train_ema_enable_dynamic_decay
        
        if enable_dynamic_decay == True:
            ema = tf.train.ExponentialMovingAverage(decay=decay_rate, num_updates=num_steps, zero_debias=enable_debias)
        else:
            ema = tf.train.ExponentialMovingAverage(decay=decay_rate, zero_debias=enable_debias)
        
        return ema
    
    def _apply_learning_rate_warmup(self,
                                    learning_rate):
        """apply learning rate warmup"""
        warmup_mode = self.hyperparams.train_optimizer_warmup_mode
        warmup_rate = self.hyperparams.train_optimizer_warmup_rate
        warmup_end_step = self.hyperparams.train_optimizer_warmup_end_step
        
        if warmup_mode == "exponential_warmup":
            warmup_factor = warmup_rate ** (1 - tf.to_float(self.global_step) / tf.to_float(warmup_end_step))
            warmup_learning_rate = warmup_factor * learning_rate
        elif warmup_mode == "inverse_exponential_warmup":
            warmup_factor = tf.log(tf.to_float(self.global_step + 1)) / tf.log(tf.to_float(warmup_end_step))
            warmup_learning_rate = warmup_factor * learning_rate
        else:
            raise ValueError("unsupported warm-up mode {0}".format(warmup_mode))
        
        warmup_learning_rate = tf.cond(tf.less(self.global_step, warmup_end_step),
            lambda: warmup_learning_rate, lambda: learning_rate)
        
        return warmup_learning_rate
    
    def _apply_learning_rate_decay(self,
                                   learning_rate):
        """apply learning rate decay"""
        decay_mode = self.hyperparams.train_optimizer_decay_mode
        decay_rate = self.hyperparams.train_optimizer_decay_rate
        decay_step = self.hyperparams.train_optimizer_decay_step
        decay_start_step = self.hyperparams.train_optimizer_decay_start_step
        
        if decay_mode == "exponential_decay":
            decayed_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step),
                decay_steps=decay_step, decay_rate=decay_rate, staircase=True)
        elif decay_mode == "inverse_time_decay":
            decayed_learning_rate = tf.train.inverse_time_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step),
                decay_steps=decay_step, decay_rate=decay_rate, staircase=True)
        else:
            raise ValueError("unsupported decay mode {0}".format(decay_mode))
        
        decayed_learning_rate = tf.cond(tf.less(self.global_step, decay_start_step),
            lambda: learning_rate, lambda: decayed_learning_rate)
        
        return decayed_learning_rate
    
    def _initialize_optimizer(self,
                              learning_rate):
        """initialize optimizer"""
        optimizer_type = self.hyperparams.train_optimizer_type
        if optimizer_type == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                momentum=self.hyperparams.train_optimizer_momentum_beta)
        elif optimizer_type == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                decay=self.hyperparams.train_optimizer_rmsprop_beta,
                epsilon=self.hyperparams.train_optimizer_rmsprop_epsilon)
        elif optimizer_type == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                rho=self.hyperparams.train_optimizer_adadelta_rho,
                epsilon=self.hyperparams.train_optimizer_adadelta_epsilon)
        elif optimizer_type == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                initial_accumulator_value=self.hyperparams.train_optimizer_adagrad_init_accumulator)
        elif optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                beta1=self.hyperparams.train_optimizer_adam_beta_1, beta2=self.hyperparams.train_optimizer_adam_beta_2,
                epsilon=self.hyperparams.train_optimizer_adam_epsilon)
        else:
            raise ValueError("unsupported optimizer type {0}".format(optimizer_type))
        
        return optimizer
    
    def _minimize_loss(self,
                       loss):
        """minimize optimization loss"""
        """compute gradients"""
        if self.num_gpus > 1:
            grads_and_vars = self.optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
        else:
            grads_and_vars = self.optimizer.compute_gradients(loss, colocate_gradients_with_ops=False)
        
        """clip gradients"""
        gradients = [x[0] for x in grads_and_vars]
        variables = [x[1] for x in grads_and_vars]
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.hyperparams.train_clip_norm)
        grads_and_vars = zip(clipped_gradients, variables)
        
        """update model based on gradients"""
        update_model = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        
        return update_model, clipped_gradients, gradient_norm
    
    def _compute_loss(self,
                      label,
                      label_mask,
                      predict,
                      predict_mask):
        """compute optimization loss"""
        loss_type = self.hyperparams.train_loss_type
        
        if loss_type == "neg_sampling":
            masked_label = label * label_mask
            masked_predict = predict * predict_mask
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=masked_predict, labels=masked_label)
            cross_entropy_mask = tf.reduce_max(tf.concat([label_mask, predict_mask], axis=-1), axis=-1, keepdims=True)
            masked_cross_entropy = cross_entropy * cross_entropy_mask
            loss = tf.reduce_mean(tf.reduce_sum(masked_cross_entropy, axis=-2))
        else:
            raise ValueError("unsupported loss type {0}".format(loss_type))
        
        return loss
    
    def _neg_sampling(self,
                      input_src_data,
                      input_src_mask,
                      input_trg_data,
                      input_trg_mask,
                      batch_size,
                      neg_num,
                      random_seed=0,
                      indice_list=None):
        """negative sampling"""
        if indice_list is None:
            indice_list = self._neg_sampling_indice(batch_size, neg_num, random_seed)
        
        input_src_shape = tf.shape(input_src_data)
        src_max_length = input_src_shape[1]
        input_src_sample = tf.reshape(input_src_data, shape=[batch_size, 1, src_max_length, -1])
        input_src_sample = tf.tile(input_src_sample, multiples=[1, neg_num+1, 1, 1])
        input_src_sample_mask = tf.reshape(input_src_mask, shape=[batch_size, 1, src_max_length, -1])
        input_src_sample_mask = tf.tile(input_src_sample_mask, multiples=[1, neg_num+1, 1, 1])
        
        input_trg_sample_list = []
        input_trg_sample_mask_list = []
        input_trg_shape = tf.shape(input_trg_data)
        trg_max_length = input_trg_shape[1]
        for indice in indice_list:
            input_trg_sample = tf.gather(input_trg_data, indice, axis=0)
            input_trg_sample = tf.reshape(input_trg_sample, shape=[1, neg_num+1, trg_max_length, -1])
            input_trg_sample_list.append(input_trg_sample)
            input_trg_sample_mask = tf.gather(input_trg_mask, indice, axis=0)
            input_trg_sample_mask = tf.reshape(input_trg_sample_mask, shape=[1, neg_num+1, trg_max_length, -1])
            input_trg_sample_mask_list.append(input_trg_sample_mask)
        
        input_trg_sample = tf.concat(input_trg_sample_list, axis=0)
        input_trg_sample_mask = tf.concat(input_trg_sample_mask_list, axis=0)
        
        return input_src_sample, input_src_sample_mask, input_trg_sample, input_trg_sample_mask
    
    def _neg_sampling_label(self,
                            batch_size,
                            neg_num):
        """generate label for negative sampling"""
        label_list = []
        for index in range(batch_size):
            label = [1] + [0] * neg_num
            label_list.append(label)
        
        label = tf.reshape(tf.convert_to_tensor(label_list, dtype=tf.float32), shape=[batch_size, neg_num+1])
        label_mask = label
        
        return label, label_mask
    
    def _neg_sampling_indice(self,
                             batch_size,
                             neg_num,
                             random_seed):
        """generate indice for negative sampling"""
        np.random.seed(random_seed)
        indice_list = []
        for index in range(batch_size):
            neg_num = min(batch_size-1, neg_num) 
            indice = list(range(batch_size))
            indice.remove(index)
            np.random.shuffle(indice)
            indice = [index] + indice[:neg_num]
            indice_list.append(indice)
        
        return indice_list
    
    def train(self,
              sess,
              src_word_embed=None,
              trg_word_embed=None):
        """train model"""
        feed_word_embed = (self.hyperparams.model_representation_src_word_embed_pretrained and
            self.hyperparams.model_representation_trg_word_embed_pretrained and
            src_word_embed is not None and self.src_word_embed_placeholder is not None and
            trg_word_embed is not None and self.trg_word_embed_placeholder is not None)
        
        if feed_word_embed == True:
            (_, loss, learning_rate, global_step, batch_size, summary) = sess.run([self.update_op,
                self.train_loss, self.learning_rate, self.global_step, self.batch_size, self.train_summary],
                feed_dict={self.src_word_embed_placeholder: src_word_embed, self.trg_word_embed_placeholder: trg_word_embed})
        else:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_op,
                self.train_loss, self.learning_rate, self.global_step, self.batch_size, self.train_summary])
                
        return TrainResult(loss=loss, learning_rate=learning_rate,
            global_step=global_step, batch_size=batch_size, summary=summary)
    
    def infer(self,
              sess,
              src_word_embed=None,
              trg_word_embed=None):
        """infer model"""
        feed_word_embed = (self.hyperparams.model_representation_src_word_embed_pretrained and
            self.hyperparams.model_representation_trg_word_embed_pretrained and
            src_word_embed is not None and self.src_word_embed_placeholder is not None and
            trg_word_embed is not None and self.trg_word_embed_placeholder is not None)
        
        if feed_word_embed == True:
            (infer_predict, batch_size,
                summary) = sess.run([self.infer_predict, self.batch_size, self.infer_summary],
                    feed_dict={self.src_word_embed_placeholder: src_word_embed, self.trg_word_embed_placeholder: trg_word_embed})
        else:
            (infer_predict, batch_size,
                summary) = sess.run([self.infer_predict, self.batch_size, self.infer_summary])
        
        return InferResult(predict=infer_predict, batch_size=batch_size, summary=summary)
        
    def _get_train_summary(self):
        """get train summary"""
        return tf.summary.merge([tf.summary.scalar("learning_rate", self.learning_rate),
            tf.summary.scalar("train_loss", self.train_loss), tf.summary.scalar("gradient_norm", self.gradient_norm)])
    
    def _get_infer_summary(self):
        """get infer summary"""
        return tf.no_op()

class FusionModule(object):
    """fusion-module layer"""
    def __init__(self,
                 input_unit_dim,
                 output_unit_dim,
                 fusion_type,
                 num_layer,
                 activation,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="fusion"):
        """initialize fusion-module layer"""
        self.input_unit_dim = input_unit_dim
        self.output_unit_dim = output_unit_dim
        self.fusion_type = fusion_type
        self.num_layer= num_layer
        self.activation = activation
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if self.fusion_type == "concate":
                self.fusion_layer_list = []
                if self.input_unit_dim != self.output_unit_dim:
                    convert_layer = create_convolution_layer("stacked_1d", 1, self.input_unit_dim, self.output_unit_dim,
                        1, 1, "SAME", None, [self.dropout] * self.num_layer, None, False, False, False, self.num_gpus,
                        self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
                    self.fusion_layer_list.append(convert_layer)
            elif self.fusion_type == "dense":
                fusion_layer = create_dense_layer("single", self.num_layer, self.output_unit_dim, 1,
                    self.activation, [self.dropout] * self.num_layer, None, False, False, False, self.num_gpus,
                    self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
                self.fusion_layer_list = [fusion_layer]
            elif self.fusion_type == "highway":
                self.fusion_layer_list = []
                if self.input_unit_dim != self.output_unit_dim:
                    convert_layer = create_convolution_layer("stacked_1d", 1, self.input_unit_dim, self.output_unit_dim,
                        1, 1, "SAME", None, [self.dropout] * self.num_layer, None, False, False, False, self.num_gpus,
                        self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
                    self.fusion_layer_list.append(convert_layer)
                
                fusion_layer = create_highway_layer("highway", self.num_layer, self.output_unit_dim, None,
                    self.activation, [self.dropout] * self.num_layer, self.num_gpus, self.default_gpu_id,
                    self.regularizer, self.random_seed, self.trainable)
                self.fusion_layer_list.append(fusion_layer)
            elif self.fusion_type == "conv":
                fusion_layer = create_convolution_layer("stacked_1d", self.num_layer, self.input_unit_dim, self.output_unit_dim,
                    1, 1, "SAME", self.activation, [self.dropout] * self.num_layer, None, False, False, False, self.num_gpus,
                    self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
                self.fusion_layer_list = [fusion_layer]
            else:
                raise ValueError("unsupported fusion type {0}".format(self.fusion_type))
    
    def __call__(self,
                 input_data_list,
                 input_mask_list):
        """call fusion-module layer"""
        input_fusion = tf.concat(input_data_list, axis=-1)
        input_fusion_mask = tf.reduce_max(tf.concat(input_mask_list, axis=-1), axis=-1, keepdims=True)
        
        if input_fusion.get_shape().as_list()[-1] is None:
            input_fusion_shape = tf.shape(input_fusion)
            input_fusion = tf.reshape(input_fusion, shape=tf.concat([input_fusion_shape[:-1], [self.input_unit_dim]], axis=0))
                
        for fusion_layer in self.fusion_layer_list:
            input_fusion, input_fusion_mask = fusion_layer(input_fusion, input_fusion_mask)
        
        return input_fusion, input_fusion_mask

class CosineScore(object):
    """cosine score layer"""
    def __init__(self,
                 pooling_type,
                 num_gpus=1,
                 default_gpu_id=0,
                 scope="cosine_score"):
        """initialize cosine-score layer"""
        self.pooling_type = pooling_type
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.pooling_layer = create_pooling_layer(self.pooling_type, 1, 1, self.num_gpus, self.default_gpu_id)
    
    def __call__(self,
                 input_src_data,
                 input_src_mask,
                 input_trg_data,
                 input_trg_mask):
        """call cosine-score layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_src_pool, input_src_pool_mask = self.pooling_layer(input_src_data, input_src_mask)
            input_trg_pool, input_trg_pool_mask = self.pooling_layer(input_trg_data, input_trg_mask)
            
            input_src_norm = tf.expand_dims(tf.nn.l2_normalize(input_src_pool, axis=-1), axis=-2)
            input_src_norm_mask = tf.expand_dims(input_src_pool_mask, axis=-2)
            input_trg_norm = tf.expand_dims(tf.nn.l2_normalize(input_trg_pool, axis=-1), axis=-1)
            input_trg_norm_mask = tf.expand_dims(input_trg_pool_mask, axis=-1)
            
            output_matching = tf.squeeze(tf.matmul(input_src_norm, input_trg_norm), axis=[-2,-1])
            output_matching_mask = tf.squeeze(tf.matmul(input_src_norm_mask, input_trg_norm_mask), axis=[-2,-1])
        
        return output_matching, output_matching_mask

class DenseScore(object):
    """dense-score layer"""
    def __init__(self,
                 pooling_type,
                 num_layer,
                 unit_dim,
                 activation,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="dense_score"):
        """initialize dense-score layer"""
        self.pooling_type = pooling_type
        self.num_layer = num_layer
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.pooling_layer = create_pooling_layer(self.pooling_type, 1, 1, self.num_gpus, self.default_gpu_id)
            
            self.dense_layer = create_dense_layer("double", self.num_layer, self.unit_dim,
                1, self.activation, [self.dropout] * self.num_layer, None, True, True, True,
                self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            self.project_layer = create_dense_layer("single", 1, 1, 1, None, [0.0], None, False, False, False,
                self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
    
    def __call__(self,
                 input_src_data,
                 input_src_mask,
                 input_trg_data,
                 input_trg_mask):
        """call dense-score layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_src_pool, input_src_pool_mask = self.pooling_layer(input_src_data, input_src_mask)
            input_trg_pool, input_trg_pool_mask = self.pooling_layer(input_trg_data, input_trg_mask)
            
            input_norm = tf.nn.l2_normalize(tf.concat([input_src_pool, input_trg_pool], axis=-1), axis=-1)
            input_norm_mask = tf.reduce_max(tf.concat([input_src_pool_mask, input_trg_pool_mask], axis=-1), axis=-1, keepdims=True)
            
            input_dense, input_dense_mask = self.dense_layer(input_norm, input_norm_mask)
            input_project, input_project_mask = self.project_layer(input_dense, input_dense_mask)
            
            output_matching = tf.squeeze(input_project, axis=-1)
            output_matching_mask = tf.squeeze(input_project_mask, axis=-1)
        
        return output_matching, output_matching_mask

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
                self.num_gpus, self.default_gpu_id, None, self.random_seed, False, self.trainable)
            
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
                self.num_gpus, self.default_gpu_id, None, self.random_seed, False, self.trainable)
            
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
