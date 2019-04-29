import collections

import numpy as np
import tensorflow as tf

from model.conv_enc import *
from model.seq_enc import *
from model.att_enc import *
from util.data_util import *

__all__ = ["TrainModel", "InferModel",
           "create_train_model", "create_infer_model",
           "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel",
    ("graph", "model", "data_pipeline", "word_embedding", "input_data", "input_src_data", "input_trg_data", "input_label_data"))):
    pass

class InferModel(collections.namedtuple("InferModel",
    ("graph", "model", "data_pipeline", "word_embedding", "input_data", "input_src_data", "input_trg_data", "input_label_data"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_dual_data, input_src_data, input_trg_data, input_label_data, src_word_embed_data,
            src_word_vocab_size, src_word_vocab_index, src_word_vocab_inverted_index, src_char_vocab_size,
            src_char_vocab_index, src_char_vocab_inverted_index, trg_word_embed_data, trg_word_vocab_size,
            trg_word_vocab_index, trg_word_vocab_inverted_index, trg_char_vocab_size, trg_char_vocab_index,
            trg_char_vocab_inverted_index) = prepare_dual_data(logger, hyperparams.data_train_dual_file,
                hyperparams.data_train_dual_file_type, hyperparams.data_src_word_vocab_file,
                hyperparams.data_src_word_vocab_size, hyperparams.data_src_word_vocab_threshold,
                hyperparams.model_representation_src_word_embed_dim, hyperparams.data_src_embed_file,
                hyperparams.data_src_embed_full_file, hyperparams.data_src_word_unk, hyperparams.data_src_word_pad,
                hyperparams.model_representation_src_word_feat_enable, hyperparams.model_representation_src_word_embed_pretrained,
                hyperparams.data_src_char_vocab_file, hyperparams.data_src_char_vocab_size, hyperparams.data_src_char_vocab_threshold,
                hyperparams.data_src_char_unk, hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable,
                hyperparams.data_trg_word_vocab_file, hyperparams.data_trg_word_vocab_size, hyperparams.data_trg_word_vocab_threshold,
                hyperparams.model_representation_trg_word_embed_dim, hyperparams.data_trg_embed_file,
                hyperparams.data_trg_embed_full_file, hyperparams.data_trg_word_unk, hyperparams.data_trg_word_pad,
                hyperparams.model_representation_trg_word_feat_enable, hyperparams.model_representation_trg_word_embed_pretrained,
                hyperparams.data_trg_char_vocab_file, hyperparams.data_trg_char_vocab_size, hyperparams.data_trg_char_vocab_threshold,
                hyperparams.data_trg_char_unk, hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable)
        
        external_data = {}
        if hyperparams.data_pipeline_mode == "dynamic":
            logger.log_print("# create train source dataset")
            input_src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_src_dataset = tf.data.Dataset.from_tensor_slices(input_src_placeholder)
            input_src_word_dataset, input_src_char_dataset = create_text_dataset(input_src_dataset,
                src_word_vocab_index, hyperparams.data_src_word_max_length, hyperparams.data_src_word_pad,
                hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index, hyperparams.data_src_char_max_length,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, hyperparams.data_num_parallel)
            
            logger.log_print("# create train target dataset")
            input_trg_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_trg_dataset = tf.data.Dataset.from_tensor_slices(input_trg_placeholder)
            input_trg_word_dataset, input_trg_char_dataset = create_text_dataset(input_trg_dataset,
                trg_word_vocab_index, hyperparams.data_trg_word_max_length, hyperparams.data_trg_word_pad,
                hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index, hyperparams.data_trg_char_max_length,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create train label dataset")
            input_label_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_label_dataset = tf.data.Dataset.from_tensor_slices(input_label_placeholder)
            input_label_dataset = create_label_dataset(input_label_dataset, 1, hyperparams.data_num_parallel)

            logger.log_print("# create train data pipeline")
            data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            data_pipeline = create_dynamic_pipeline(input_src_word_dataset, input_src_char_dataset,
                input_trg_word_dataset, input_trg_char_dataset, input_label_dataset, src_word_vocab_index,
                hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, trg_word_vocab_index,
                hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable,
                hyperparams.train_random_seed, hyperparams.train_enable_shuffle, hyperparams.train_shuffle_buffer_size,
                input_src_placeholder, input_trg_placeholder, input_label_placeholder, data_size_placeholder, batch_size_placeholder)
        else:
            if word_embed_data is not None:
                external_data["word_embedding"] = word_embed_data
            
            logger.log_print("# create train source dataset")
            input_src_dataset = tf.data.Dataset.from_tensor_slices(input_src_data)
            input_src_word_dataset, input_src_char_dataset = create_text_dataset(input_src_dataset,
                src_word_vocab_index, hyperparams.data_src_word_max_length, hyperparams.data_src_word_pad,
                hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index, hyperparams.data_src_char_max_length,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, hyperparams.data_num_parallel)
            
            logger.log_print("# create train target dataset")
            input_trg_dataset = tf.data.Dataset.from_tensor_slices(input_trg_data)
            input_trg_word_dataset, input_trg_char_dataset = create_text_dataset(input_trg_dataset,
                trg_word_vocab_index, hyperparams.data_trg_word_max_length, hyperparams.data_trg_word_pad,
                hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index, hyperparams.data_trg_char_max_length,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable, hyperparams.data_num_parallel)
            
            logger.log_print("# create train label dataset")
            input_label_dataset = tf.data.Dataset.from_tensor_slices(input_label_data)
            input_label_dataset = create_label_dataset(input_label_dataset, 1, hyperparams.data_num_parallel)
            
            logger.log_print("# create train data pipeline")
            data_pipeline = create_data_pipeline(input_src_word_dataset, input_src_char_dataset,
                input_trg_word_dataset, input_trg_char_dataset, input_label_dataset, src_word_vocab_index,
                hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, trg_word_vocab_index,
                hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable,
                hyperparams.train_random_seed, hyperparams.train_enable_shuffle, hyperparams.train_shuffle_buffer_size,
                len(input_data), hyperparams.train_batch_size)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            external_data=external_data, mode="train", scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline,
            word_embedding=word_embed_data, input_data=input_data, input_src_data=input_src_data,
            input_trg_data=input_trg_data, input_label=input_label_data)

def create_infer_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare infer data")
        (input_dual_data, input_src_data, input_trg_data, input_label_data, src_word_embed_data,
            src_word_vocab_size, src_word_vocab_index, src_word_vocab_inverted_index, src_char_vocab_size,
            src_char_vocab_index, src_char_vocab_inverted_index, trg_word_embed_data, trg_word_vocab_size,
            trg_word_vocab_index, trg_word_vocab_inverted_index, trg_char_vocab_size, trg_char_vocab_index,
            trg_char_vocab_inverted_index) = prepare_dual_data(logger, hyperparams.data_eval_dual_file,
                hyperparams.data_eval_dual_file_type, hyperparams.data_src_word_vocab_file,
                hyperparams.data_src_word_vocab_size, hyperparams.data_src_word_vocab_threshold,
                hyperparams.model_representation_src_word_embed_dim, hyperparams.data_src_embed_file,
                hyperparams.data_src_embed_full_file, hyperparams.data_src_word_unk, hyperparams.data_src_word_pad,
                hyperparams.model_representation_src_word_feat_enable, hyperparams.model_representation_src_word_embed_pretrained,
                hyperparams.data_src_char_vocab_file, hyperparams.data_src_char_vocab_size, hyperparams.data_src_char_vocab_threshold,
                hyperparams.data_src_char_unk, hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable,
                hyperparams.data_trg_word_vocab_file, hyperparams.data_trg_word_vocab_size, hyperparams.data_trg_word_vocab_threshold,
                hyperparams.model_representation_trg_word_embed_dim, hyperparams.data_trg_embed_file,
                hyperparams.data_trg_embed_full_file, hyperparams.data_trg_word_unk, hyperparams.data_trg_word_pad,
                hyperparams.model_representation_trg_word_feat_enable, hyperparams.model_representation_trg_word_embed_pretrained,
                hyperparams.data_trg_char_vocab_file, hyperparams.data_trg_char_vocab_size, hyperparams.data_trg_char_vocab_threshold,
                hyperparams.data_trg_char_unk, hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable)
        
        external_data = {}
        if hyperparams.data_pipeline_mode == "dynamic":
            logger.log_print("# create infer source dataset")
            input_src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_src_dataset = tf.data.Dataset.from_tensor_slices(input_src_placeholder)
            input_src_word_dataset, input_src_char_dataset = create_text_dataset(input_src_dataset,
                src_word_vocab_index, hyperparams.data_src_word_max_length, hyperparams.data_src_word_pad,
                hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index, hyperparams.data_src_char_max_length,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, hyperparams.data_num_parallel)
            
            logger.log_print("# create infer target dataset")
            input_trg_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_trg_dataset = tf.data.Dataset.from_tensor_slices(input_trg_placeholder)
            input_trg_word_dataset, input_trg_char_dataset = create_text_dataset(input_trg_dataset,
                trg_word_vocab_index, hyperparams.data_trg_word_max_length, hyperparams.data_trg_word_pad,
                hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index, hyperparams.data_trg_char_max_length,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create infer label dataset")
            input_label_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            input_label_dataset = tf.data.Dataset.from_tensor_slices(input_label_placeholder)
            input_label_dataset = create_label_dataset(input_label_dataset, 1, hyperparams.data_num_parallel)

            logger.log_print("# create infer data pipeline")
            data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            data_pipeline = create_dynamic_pipeline(input_src_word_dataset, input_src_char_dataset,
                input_trg_word_dataset, input_trg_char_dataset, input_label_dataset, src_word_vocab_index,
                hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, trg_word_vocab_index,
                hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable, None, False, 0,
                input_src_placeholder, input_trg_placeholder, input_label_placeholder, data_size_placeholder, batch_size_placeholder)
        else:
            if word_embed_data is not None:
                external_data["word_embedding"] = word_embed_data
            
            logger.log_print("# create infer source dataset")
            input_src_dataset = tf.data.Dataset.from_tensor_slices(input_src_data)
            input_src_word_dataset, input_src_char_dataset = create_text_dataset(input_src_dataset,
                src_word_vocab_index, hyperparams.data_src_word_max_length, hyperparams.data_src_word_pad,
                hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index, hyperparams.data_src_char_max_length,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, hyperparams.data_num_parallel)
            
            logger.log_print("# create infer target dataset")
            input_trg_dataset = tf.data.Dataset.from_tensor_slices(input_trg_data)
            input_trg_word_dataset, input_trg_char_dataset = create_text_dataset(input_trg_dataset,
                trg_word_vocab_index, hyperparams.data_trg_word_max_length, hyperparams.data_trg_word_pad,
                hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index, hyperparams.data_trg_char_max_length,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable, hyperparams.data_num_parallel)
            
            logger.log_print("# create infer label dataset")
            input_label_dataset = tf.data.Dataset.from_tensor_slices(input_label_data)
            input_label_dataset = create_label_dataset(input_label_dataset, 1, hyperparams.data_num_parallel)
            
            logger.log_print("# create infer data pipeline")
            data_pipeline = create_data_pipeline(input_src_word_dataset, input_src_char_dataset,
                input_trg_word_dataset, input_trg_char_dataset, input_label_dataset, src_word_vocab_index,
                hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, trg_word_vocab_index,
                hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable,
                None, False, 0, len(input_data), hyperparams.train_eval_batch_size)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            external_data=external_data, mode="infer", scope=hyperparams.model_scope)
        
        return InferModel(graph=graph, model=model, data_pipeline=data_pipeline,
            word_embedding=word_embed_data, input_data=input_data, input_context=input_src_data,
            input_response=input_trg_data, input_label=input_label_data)

def get_model_creator(model_type):
    if model_type == "conv_enc":
        model_creator = ConvolutionEncoder
    elif model_type == "seq_enc":
        model_creator = SequenceEncoder
    elif model_type == "att_enc":
        model_creator = AttentionEncoder
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(model_type))
    
    return model_creator

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model,
               ckpt_file,
               ckpt_type):
    with model.graph.as_default():
        model.model.restore(sess, ckpt_file, ckpt_type)
