import collections

import numpy as np
import tensorflow as tf

from model.conv_enc import *
from model.seq_enc import *
from model.att_enc import *
from util.data_util import *

__all__ = ["TrainModel", "InferModel", "SimilarityModel", "EmbeddingModel",
           "create_train_model", "create_infer_model", "create_similarity_model", "create_embedding_model",
           "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel",
    ("graph", "model", "data_pipeline", "src_word_embed", "trg_word_embed",
     "input_data", "input_src_data", "input_trg_data", "input_label_data"))):
    pass

class InferModel(collections.namedtuple("InferModel",
    ("graph", "model", "data_pipeline", "src_word_embed", "trg_word_embed",
     "input_data", "input_src_data", "input_trg_data", "input_label_data"))):
    pass

class SimilarityModel(collections.namedtuple("SimilarityModel", ("model", "data_pipeline"))):
    pass

class EmbeddingModel(collections.namedtuple("EmbeddingModel", ("model", "data_pipeline"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_data, input_src_data, input_trg_data, input_label_data, src_word_embed_data,
            src_word_vocab_size, src_word_vocab_index, src_word_vocab_inverted_index, src_char_vocab_size,
            src_char_vocab_index, src_char_vocab_inverted_index, trg_word_embed_data, trg_word_vocab_size,
            trg_word_vocab_index, trg_word_vocab_inverted_index, trg_char_vocab_size, trg_char_vocab_index,
            trg_char_vocab_inverted_index) = prepare_dual_data(logger, hyperparams.data_train_dual_file,
                hyperparams.data_train_dual_file_type, True, hyperparams.train_batch_size, hyperparams.data_share_vocab,
                hyperparams.data_src_word_vocab_file, hyperparams.data_src_word_vocab_size, hyperparams.data_src_word_vocab_threshold,
                hyperparams.model_representation_src_word_embed_dim, hyperparams.data_src_embed_file,
                hyperparams.data_src_embed_full_file, hyperparams.model_representation_src_word_embed_pretrained,
                hyperparams.data_src_word_unk, hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable,
                hyperparams.data_src_char_vocab_file, hyperparams.data_src_char_vocab_size, hyperparams.data_src_char_vocab_threshold,
                hyperparams.data_src_char_unk, hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable,
                hyperparams.data_trg_word_vocab_file, hyperparams.data_trg_word_vocab_size, hyperparams.data_trg_word_vocab_threshold,
                hyperparams.model_representation_trg_word_embed_dim, hyperparams.data_trg_embed_file,
                hyperparams.data_trg_embed_full_file, hyperparams.model_representation_trg_word_embed_pretrained,
                hyperparams.data_trg_word_unk, hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable,
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
            data_pipeline = create_dynamic_pipeline(input_src_word_dataset, input_src_char_dataset,
                input_trg_word_dataset, input_trg_char_dataset, input_label_dataset, src_word_vocab_index,
                hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, trg_word_vocab_index,
                hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable, hyperparams.train_random_seed,
                hyperparams.train_enable_shuffle, hyperparams.train_shuffle_buffer_size, hyperparams.train_batch_size,
                data_size_placeholder, input_src_placeholder, input_trg_placeholder, input_label_placeholder)
        else:
            if src_word_embed_data is not None and trg_word_embed_data is not None:
                external_data["src_word_embed"] = src_word_embed_data
                external_data["trg_word_embed"] = trg_word_embed_data
            
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
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable, hyperparams.train_random_seed,
                hyperparams.train_enable_shuffle, hyperparams.train_shuffle_buffer_size, hyperparams.train_batch_size, len(input_data))
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            external_data=external_data, mode="train", scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline,
            src_word_embed=src_word_embed_data, trg_word_embed=trg_word_embed_data, input_data=input_data,
            input_src_data=input_src_data, input_trg_data=input_trg_data, input_label_data=input_label_data)

def create_infer_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare infer data")
        (input_data, input_src_data, input_trg_data, input_label_data, src_word_embed_data,
            src_word_vocab_size, src_word_vocab_index, src_word_vocab_inverted_index, src_char_vocab_size,
            src_char_vocab_index, src_char_vocab_inverted_index, trg_word_embed_data, trg_word_vocab_size,
            trg_word_vocab_index, trg_word_vocab_inverted_index, trg_char_vocab_size, trg_char_vocab_index,
            trg_char_vocab_inverted_index) = prepare_dual_data(logger, hyperparams.data_eval_dual_file,
                hyperparams.data_eval_dual_file_type, False, hyperparams.train_eval_batch_size, hyperparams.data_share_vocab,
                hyperparams.data_src_word_vocab_file, hyperparams.data_src_word_vocab_size, hyperparams.data_src_word_vocab_threshold,
                hyperparams.model_representation_src_word_embed_dim, hyperparams.data_src_embed_file,
                hyperparams.data_src_embed_full_file, hyperparams.model_representation_src_word_embed_pretrained,
                hyperparams.data_src_word_unk, hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable,
                hyperparams.data_src_char_vocab_file, hyperparams.data_src_char_vocab_size, hyperparams.data_src_char_vocab_threshold,
                hyperparams.data_src_char_unk, hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable,
                hyperparams.data_trg_word_vocab_file, hyperparams.data_trg_word_vocab_size, hyperparams.data_trg_word_vocab_threshold,
                hyperparams.model_representation_trg_word_embed_dim, hyperparams.data_trg_embed_file,
                hyperparams.data_trg_embed_full_file, hyperparams.model_representation_trg_word_embed_pretrained,
                hyperparams.data_trg_word_unk, hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable,
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
            data_pipeline = create_dynamic_pipeline(input_src_word_dataset, input_src_char_dataset,
                input_trg_word_dataset, input_trg_char_dataset, input_label_dataset, src_word_vocab_index,
                hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index,
                hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable, trg_word_vocab_index,
                hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index,
                hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable,
                None, False, 0, hyperparams.train_eval_batch_size,
                data_size_placeholder, input_src_placeholder, input_trg_placeholder, input_label_placeholder)
        else:
            if src_word_embed_data is not None and trg_word_embed_data is not None:
                external_data["src_word_embed"] = src_word_embed_data
                external_data["trg_word_embed"] = trg_word_embed_data
            
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
                None, False, 0, hyperparams.train_eval_batch_size, len(input_data))
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            external_data=external_data, mode="infer", scope=hyperparams.model_scope)
        
        return InferModel(graph=graph, model=model, data_pipeline=data_pipeline,
            src_word_embed=src_word_embed_data, trg_word_embed=trg_word_embed_data, input_data=input_data,
            input_src_data=input_src_data, input_trg_data=input_trg_data, input_label_data=input_label_data)

def create_similarity_model(logger,
                            hyperparams):
    logger.log_print("# prepare similarity data")
    logger.log_print("# prepare similarity source data")
    (src_word_embed_data, src_word_vocab_size, src_word_vocab_index, src_word_vocab_inverted_index,
        src_char_vocab_size, src_char_vocab_index, src_char_vocab_inverted_index) = prepare_data(logger, None,
            hyperparams.data_src_word_vocab_file, hyperparams.data_src_word_vocab_size, hyperparams.data_src_word_vocab_threshold,
            hyperparams.model_representation_src_word_embed_dim, hyperparams.data_src_embed_file,
            hyperparams.data_src_embed_full_file, hyperparams.model_representation_src_word_embed_pretrained,
            hyperparams.data_src_word_unk, hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable,
            hyperparams.data_src_char_vocab_file, hyperparams.data_src_char_vocab_size, hyperparams.data_src_char_vocab_threshold,
            hyperparams.data_src_char_unk, hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable)
    
    logger.log_print("# prepare similarity target data")
    if hyperparams.data_share_vocab == False:
        (trg_word_embed_data, trg_word_vocab_size, trg_word_vocab_index, trg_word_vocab_inverted_index,
            trg_char_vocab_size, trg_char_vocab_index, trg_char_vocab_inverted_index) = prepare_data(logger, None,
                hyperparams.data_trg_word_vocab_file, hyperparams.data_trg_word_vocab_size, hyperparams.data_trg_word_vocab_threshold,
                hyperparams.model_representation_trg_word_embed_dim, hyperparams.data_trg_embed_file,
                hyperparams.data_trg_embed_full_file, hyperparams.model_representation_trg_word_embed_pretrained,
                hyperparams.data_trg_word_unk, hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable,
                hyperparams.data_trg_char_vocab_file, hyperparams.data_trg_char_vocab_size, hyperparams.data_trg_char_vocab_threshold,
                hyperparams.data_trg_char_unk, hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable)
    else:
        trg_word_embed_data = src_word_embed_data
        trg_word_vocab_size = src_word_vocab_size
        trg_word_vocab_index = src_word_vocab_index
        trg_word_vocab_inverted_index = src_word_vocab_inverted_index
        trg_char_vocab_size = src_char_vocab_size
        trg_char_vocab_index = src_char_vocab_index
        trg_char_vocab_inverted_index = src_char_vocab_inverted_index
    
    external_data = {}
    if src_word_embed_data is not None and trg_word_embed_data is not None:
        external_data["src_word_embed"] = src_word_embed_data
        external_data["trg_word_embed"] = trg_word_embed_data
    
    logger.log_print("# create similarity data pipeline")
    data_pipeline = create_similarity_pipeline(hyperparams.data_external_index_enable,
        src_word_vocab_index, hyperparams.data_src_word_max_length, hyperparams.data_src_word_pad,
        hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index, hyperparams.data_src_char_max_length,
        hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable,
        trg_word_vocab_index, hyperparams.data_trg_word_max_length, hyperparams.data_trg_word_pad,
        hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index, hyperparams.data_trg_char_max_length,
        hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable)

    model_creator = get_model_creator(hyperparams.model_type)
    model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
        external_data=external_data, mode="similarity", scope=hyperparams.model_scope)
    
    return SimilarityModel(model=model, data_pipeline=data_pipeline)

def create_embedding_model(logger,
                           hyperparams):
    logger.log_print("# prepare embedding data")
    logger.log_print("# prepare embedding source data")
    (src_word_embed_data, src_word_vocab_size, src_word_vocab_index, src_word_vocab_inverted_index,
        src_char_vocab_size, src_char_vocab_index, src_char_vocab_inverted_index) = prepare_data(logger, None,
            hyperparams.data_src_word_vocab_file, hyperparams.data_src_word_vocab_size, hyperparams.data_src_word_vocab_threshold,
            hyperparams.model_representation_src_word_embed_dim, hyperparams.data_src_embed_file,
            hyperparams.data_src_embed_full_file, hyperparams.model_representation_src_word_embed_pretrained,
            hyperparams.data_src_word_unk, hyperparams.data_src_word_pad, hyperparams.model_representation_src_word_feat_enable,
            hyperparams.data_src_char_vocab_file, hyperparams.data_src_char_vocab_size, hyperparams.data_src_char_vocab_threshold,
            hyperparams.data_src_char_unk, hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable)
    
    logger.log_print("# prepare embedding target data")
    if hyperparams.data_share_vocab == False:
        (trg_word_embed_data, trg_word_vocab_size, trg_word_vocab_index, trg_word_vocab_inverted_index,
            trg_char_vocab_size, trg_char_vocab_index, trg_char_vocab_inverted_index) = prepare_data(logger, None,
                hyperparams.data_trg_word_vocab_file, hyperparams.data_trg_word_vocab_size, hyperparams.data_trg_word_vocab_threshold,
                hyperparams.model_representation_trg_word_embed_dim, hyperparams.data_trg_embed_file,
                hyperparams.data_trg_embed_full_file, hyperparams.model_representation_trg_word_embed_pretrained,
                hyperparams.data_trg_word_unk, hyperparams.data_trg_word_pad, hyperparams.model_representation_trg_word_feat_enable,
                hyperparams.data_trg_char_vocab_file, hyperparams.data_trg_char_vocab_size, hyperparams.data_trg_char_vocab_threshold,
                hyperparams.data_trg_char_unk, hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable)
    else:
        trg_word_embed_data = src_word_embed_data
        trg_word_vocab_size = src_word_vocab_size
        trg_word_vocab_index = src_word_vocab_index
        trg_word_vocab_inverted_index = src_word_vocab_inverted_index
        trg_char_vocab_size = src_char_vocab_size
        trg_char_vocab_index = src_char_vocab_index
        trg_char_vocab_inverted_index = src_char_vocab_inverted_index
    
    external_data = {}
    if src_word_embed_data is not None and trg_word_embed_data is not None:
        external_data["src_word_embed"] = src_word_embed_data
        external_data["trg_word_embed"] = trg_word_embed_data
    
    logger.log_print("# create embedding data pipeline")
    data_pipeline = create_embedding_pipeline(hyperparams.data_external_index_enable,
        src_word_vocab_index, hyperparams.data_src_word_max_length, hyperparams.data_src_word_pad,
        hyperparams.model_representation_src_word_feat_enable, src_char_vocab_index, hyperparams.data_src_char_max_length,
        hyperparams.data_src_char_pad, hyperparams.model_representation_src_char_feat_enable,
        trg_word_vocab_index, hyperparams.data_trg_word_max_length, hyperparams.data_trg_word_pad,
        hyperparams.model_representation_trg_word_feat_enable, trg_char_vocab_index, hyperparams.data_trg_char_max_length,
        hyperparams.data_trg_char_pad, hyperparams.model_representation_trg_char_feat_enable)

    model_creator = get_model_creator(hyperparams.model_type)
    model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
        external_data=external_data, mode="embedding", scope=hyperparams.model_scope)
    
    return EmbeddingModel(model=model, data_pipeline=data_pipeline)

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
