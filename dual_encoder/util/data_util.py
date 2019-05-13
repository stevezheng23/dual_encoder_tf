import codecs
import collections
import os.path
import json

import numpy as np
import tensorflow as tf

from util.default_util import *

__all__ = ["DataPipeline", "create_dynamic_pipeline", "create_data_pipeline",
           "create_text_dataset", "create_label_dataset", "generate_word_feat", "generate_char_feat", "generate_num_feat",
           "create_embedding_file", "load_embedding_file", "convert_embedding",
           "create_vocab_file", "load_vocab_file", "process_vocab_table", "create_word_vocab", "create_char_vocab",
           "load_tsv_data", "load_json_data", "load_dual_data", "prepare_data", "prepare_dual_data"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "input_src_word", "input_src_char", "input_trg_word", "input_trg_char", "input_label",
     "input_src_word_mask", "input_src_char_mask", "input_trg_word_mask", "input_trg_char_mask", "input_label_mask",
     "batch_size", "data_size_placeholder", "input_src_placeholder","input_trg_placeholder", "input_label_placeholder"))):
    pass

def create_dynamic_pipeline(input_src_word_dataset,
                            input_src_char_dataset,
                            input_trg_word_dataset,
                            input_trg_char_dataset,
                            input_label_dataset,
                            src_word_vocab_index,
                            src_word_pad,
                            src_word_feat_enable,
                            src_char_vocab_index,
                            src_char_pad,
                            src_char_feat_enable,
                            trg_word_vocab_index,
                            trg_word_pad,
                            trg_word_feat_enable,
                            trg_char_vocab_index,
                            trg_char_pad,
                            trg_char_feat_enable,
                            random_seed,
                            enable_shuffle,
                            buffer_size,
                            batch_size,
                            data_size_placeholder,
                            input_src_placeholder,
                            input_trg_placeholder,
                            input_label_placeholder):
    """create dynamic data pipeline for dual encoder"""
    default_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    default_dataset_tensor = tf.constant(0, shape=[1,1], dtype=tf.int32)
    
    if src_word_feat_enable == True:
        src_word_pad_id = tf.cast(src_word_vocab_index.lookup(tf.constant(src_word_pad)), dtype=tf.int32)
    else:
        src_word_pad_id = default_pad_id
        input_src_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
    
    if src_char_feat_enable == True:
        src_char_pad_id = tf.cast(src_char_vocab_index.lookup(tf.constant(src_char_pad)), dtype=tf.int32)
    else:
        src_char_pad_id = default_pad_id
        input_src_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
    
    if trg_word_feat_enable == True:
        trg_word_pad_id = tf.cast(trg_word_vocab_index.lookup(tf.constant(trg_word_pad)), dtype=tf.int32)
    else:
        trg_word_pad_id = default_pad_id
        input_trg_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
    
    if trg_char_feat_enable == True:
        trg_char_pad_id = tf.cast(trg_char_vocab_index.lookup(tf.constant(trg_char_pad)), dtype=tf.int32)
    else:
        trg_char_pad_id = default_pad_id
        input_trg_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
        
    dataset = tf.data.Dataset.zip((input_src_word_dataset, input_src_char_dataset,
        input_trg_word_dataset, input_trg_char_dataset, input_label_dataset))
    
    if enable_shuffle == True:
        dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    if src_word_feat_enable == True:
        input_src_word = batch_data[0]
        input_src_word_mask = tf.cast(tf.not_equal(batch_data[0], src_word_pad_id), dtype=tf.float32)
    else:
        input_src_word = None
        input_src_word_mask = None
    
    if src_char_feat_enable == True:
        input_src_char = batch_data[1]
        input_src_char_mask = tf.cast(tf.not_equal(batch_data[1], src_char_pad_id), dtype=tf.float32)
    else:
        input_src_char = None
        input_src_char_mask = None
    
    if trg_word_feat_enable == True:
        input_trg_word = batch_data[2]
        input_trg_word_mask = tf.cast(tf.not_equal(batch_data[2], trg_word_pad_id), dtype=tf.float32)
    else:
        input_trg_word = None
        input_trg_word_mask = None
    
    if trg_char_feat_enable == True:
        input_trg_char = batch_data[3]
        input_trg_char_mask = tf.cast(tf.not_equal(batch_data[3], trg_char_pad_id), dtype=tf.float32)
    else:
        input_trg_char = None
        input_trg_char_mask = None
    
    label_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    input_label = tf.cast(batch_data[4], dtype=tf.float32)
    input_label_mask = tf.cast(tf.greater_equal(batch_data[4], label_pad_id), dtype=tf.float32)
    
    return DataPipeline(initializer=iterator.initializer,
        input_src_word=input_src_word, input_src_char=input_src_char,
        input_trg_word=input_trg_word, input_trg_char=input_trg_char, input_label=input_label,
        input_src_word_mask=input_src_word_mask, input_src_char_mask=input_src_char_mask,
        input_trg_word_mask=input_trg_word_mask, input_trg_char_mask=input_trg_char_mask,
        input_label_mask=input_label_mask, batch_size=batch_size,
        data_size_placeholder=data_size_placeholder, input_src_placeholder=input_src_placeholder,
        input_trg_placeholder=input_trg_placeholder, input_label_placeholder=input_label_placeholder)

def create_data_pipeline(input_src_word_dataset,
                         input_src_char_dataset,
                         input_trg_word_dataset,
                         input_trg_char_dataset,
                         input_label_dataset,
                         src_word_vocab_index,
                         src_word_pad,
                         src_word_feat_enable,
                         src_char_vocab_index,
                         src_char_pad,
                         src_char_feat_enable,
                         trg_word_vocab_index,
                         trg_word_pad,
                         trg_word_feat_enable,
                         trg_char_vocab_index,
                         trg_char_pad,
                         trg_char_feat_enable,
                         random_seed,
                         enable_shuffle,
                         buffer_size,
                         batch_size,
                         data_size):
    """create data pipeline for dual encoder"""
    default_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    default_dataset_tensor = tf.constant(0, shape=[1,1], dtype=tf.int32)
    
    if src_word_feat_enable == True:
        src_word_pad_id = tf.cast(src_word_vocab_index.lookup(tf.constant(src_word_pad)), dtype=tf.int32)
    else:
        src_word_pad_id = default_pad_id
        input_src_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
    
    if src_char_feat_enable == True:
        src_char_pad_id = tf.cast(src_char_vocab_index.lookup(tf.constant(src_char_pad)), dtype=tf.int32)
    else:
        src_char_pad_id = default_pad_id
        input_src_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
    
    if trg_word_feat_enable == True:
        trg_word_pad_id = tf.cast(trg_word_vocab_index.lookup(tf.constant(trg_word_pad)), dtype=tf.int32)
    else:
        trg_word_pad_id = default_pad_id
        input_trg_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
    
    if trg_char_feat_enable == True:
        trg_char_pad_id = tf.cast(trg_char_vocab_index.lookup(tf.constant(trg_char_pad)), dtype=tf.int32)
    else:
        trg_char_pad_id = default_pad_id
        input_trg_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
    
    dataset = tf.data.Dataset.zip((input_src_word_dataset, input_src_char_dataset,
        input_trg_word_dataset, input_trg_char_dataset, input_label_dataset))
    
    if enable_shuffle == True:
        dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    if src_word_feat_enable == True:
        input_src_word = batch_data[0]
        input_src_word_mask = tf.cast(tf.not_equal(batch_data[0], src_word_pad_id), dtype=tf.float32)
    else:
        input_src_word = None
        input_src_word_mask = None
    
    if src_char_feat_enable == True:
        input_src_char = batch_data[1]
        input_src_char_mask = tf.cast(tf.not_equal(batch_data[1], src_char_pad_id), dtype=tf.float32)
    else:
        input_src_char = None
        input_src_char_mask = None
    
    if trg_word_feat_enable == True:
        input_trg_word = batch_data[2]
        input_trg_word_mask = tf.cast(tf.not_equal(batch_data[2], trg_word_pad_id), dtype=tf.float32)
    else:
        input_trg_word = None
        input_trg_word_mask = None
    
    if trg_char_feat_enable == True:
        input_trg_char = batch_data[3]
        input_trg_char_mask = tf.cast(tf.not_equal(batch_data[3], trg_char_pad_id), dtype=tf.float32)
    else:
        input_trg_char = None
        input_trg_char_mask = None
    
    label_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    input_label = tf.cast(batch_data[4], dtype=tf.float32)
    input_label_mask = tf.cast(tf.greater_equal(batch_data[4], label_pad_id), dtype=tf.float32)
    
    return DataPipeline(initializer=iterator.initializer,
        input_src_word=input_src_word, input_src_char=input_src_char,
        input_trg_word=input_trg_word, input_trg_char=input_trg_char, input_label=input_label,
        input_src_word_mask=input_src_word_mask, input_src_char_mask=input_src_char_mask,
        input_trg_word_mask=input_trg_word_mask, input_trg_char_mask=input_trg_char_mask,
        input_label_mask=input_label_mask, batch_size=batch_size,
        data_size_placeholder=None, input_src_placeholder=None,
        input_trg_placeholder=None, input_label_placeholder=None)
    
def create_text_dataset(input_data_set,
                        word_vocab_index,
                        word_max_size,
                        word_pad,
                        word_feat_enable,
                        char_vocab_index,
                        char_max_size,
                        char_pad,
                        char_feat_enable,
                        num_parallel):
    """create word/char-level dataset for input source data"""
    dataset = input_data_set
    
    word_dataset = None
    if word_feat_enable == True:
        word_dataset = dataset.map(lambda sent: generate_word_feat(sent,
            word_vocab_index, word_max_size, word_pad), num_parallel_calls=num_parallel)
    
    char_dataset = None
    if char_feat_enable == True:
        char_dataset = dataset.map(lambda sent: generate_char_feat(sent,
            word_max_size, char_vocab_index, char_max_size, char_pad), num_parallel_calls=num_parallel)
    
    return word_dataset, char_dataset

def create_label_dataset(input_data_set,
                         string_max_size,
                         num_parallel):
    """create label dataset for input target data"""
    dataset = input_data_set
    
    num_dataset = dataset.map(lambda sent: generate_num_feat(sent, string_max_size), num_parallel_calls=num_parallel)
    
    return num_dataset

def generate_word_feat(sentence,
                       word_vocab_index,
                       word_max_size,
                       word_pad):
    """process words for sentence"""
    sentence_words = tf.string_split([sentence], delimiter=' ').values
    sentence_words = tf.concat([sentence_words[:word_max_size],
        tf.constant(word_pad, shape=[word_max_size])], axis=0)
    sentence_words = tf.reshape(sentence_words[:word_max_size], shape=[word_max_size])
    sentence_words = tf.cast(word_vocab_index.lookup(sentence_words), dtype=tf.int32)
    sentence_words = tf.expand_dims(sentence_words, axis=-1)
    
    return sentence_words

def generate_char_feat(sentence,
                       word_max_size,
                       char_vocab_index,
                       char_max_size,
                       char_pad):
    """generate characters for sentence"""
    def word_to_char(word):
        """process characters for word"""
        word_chars = tf.string_split([word], delimiter='').values
        word_chars = tf.concat([word_chars[:char_max_size],
            tf.constant(char_pad, shape=[char_max_size])], axis=0)
        word_chars = tf.reshape(word_chars[:char_max_size], shape=[char_max_size])
        
        return word_chars
    
    sentence_words = tf.string_split([sentence], delimiter=' ').values
    sentence_words = tf.concat([sentence_words[:word_max_size],
        tf.constant(char_pad, shape=[word_max_size])], axis=0)
    sentence_words = tf.reshape(sentence_words[:word_max_size], shape=[word_max_size])
    sentence_chars = tf.map_fn(word_to_char, sentence_words)
    sentence_chars = tf.cast(char_vocab_index.lookup(sentence_chars), dtype=tf.int32)
    
    return sentence_chars

def generate_num_feat(strings,
                      string_max_size):
    """generate numbers for strings"""
    string_nums = tf.string_to_number(strings, out_type=tf.int32)
    string_nums = tf.expand_dims(string_nums, axis=-1)
    
    return string_nums

def create_embedding_file(embedding_file,
                          embedding_table):
    """create embedding file based on embedding table"""
    embedding_dir = os.path.dirname(embedding_file)
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    
    if not os.path.exists(embedding_file):
        with codecs.getwriter("utf-8")(open(embedding_file, "wb")) as file:
            for vocab in embedding_table.keys():
                embed = embedding_table[vocab]
                embed_str = " ".join(map(str, embed))
                file.write("{0} {1}\n".format(vocab, embed_str))

def load_embedding_file(embedding_file,
                        embedding_size,
                        unk,
                        pad):
    """load pre-train embeddings from embedding file"""
    if os.path.exists(embedding_file):
        with codecs.getreader("utf-8")(open(embedding_file, "rb")) as file:
            embedding = {}
            for line in file:
                items = line.strip().split(' ')
                if len(items) != embedding_size + 1:
                    continue
                word = items[0]
                vector = [float(x) for x in items[1:]]
                if word not in embedding:
                    embedding[word] = vector
            
            if unk not in embedding:
                embedding[unk] = np.random.rand(embedding_size)
            if pad not in embedding:
                embedding[pad] = np.random.rand(embedding_size)
            
            return embedding
    else:
        raise FileNotFoundError("embedding file not found")

def convert_embedding(embedding_lookup):
    if embedding_lookup is not None:
        embedding = [v for k,v in embedding_lookup.items()]
    else:
        embedding = None
    
    return embedding

def create_vocab_file(vocab_file,
                      vocab_table):
    """create vocab file based on vocab table"""
    vocab_dir = os.path.dirname(vocab_file)
    if not os.path.exists(vocab_dir):
        os.mkdir(vocab_dir)
    
    if not os.path.exists(vocab_file):
        with codecs.getwriter("utf-8")(open(vocab_file, "wb")) as file:
            for vocab in vocab_table:
                file.write("{0}\n".format(vocab))

def load_vocab_file(vocab_file):
    """load vocab data from vocab file"""
    if os.path.exists(vocab_file):
        with codecs.getreader("utf-8")(open(vocab_file, "rb")) as file:
            vocab = {}
            for line in file:
                items = line.strip().split('\t')
                
                item_size = len(items)
                if item_size > 1:
                    vocab[items[0]] = int(items[1])
                elif item_size > 0:
                    vocab[items[0]] = MAX_INT
            
            return vocab
    else:
        raise FileNotFoundError("vocab file not found")

def process_vocab_table(vocab,
                        vocab_size,
                        vocab_threshold,
                        vocab_lookup,
                        unk,
                        pad):
    """process vocab table"""
    default_vocab = [unk, pad]
    
    if unk in vocab:
        del vocab[unk]
    if pad in vocab:
        del vocab[pad]
    
    vocab = { k: vocab[k] for k in vocab.keys() if vocab[k] >= vocab_threshold }
    if vocab_lookup is not None:
        vocab = { k: vocab[k] for k in vocab.keys() if k in vocab_lookup }
    
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    sorted_vocab = default_vocab + sorted_vocab
    
    vocab_table = sorted_vocab[:vocab_size]
    vocab_size = len(vocab_table)
    
    vocab_index = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(vocab_table), default_value=0)
    vocab_inverted_index = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=tf.constant(vocab_table), default_value=unk)
    
    return vocab_table, vocab_size, vocab_index, vocab_inverted_index

def create_word_vocab(input_data):
    """create word vocab from input data"""
    word_vocab = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            if word not in word_vocab:
                word_vocab[word] = 1
            else:
                word_vocab[word] += 1
    
    return word_vocab

def create_char_vocab(input_data):
    """create char vocab from input data"""
    char_vocab = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            chars = list(word)
            for ch in chars:
                if ch not in char_vocab:
                    char_vocab[ch] = 1
                else:
                    char_vocab[ch] += 1
    
    return char_vocab

def load_tsv_data(input_file):
    """load data from tsv file"""
    if os.path.exists(input_file):
        with codecs.getreader("utf-8")(open(input_file, "rb")) as file:
            input_data = []
            item_separator = "\t"
            for line in file:
                items = line.strip().split(item_separator)
                num_items = len(items)
                
                if num_items < 3:
                    continue
                
                sample_id = items[0]
                source = items[1]
                target = items[2]
                label = "1"
                
                if num_items > 3:
                    label = items[3]
                
                input_data.append({
                    "id": sample_id,
                    "source": source,
                    "target": target,
                    "label": label
                })
            
            return input_data
    else:
        raise FileNotFoundError("input file not found")

def load_json_data(input_file):
    """load data from json file"""
    if os.path.exists(input_file):
        with codecs.getreader("utf-8")(open(input_file, "rb") ) as file:
            input_data = json.load(file)
            input_data = [{
                "id": data["id"],
                "source": data["source"],
                "target": data["target"],
                "label": data["label"] if "label" in data else "1"
            } for data in input_data]
            
            return input_data
    else:
        raise FileNotFoundError("input file not found")

def load_dual_data(input_file,
                   file_type):
    """load dual data from input file"""
    if file_type == "tsv":
        input_data = load_tsv_data(input_file)
    elif file_type == "json":
        input_data = load_json_data(input_file)
    else:
        raise ValueError("can not load data from unsupported file type {0}".format(file_type))
    
    return input_data

def prepare_data(logger,
                 input_data,
                 word_vocab_file,
                 word_vocab_size,
                 word_vocab_threshold,
                 word_embed_dim,
                 word_embed_file,
                 word_embed_full_file,
                 word_embed_pretrained,
                 word_unk,
                 word_pad,
                 word_feat_enable,
                 char_vocab_file,
                 char_vocab_size,
                 char_vocab_threshold,
                 char_unk,
                 char_pad,
                 char_feat_enable):
    """prepare data"""    
    word_embed_data = None
    if word_embed_pretrained == True:
        if os.path.exists(word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(word_embed_file))
            word_embed_data = load_embedding_file(word_embed_file, word_embed_dim, word_unk, word_pad)
        elif os.path.exists(word_embed_full_file):
            logger.log_print("# loading word embeddings from {0}".format(word_embed_full_file))
            word_embed_data = load_embedding_file(word_embed_full_file, word_embed_dim, word_unk, word_pad)
        else:
            raise ValueError("{0} or {1} must be provided".format(word_embed_file, word_embed_full_file))
        
        word_embed_size = len(word_embed_data) if word_embed_data is not None else 0
        logger.log_print("# word embedding table has {0} words".format(word_embed_size))
    
    word_vocab = None
    word_vocab_index = None
    word_vocab_inverted_index = None
    if os.path.exists(word_vocab_file):
        logger.log_print("# loading word vocab table from {0}".format(word_vocab_file))
        word_vocab = load_vocab_file(word_vocab_file)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad)
    elif input_data is not None:
        logger.log_print("# creating word vocab table from input data")
        word_vocab = create_word_vocab(input_data)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad)
        logger.log_print("# creating word vocab file {0}".format(word_vocab_file))
        create_vocab_file(word_vocab_file, word_vocab_table)
    else:
        raise ValueError("{0} or input data must be provided".format(word_vocab_file))
    
    logger.log_print("# word vocab table has {0} words".format(word_vocab_size))
    
    char_vocab = None
    char_vocab_index = None
    char_vocab_inverted_index = None
    if char_feat_enable is True:
        if os.path.exists(char_vocab_file):
            logger.log_print("# loading char vocab table from {0}".format(char_vocab_file))
            char_vocab = load_vocab_file(char_vocab_file)
            (_, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad)
        elif input_data is not None:
            logger.log_print("# creating char vocab table from input data")
            char_vocab = create_char_vocab(input_data)
            (char_vocab_table, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad)
            logger.log_print("# creating char vocab file {0}".format(char_vocab_file))
            create_vocab_file(char_vocab_file, char_vocab_table)
        else:
            raise ValueError("{0} or input data must be provided".format(char_vocab_file))

        logger.log_print("# char vocab table has {0} chars".format(char_vocab_size))
    
    if word_embed_data is not None and word_vocab_table is not None:
        word_embed_data = { k: word_embed_data[k] for k in word_vocab_table if k in word_embed_data }
        logger.log_print("# word embedding table has {0} words after filtering".format(len(word_embed_data)))
        if not os.path.exists(word_embed_file):
            logger.log_print("# creating word embedding file {0}".format(word_embed_file))
            create_embedding_file(word_embed_file, word_embed_data)
        
        word_embed_data = convert_embedding(word_embed_data)
    
    return (word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index)

def prepare_dual_data(logger,
                      input_dual_file,
                      input_file_type,
                      batch_padding_enable,
                      batch_size,
                      share_vocab,
                      src_word_vocab_file,
                      src_word_vocab_size,
                      src_word_vocab_threshold,
                      src_word_embed_dim,
                      src_word_embed_file,
                      src_word_embed_full_file,
                      src_word_embed_pretrained,
                      src_word_unk,
                      src_word_pad,
                      src_word_feat_enable,
                      src_char_vocab_file,
                      src_char_vocab_size,
                      src_char_vocab_threshold,
                      src_char_unk,
                      src_char_pad,
                      src_char_feat_enable,
                      trg_word_vocab_file,
                      trg_word_vocab_size,
                      trg_word_vocab_threshold,
                      trg_word_embed_dim,
                      trg_word_embed_file,
                      trg_word_embed_full_file,
                      trg_word_embed_pretrained,
                      trg_word_unk,
                      trg_word_pad,
                      trg_word_feat_enable,
                      trg_char_vocab_file,
                      trg_char_vocab_size,
                      trg_char_vocab_threshold,
                      trg_char_unk,
                      trg_char_pad,
                      trg_char_feat_enable):
    """prepare dual data"""
    logger.log_print("# loading input dual data from {0}".format(input_dual_file))
    input_dual_data = load_dual_data(input_dual_file, input_file_type)
    
    input_dual_size = len(input_dual_data)
    logger.log_print("# input dual data has {0} lines".format(input_dual_size))
    
    if batch_padding_enable == True and input_dual_size % batch_size != 0:
        padding_size = batch_size - input_dual_size % batch_size
        input_dual_data = input_dual_data + input_dual_data[:padding_size] 
    
    input_src_data = [dual_data["source"] for dual_data in input_dual_data]
    input_trg_data = [dual_data["target"] for dual_data in input_dual_data]
    input_label_data = [dual_data["label"] for dual_data in input_dual_data]
    
    logger.log_print("# prepare source data")
    if share_vocab == False:
        (src_word_embed_data, src_word_vocab_size, src_word_vocab_index, src_word_vocab_inverted_index,
            src_char_vocab_size, src_char_vocab_index, src_char_vocab_inverted_index) = prepare_data(logger, input_src_data,
                src_word_vocab_file, src_word_vocab_size, src_word_vocab_threshold, src_word_embed_dim, src_word_embed_file,
                src_word_embed_full_file, src_word_embed_pretrained, src_word_unk, src_word_pad, src_word_feat_enable, 
                src_char_vocab_file, src_char_vocab_size, src_char_vocab_threshold, src_char_unk, src_char_pad, src_char_feat_enable)
    else:
        input_data = []
        input_data.update(input_src_data)
        input_data.update(input_trg_data)
        (src_word_embed_data, src_word_vocab_size, src_word_vocab_index, src_word_vocab_inverted_index,
            src_char_vocab_size, src_char_vocab_index, src_char_vocab_inverted_index) = prepare_data(logger, input_src_data,
                src_word_vocab_file, src_word_vocab_size, src_word_vocab_threshold, src_word_embed_dim, src_word_embed_file,
                src_word_embed_full_file, src_word_embed_pretrained, src_word_unk, src_word_pad, src_word_feat_enable, 
                src_char_vocab_file, src_char_vocab_size, src_char_vocab_threshold, src_char_unk, src_char_pad, src_char_feat_enable)
    
    logger.log_print("# prepare target data")
    if share_vocab == False:
        (trg_word_embed_data, trg_word_vocab_size, trg_word_vocab_index, trg_word_vocab_inverted_index,
            trg_char_vocab_size, trg_char_vocab_index, trg_char_vocab_inverted_index) = prepare_data(logger, input_trg_data,
                trg_word_vocab_file, trg_word_vocab_size, trg_word_vocab_threshold, trg_word_embed_dim, trg_word_embed_file,
                trg_word_embed_full_file, trg_word_embed_pretrained, trg_word_unk, trg_word_pad, trg_word_feat_enable, 
                trg_char_vocab_file, trg_char_vocab_size, trg_char_vocab_threshold, trg_char_unk, trg_char_pad, trg_char_feat_enable)
    else:
        trg_word_embed_data = src_word_embed_data
        trg_word_vocab_size = src_word_vocab_size
        trg_word_vocab_index = src_word_vocab_index
        trg_word_vocab_inverted_index = src_word_vocab_inverted_index
        trg_char_vocab_size = src_char_vocab_size
        trg_char_vocab_index = src_char_vocab_index
        trg_char_vocab_inverted_index = src_char_vocab_inverted_index
    
    return (input_dual_data, input_src_data, input_trg_data, input_label_data, src_word_embed_data,
        src_word_vocab_size, src_word_vocab_index, src_word_vocab_inverted_index, src_char_vocab_size,
        src_char_vocab_index, src_char_vocab_inverted_index, trg_word_embed_data, trg_word_vocab_size, trg_word_vocab_index,
        trg_word_vocab_inverted_index, trg_char_vocab_size, trg_char_vocab_index, trg_char_vocab_inverted_index)
