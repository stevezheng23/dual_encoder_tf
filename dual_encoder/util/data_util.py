import codecs
import collections
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["DataPipeline", "create_data_pipeline", "create_input_dataset", "create_output_dataset",
           "generate_word_feat", "generate_subword_feat", "generate_char_feat",
           "create_embedding_file", "load_embedding_file", "convert_embedding",
           "create_vocab_file", "load_vocab_file", "process_vocab_table",
           "create_word_vocab", "create_subword_vocab", "create_char_vocab",
           "load_input_data", "prepare_data"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "input_source_word", "input_source_subword", "input_source_char",
     "input_target_word", "input_target_subword", "input_target_char", "output_label_word",
     "input_source_word_mask", "input_source_subword_mask", "input_source_char_mask",
     "input_target_word_mask", "input_target_subword_mask", "input_target_char_mask",
     "output_label_word_mask", "input_data_placeholder", "batch_size_placeholder"))):
    pass

def create_data_pipeline(input_src_word_dataset,
                         input_src_subword_dataset,
                         input_src_char_dataset,
                         input_trg_word_dataset,
                         input_trg_subword_dataset,
                         input_trg_char_dataset,
                         output_label_word_dataset,
                         word_vocab_index,
                         word_pad,
                         word_feat_enable,
                         subword_vocab_index,
                         subword_pad,
                         subword_feat_enable,
                         char_vocab_index,
                         char_pad,
                         char_feat_enable,
                         batch_size,
                         random_seed,
                         enable_shuffle):
    """create data pipeline for dual encoder"""
    word_pad_id = word_vocab_index.lookup(tf.constant(word_pad))
    subword_pad_id = subword_vocab_index.lookup(tf.constant(subword_pad))
    char_pad_id = char_vocab_index.lookup(tf.constant(char_pad))
    
    dataset = tf.data.Dataset.zip((input_src_word_dataset, input_src_subword_dataset, input_src_char_dataset,
        input_trg_word_dataset, input_trg_subword_dataset, input_trg_char_dataset, output_label_word_dataset))
    
    if enable_shuffle == True:
        buffer_size = batch_size * 1000
        dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([None])),
        padding_values=(
            word_pad_id,
            word_pad_id,
            subword_pad_id,
            subword_pad_id,
            char_pad_id,
            char_pad_id,
            word_pad_id))
        
    iterator = dataset.make_initializable_iterator()
    (input_src_word, input_src_subword, input_src_char, input_trg_word,
        input_trg_subword, input_trg_char, output_label_word) = iterator.get_next()
    
    return DataPipeline(initializer=iterator.initializer, input_source_word=input_src_word, input_source_subword=input_src_subword,
        input_source_char=input_src_char, input_target_word=input_trg_word, input_target_subword=input_trg_subword,
        input_target_char=input_trg_char, output_label_word=output_label_word, input_source_word_mask=None,
        input_source_subword_mask=None, input_source_char_mask=None, input_target_word_mask=None,
        input_target_subword_mask=None, input_target_char_mask=None, output_label_word_mask=None,
        input_data_placeholder=None, batch_size_placeholder=None)
    
def create_input_dataset(input_file,
                         word_vocab_index,
                         word_max_length,
                         word_pad,
                         word_sos,
                         word_eos,
                         word_feat_enable,
                         subword_vocab_index,
                         subword_max_length,
                         subword_pad,
                         subword_size,
                         subword_feat_enable,
                         char_vocab_index,
                         char_max_length,
                         char_pad,
                         char_feat_enable):
    """create word/subword/char-level dataset for input data"""
    dataset = tf.data.TextLineDataset([input_file])
    
    word_dataset = None
    subword_dataset = None
    char_dataset = None
    if word_feat_enable == True:
        word_dataset = dataset.map(lambda sent: generate_word_feat(sent, word_vocab_index,
            word_max_length, word_sos, word_eos))

    if subword_feat_enable == True:
        subword_pad_id = subword_vocab_index.lookup(tf.constant(subword_pad))
        subword_dataset = dataset.map(lambda sent: generate_subword_feat(sent, subword_vocab_index,
            word_max_length, subword_max_length, subword_size, word_sos, word_eos, subword_pad_id))

    if char_feat_enable == True:
        char_pad_id = char_vocab_index.lookup(tf.constant(char_pad))
        char_dataset = dataset.map(lambda sent: generate_char_feat(sent, char_vocab_index,
            word_max_length, char_max_length, word_sos, word_eos, char_pad_id))
    
    return word_dataset, subword_dataset, char_dataset

def create_output_dataset(output_file,
                          word_vocab_index,
                          word_max_length,
                          word_sos,
                          word_eos,
                          word_feat_enable):
    """create word-level dataset for output data"""
    dataset = tf.data.TextLineDataset([output_file])
    
    word_dataset = None
    if word_feat_enable == True:
        word_dataset = dataset.map(lambda sent: generate_word_feat(sent, word_vocab_index,
            word_max_length, word_sos, word_eos))
    
    return word_dataset

def generate_word_feat(sentence,
                       word_vocab_index,
                       word_max_length,
                       word_sos,
                       word_eos):
    """process words for sentence"""
    words = tf.string_split([sentence], delimiter=' ').values
    words = tf.concat([[word_sos], words[:word_max_length], [word_eos]], 0)
    words = word_vocab_index.lookup(words)
    
    return words
                          
                         
def generate_subword_feat(sentence,
                          subword_vocab_index,
                          word_max_length,
                          subword_max_length,
                          subword_size,
                          word_sos,
                          word_eos,
                          subword_pad_id):
    """generate char feature for sentence"""
    def word_to_subword(word):
        """process subwords for word"""
        word_len = tf.size(tf.string_split([word], delimiter=''))
        subwords = tf.substr([word], 0, subword_size)
        for i in range(1, subword_max_length):
            subwords = tf.cond(i+subword_size-1 < word_len,
                lambda: tf.concat([subwords, tf.substr([word], i, subword_size)], 0),
                lambda: subwords)
        
        subwords = subwords[:subword_max_length]
        subwords = subword_vocab_index.lookup(subwords)
        padding = tf.constant([[0, subword_max_length]])
        subwords = tf.pad(subwords, padding, "CONSTANT", constant_values=subword_pad_id)
        subwords = subwords[:subword_max_length]
        
        return subwords
    
    """process words for sentence"""
    words = tf.string_split([sentence], delimiter=' ').values
    words = tf.concat([[word_sos], words[:word_max_length], [word_eos]], 0)
    word_subwords = tf.map_fn(word_to_subword, words, dtype=tf.int64)
    
    return word_subwords

def generate_char_feat(sentence,
                       char_vocab_index,
                       word_max_length,
                       char_max_length,
                       word_sos,
                       word_eos,
                       char_pad_id):
    """generate char feature for sentence"""
    def word_to_char(word):
        """process characters for word"""
        chars = tf.string_split([word], delimiter='').values
        chars = chars[:char_max_length]
        chars = char_vocab_index.lookup(chars)
        padding = tf.constant([[0, char_max_length]])
        chars = tf.pad(chars, padding, "CONSTANT", constant_values=char_pad_id)
        chars = chars[:char_max_length]
        
        return chars
    
    """process words for sentence"""
    words = tf.string_split([sentence], delimiter=' ').values
    words = tf.concat([[word_sos], words[:word_max_length], [word_eos]], 0)
    word_chars = tf.map_fn(word_to_char, words, dtype=tf.int64)
    
    return word_chars

def create_embedding_file(embedding_file,
                          embedding_table):
    """create embedding file based on embedding table"""
    embedding_dir = os.path.dirname(embedding_file)
    if not tf.gfile.Exists(embedding_dir):
        tf.gfile.MakeDirs(embedding_dir)
    
    if not tf.gfile.Exists(embedding_file):
        with codecs.getwriter("utf-8")(tf.gfile.GFile(embedding_file, "w")) as file:
            for vocab in embedding_table.keys():
                embed = embedding_table[vocab]
                embed_str = " ".join(map(str, embed))
                file.write("{0} {1}\n".format(vocab, embed_str))

def load_embedding_file(embedding_file,
                        embedding_size,
                        unk,
                        pad,
                        sos,
                        eos):
    """load pre-train embeddings from embedding file"""
    if tf.gfile.Exists(embedding_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(embedding_file, "rb")) as file:
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
            if sos and sos not in embedding:
                embedding[sos] = np.random.rand(embedding_size)
            if eos and eos not in embedding:
                embedding[eos] = np.random.rand(embedding_size)
            
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
    if not tf.gfile.Exists(vocab_dir):
        tf.gfile.MakeDirs(vocab_dir)
    
    if not tf.gfile.Exists(vocab_file):
        with codecs.getwriter("utf-8")(tf.gfile.GFile(vocab_file, "w")) as file:
            for vocab in vocab_table:
                file.write("{0}\n".format(vocab))

def load_vocab_file(vocab_file):
    """load vocab data from vocab file"""
    if tf.gfile.Exists(vocab_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as file:
            vocab = {}
            for line in file:
                items = line.strip().split('\t')
                
                item_size = len(items)
                if item_size > 1:
                    vocab[items[0]] = int(items[1])
                elif item_size > 0:
                    vocab[items[0]] = 1
            
            return vocab
    else:
        raise FileNotFoundError("vocab file not found")

def process_vocab_table(vocab,
                        vocab_size,
                        vocab_lookup,
                        unk,
                        pad,
                        sos,
                        eos):
    """process vocab table"""
    default_vocab = [unk, pad]
    if sos:
        default_vocab.append(sos)
    if eos:
        default_vocab.append(eos)
    
    if unk in vocab:
        del vocab[unk]
    if pad in vocab:
        del vocab[pad]
    if sos and sos in vocab:
        del vocab[sos]
    if eos and eos in vocab:
        del vocab[eos]
    
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

def create_subword_vocab(input_data,
                         subword_size):
    """create subword vocab from input data"""
    def generate_subword(word,
                         subword_size):
        """generate subword for word"""
        subwords = []
        chars = list(word)
        char_length = len(chars)
        for i in range(char_length-subword_size+1):
            subword =  ''.join(chars[i:i+subword_size])
            subwords.append(subword)
        
        return subwords
    
    subword_vocab = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            subwords = generate_subword(word, subword_size)
            for subword in subwords:
                if subword not in subword_vocab:
                    subword_vocab[subword] = 1
                else:
                    subword_vocab[subword] += 1
    
    return subword_vocab

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

def load_input_data(input_file):
    """load input data from input file"""
    input_data = []
    if tf.gfile.Exists(input_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(input_file, "rb")) as file:
            for line in file:
                input_data.append(line.strip())
            input_size = len(input_data)
            
            return input_data, input_size
    else:
        raise FileNotFoundError("input file not found")

def prepare_data(logger,
                 input_file,
                 word_vocab_file,
                 word_vocab_size,
                 word_embed_dim,
                 word_embed_file,
                 full_word_embed_file,
                 word_unk,
                 word_pad,
                 word_sos,
                 word_eos,
                 word_feat_enable,
                 pretrain_word_embed,
                 subword_vocab_file,
                 subword_vocab_size,
                 subword_unk,
                 subword_pad,
                 subword_size,
                 subword_feat_enable,
                 char_vocab_file,
                 char_vocab_size,
                 char_unk,
                 char_pad,
                 char_feat_enable):
    """prepare data for word representation"""
    input_data = None
    if tf.gfile.Exists(input_file):
        logger.log_print("# loading input data from {0}".format(input_file))
        input_data, input_size = load_input_data(input_file)
        logger.log_print("# input data has {0} lines".format(input_size))
    
    word_embed_data = None
    if pretrain_word_embed == True:
        if tf.gfile.Exists(word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(word_embed_file))
            word_embed_data = load_embedding_file(word_embed_file,
                word_embed_dim, word_unk, word_pad, word_sos, word_eos)
        elif tf.gfile.Exists(full_word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(full_word_embed_file))
            word_embed_data = load_embedding_file(full_word_embed_file,
                word_embed_dim, word_unk, word_pad, word_sos, word_eos)
        else:
            raise ValueError("{0} or {1} must be provided".format(word_vocab_file, full_word_embed_file))
        
        word_embed_size = len(word_embed_data) if word_embed_data is not None else 0
        logger.log_print("# word embedding table has {0} words".format(word_embed_size))
    
    word_vocab = None
    word_vocab_table = None
    word_vocab_size = None
    word_vocab_index = None
    word_vocab_inverted_index = None
    if word_feat_enable is True:
        if tf.gfile.Exists(word_vocab_file):
            logger.log_print("# loading word vocab table from {0}".format(word_vocab_file))
            word_vocab = load_vocab_file(word_vocab_file)
            (word_vocab_table, word_vocab_size, word_vocab_index,
                word_vocab_inverted_index) = process_vocab_table(word_vocab,
                word_vocab_size, word_embed_data, word_unk, word_pad, word_sos, word_eos)
        elif input_data is not None:
            logger.log_print("# creating word vocab table from {0}".format(input_file))
            word_vocab = create_word_vocab(input_data)
            (word_vocab_table, word_vocab_size, word_vocab_index,
                word_vocab_inverted_index) = process_vocab_table(word_vocab,
                word_vocab_size, word_embed_data, word_unk, word_pad, word_sos, word_eos)
            logger.log_print("# creating word vocab file {0}".format(word_vocab_file))
            create_vocab_file(word_vocab_file, word_vocab_table)
        else:
            raise ValueError("{0} or {1} must be provided".format(word_vocab_file, input_file))

        logger.log_print("# word vocab table has {0} words".format(word_vocab_size))
    
    subword_vocab = None
    subword_vocab_size = None
    subword_vocab_index = None
    subword_vocab_inverted_index = None
    if subword_feat_enable is True:
        if tf.gfile.Exists(subword_vocab_file):
            logger.log_print("# loading subword vocab table from {0}".format(subword_vocab_file))
            subword_vocab = load_vocab_file(subword_vocab_file)
            (_, subword_vocab_size, subword_vocab_index,
                subword_vocab_inverted_index) = process_vocab_table(subword_vocab,
                subword_vocab_size, None, subword_unk, subword_pad, None, None)
        elif input_data is not None:
            logger.log_print("# creating subword vocab table from {0}".format(input_file))
            subword_vocab = create_subword_vocab(input_data, subword_size)
            (subword_vocab_table, subword_vocab_size, subword_vocab_index,
                subword_vocab_inverted_index) = process_vocab_table(subword_vocab,
                subword_vocab_size, None, subword_unk, subword_pad, None, None)
            logger.log_print("# creating subword vocab file {0}".format(subword_vocab_file))
            create_vocab_file(subword_vocab_file, subword_vocab_table)
        else:
            raise ValueError("{0} or {1} must be provided".format(subword_vocab_file, input_file))

        logger.log_print("# subword vocab table has {0} subwords".format(subword_vocab_size))
    
    char_vocab = None
    char_vocab_size = None
    char_vocab_index = None
    char_vocab_inverted_index = None
    if char_feat_enable is True:
        if tf.gfile.Exists(char_vocab_file):
            logger.log_print("# loading char vocab table from {0}".format(char_vocab_file))
            char_vocab = load_vocab_file(char_vocab_file)
            (_, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab,
                char_vocab_size, None, char_unk, char_pad, None, None)
        elif input_data is not None:
            logger.log_print("# creating char vocab table from {0}".format(input_file))
            char_vocab = create_char_vocab(input_data)
            (char_vocab_table, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab,
                char_vocab_size, None, char_unk, char_pad, None, None)
            logger.log_print("# creating char vocab file {0}".format(char_vocab_file))
            create_vocab_file(char_vocab_file, char_vocab_table)
        else:
            raise ValueError("{0} or {1} must be provided".format(char_vocab_file, input_file))

        logger.log_print("# char vocab table has {0} chars".format(char_vocab_size))
    
    if word_embed_data is not None and word_vocab_table is not None:
        word_embed_data = { k: word_embed_data[k] for k in word_vocab_table if k in word_embed_data }
        logger.log_print("# word embedding table has {0} words after filtering".format(len(word_embed_data)))
        if not tf.gfile.Exists(word_embed_file):
            logger.log_print("# creating word embedding file {0}".format(word_embed_file))
            create_embedding_file(word_embed_file, word_embed_data)
        
        word_embed_data = convert_embedding(word_embed_data)
    
    return (input_data, word_embed_data,
        word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        subword_vocab_size, subword_vocab_index, subword_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index)