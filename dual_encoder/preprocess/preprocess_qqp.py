import argparse
import codecs
import json
import os.path
import string
import re
import uuid
import nltk
import spacy
import numpy

spacy_nlp = spacy.load('en')

def add_arguments(parser):
    parser.add_argument("--format", help="format to generate", required=True)
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)
    parser.add_argument("--process_mode", help="process mode", default="train")
    parser.add_argument("--neg_num", help="number of negative sampling", type=int, default=5)
    parser.add_argument("--random_seed", help="random seed", type=int, default=0)

def nltk_tokenize(text, lower_case=False, remove_punc=False):
    def process_token(tokens):
        special = ("-", "£", "€", "¥", "¢", "₹", "\u2212", "\u2014", "\u2013",
                   "/", "~", '"', "'", "\ud01C", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        pattern = "([{}])".format("".join(special))
        processed_tokens = []
        for token in tokens:
            token = token.replace("''", '" ').replace("``", '" ')
            processed_tokens.extend(re.split(pattern, token))
        
        return processed_tokens
    
    def remove_punctuation(tokens):
        exclude = set(string.punctuation)
        return [token for token in tokens if token not in exclude]
    
    def fix_white_space(tokens):
        return [token for token in tokens if token and not token.isspace()]
    
    sents = nltk.sent_tokenize(text)
    norm_sents = []
    for sent in sents:
        words = nltk.word_tokenize(sent)
        words = process_token(words)
        if remove_punc:
            words = remove_punctuation(words)
        
        words = fix_white_space(words)
        norm_sents.append(' '.join(words))
    
    norm_text = ' '.join(norm_sents)
    if lower_case:
        norm_text = norm_text.lower()
    
    return norm_text

def spacy_tokenize(text, lower_case=False, remove_punc=False):
    def process_token(tokens):
        special = ("-", "£", "€", "¥", "¢", "₹", "\u2212", "\u2014", "\u2013",
                   "/", "~", '"', "'", "\ud01C", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        pattern = "([{}])".format("".join(special))
        processed_tokens = []
        for token in tokens:
            token = token.replace("''", '" ').replace("``", '" ')
            processed_tokens.extend(re.split(pattern, token))
        
        return processed_tokens
    
    def remove_punctuation(tokens):
        exclude = set(string.punctuation)
        return [token for token in tokens if token not in exclude]
    
    def fix_white_space(tokens):
        return [token for token in tokens if token and not token.isspace()]
    
    word_docs = spacy_nlp(text)
    words = [word.text for word in word_docs]
    words = process_token(words)
    if remove_punc:
        words = remove_punctuation(words)
    
    words = fix_white_space(words)
    
    norm_text = ' '.join(words)
    if lower_case:
        norm_text = norm_text.lower()
    
    return norm_text

def preprocess(file_name,
               process_mode,
               neg_num,
               random_seed):
    if not os.path.exists(file_name):
        raise FileNotFoundError("file not found")
    
    data_list = []
    with open(file_name, "rb") as file:
        for line in file:
            items = [item for item in line.decode("utf-8").strip().split('\t') if item]
            if len(items) < 6:
                continue
            
            if (items[0] == "id" and items[1] == "qid1" and items[2] == "qid2" and
                items[3] == "question1" and items[3] == "question2" and items[3] == "is_duplicate"):
                continue
            
            source = spacy_tokenize(items[3].strip())
            target = spacy_tokenize(items[4].strip())
            label = items[5].strip()
            
            if label == "0":
                continue
            
            data_list.append({
                "id": str(uuid.uuid4()),
                "source": source,
                "target": target,
                "label": label
            })
        
        if process_mode == "eval":
            processed_data_list = []
            data_size = len(data_list)
            indice_list = neg_sampling_indice(data_size, neg_num, random_seed)
            for i, indice in enumerate(indice_list):
                for j in indice:
                    processed_data_list.append({
                        "id": str(uuid.uuid4()),
                        "source": data_list[i]["source"],
                        "target": data_list[j]["target"],
                        "label": "1" if i == j else "0"
                    })
        else:
            processed_data_list = data_list
    
    return processed_data_list

def neg_sampling_indice(data_size,
                        neg_num,
                        random_seed):
    """generate indice for negative sampling"""
    numpy.random.seed(random_seed)
    indice_list = []
    for index in range(data_size):
        neg_num = min(data_size-1, neg_num) 
        indice = list(range(data_size))
        indice.remove(index)
        numpy.random.shuffle(indice)
        indice = [index] + indice[:neg_num]
        indice_list.append(indice)

    return indice_list

def output_to_json(data_list, file_name):
    with open(file_name, "w") as file:
        data_json = json.dumps(data_list, indent=4)
        file.write(data_json)

def output_to_plain(data_list, file_name):
    with open(file_name, "wb") as file:
        for data in data_list:
            data_plain = "{0}\t{1}\t{2}\t{3}\r\n".format(data["id"], data["source"], data["target"], data["label"])
            file.write(data_plain.encode("utf-8"))

def main(args):
    processed_data = preprocess(args.input_file, args.process_mode, args.neg_num, args.random_seed)
    if (args.format == 'json'):
        output_to_json(processed_data, args.output_file)
    elif (args.format == 'plain'):
        output_to_plain(processed_data, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)