import os
import json
import argparse
from mindspore.mindrecord import FileWriter
from mindspore.log import logging
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from elmo.data.vocabulary import Vocabulary, UnicodeCharsVocabulary
from elmo.data.dataset import LMDataset, BidirectionalLMDataset

class BaseReader(object):
    """"base reader for elmo"""

    def __init__(self, options):
        self.batch_size = options["batch_size"]
        self.num_steps = options["unroll_steps"]
        self.max_word_length = options['char_cnn']['max_characters_per_token']
        self.n_train_token = options.get('n_train_tokens')

    def get_data_batches(self, file_path, vocab_path):
        vocab = UnicodeCharsVocabulary(vocab_path, self.max_word_length)
        data = BidirectionalLMDataset(file_path, vocab)
        return data

    def file_based_convert_examples_to_features(self, input_file, vocab_file, output_file):
        nlp_schema = {
            "tokens_characters":{"type": "int32", "shape":[20, 50]},
            "tokens_characters_reverse":{"type": "int32", "shape":[20, 50]},
            "next_token_id":{"type": "int32", "shape":[-1]},
            "next_token_id_reverse":{"type": "int32", "shape":[-1]}
        }
        
        writer = FileWriter(file_name=output_file, shard_num=1)
        writer.add_schema(nlp_schema, "preprocessed languae model dataset")
        data = []
        n_tokens_per_batch = self.batch_size * self.num_steps
        n_total_batchs = self.n_train_token / n_tokens_per_batch
        data_gen = self.get_data_batches(input_file, vocab_file)
        for batch_no, batch in enumerate(data_gen.iter_batches(self.batch_size, self.num_steps)):
            for i in range(self.batch_size):
                sample = {
                    "tokens_characters": batch['tokens_characters'][i],
                    "tokens_characters_reverse": batch['tokens_characters_reverse'][i],
                    "next_token_id": batch['next_token_id'][i],
                    "next_token_id_reverse": batch['next_token_id_reverse'][i]
                }
                
                data.append(sample)
            if batch_no >= n_total_batchs:
                break
        writer.write_raw_data(data)
        writer.commit()

def create_elmo_dataset(batch_size=1, repeat_count=1, data_file_path=None,
                        schema_file_path=None, do_shuffle=True):
    """create training dataset"""
    data_set = ds.MindDataset([data_file_path], columns_list=["tokens_characters", "tokens_characters_reverse", \
                                "next_token_id", "next_token_id_reverse"], shuffle=do_shuffle)
    columns_list=["tokens_characters", "tokens_characters_reverse", \
                    "next_token_id", "next_token_id_reverse"]
    
    data_set = data_set.batch(batch_size, drop_remainder=False)
    return data_set

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="read dataset and save it to minddata")
    parser.add_argument("--vocab_path", type=str, default="", help="vocab file")
    parser.add_argument("--options_path", type=str, default="", help="options file")
    parser.add_argument("--input_file", type=str, default="", help="raw data file")
    parser.add_argument("--output_file", type=str, default="", help="minddata file")
    args_opt = parser.parse_args()
    with open(args_opt.options_path) as fin:
        options = json.load(fin)
    reader = BaseReader(options)
    reader.file_based_convert_examples_to_features(args_opt.input_file, args_opt.vocab_path,
                                                   args_opt.output_file)