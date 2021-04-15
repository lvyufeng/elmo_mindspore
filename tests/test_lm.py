import unittest
import json
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as P
from elmo.model import LanguageModel
from elmo.data.vocabulary import Vocabulary, UnicodeCharsVocabulary
from elmo.data.dataset import LMDataset, BidirectionalLMDataset
from elmo.modules.embedding import CharacterEncoder
from elmo.nn.rnn_cells import LSTMCell
from elmo.ops.sampled_softmax_loss import SampledSoftmaxLoss

from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
def get_data():
    options_file = 'tests/fixtures/model/options.json'
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    train_data = './tests/fixtures/train/data.txt'
    vocab_path = './tests/fixtures/train/vocab.txt'
    max_word_length = options['char_cnn']['max_characters_per_token']
    #vocab = Vocabulary(vocab_path, validata_file=True)
    vocab = UnicodeCharsVocabulary(vocab_path, max_word_length=max_word_length)
    data = LMDataset(train_data, vocab)
    batch_size = options['batch_size']
    cur_stream = [None] * batch_size
    no_more_data = False
    num_steps = 20
    inputs = np.zeros([batch_size, num_steps, max_word_length], np.int32)
    targets = np.zeros([batch_size, num_steps], np.int32)
    targets_reverse = np.zeros([batch_size, num_steps], np.int32)
    #[batch_size, num_steps]
    for i in range(batch_size):
        cur_pos = 0
        while cur_pos < num_steps:
            if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                try:
                    cur_stream[i] = list(next(data.get_sentence()))
                except StopIteration:   
                    no_more_data=True
                    break
            
            how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
            assert how_many > 0
            next_pos = cur_pos + how_many
            inputs[i, cur_pos: next_pos] = cur_stream[i][1][:how_many]
            targets[i, cur_pos: next_pos] = cur_stream[i][0][1: how_many+1]
            if cur_pos > 2:
                targets_reverse[i, cur_pos] = targets[i, cur_pos-2]
            targets_reverse[i, cur_pos+1: next_pos] = cur_stream[i][0][: how_many-1]
            
            cur_pos = next_pos
            cur_stream[i][0] = cur_stream[how_many:]
            cur_stream[i][1] = cur_stream[how_many:]
        if no_more_data:
            break  
    return inputs, targets, targets_reverse

class TestLanguageModel(unittest.TestCase):
    
    def test_language_model_with_batch(self):
        options_file = 'tests/fixtures/model/options.json'
        with open(options_file, 'r') as fin:
            options = json.load(fin)
        batch_size = options['batch_size']
        lm = LanguageModel(options=options, training=True)
        max_word_length = options['char_cnn']['max_characters_per_token']
        train_data = './tests/fixtures/train/data.txt'
        vocab_path = './tests/fixtures/train/vocab.txt'
        vocab = UnicodeCharsVocabulary(vocab_path, max_word_length=max_word_length)

        data =BidirectionalLMDataset(train_data, vocab)

        for i, batch in enumerate(data.iter_batches(batch_size, 3)):
            X = batch
            inputs = Tensor(batch["tokens_characters"], mindspore.int32)
            inputs_backward = Tensor(batch["tokens_characters_reverse"], mindspore.int32)
            targets = Tensor(batch["next_token_id"], mindspore.int32)
            targets_back = Tensor(batch["next_token_id_reverse"], mindspore.int32)
            print(targets, targets_back)
            loss = lm(inputs, inputs_backward, targets, targets_back)
            if i==3:
                break