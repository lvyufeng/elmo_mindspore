import unittest
import mindspore
import numpy as np
from mindspore import Tensor
from elmo.model import LanguageModel
from elmo.data.vocabulary import Vocabulary, UnicodeCharsVocabulary
from elmo.data.dataset import LMDataset
import json

class TestLanguageModel(unittest.TestCase):
    def test_language_model(self):
        options_file = './elmo_2x1024_128_2048cnn_1xhighway_options.json'
        with open(options_file, 'r') as fin:
            options = json.load(fin)
        train_data = './tests/fixtures/train/data.txt'
        vocab_path = './tests/fixtures/train/vocab.txt'
        max_word_length = options['char_cnn']['max_characters_per_token']
        #vocab = Vocabulary(vocab_path, validata_file=True)
        vocab = UnicodeCharsVocabulary(vocab_path, max_word_length=max_word_length)
        data = LMDataset(train_data, vocab)
        lm = LanguageModel(options=options, training=True)
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
                targets_reverse[i, cur_pos+1: next_pos] = cur_stream[i][0][: how_many-1]
                cur_pos = next_pos
                cur_stream[i][0] = cur_stream[how_many:]
                cur_stream[i][1] = cur_stream[how_many:]
            if no_more_data:
                break  
        assert inputs.shape == (batch_size, num_steps, max_word_length)
        loss = lm(Tensor(inputs[0], mindspore.int32), Tensor(targets[0], mindspore.int32),
                 Tensor(targets_reverse[0], mindspore.int32))
        