import json
import mindspore
import mindspore.ops as P
import mindspore.nn as nn
from mindspore import Tensor, Model
from elmo.data.reader import create_elmo_dataset
from elmo.model import LanguageModel
from elmo.data.vocabulary import Vocabulary, UnicodeCharsVocabulary
from elmo.data.dataset import LMDataset, BidirectionalLMDataset
from ElmoTrainOne import ElmoTrainOnestepWithLoss

def get_data(options, train_path, vocab_path):
    max_word_length = options['char_cnn']['max_characters_per_token']
    batch_size = options['batch_size']
    steps = options['unroll_steps']
    if max_word_length != None:
        vocab = UnicodeCharsVocabulary(vocab_path, max_word_length=max_word_length)
    else:
        vocab = Vocabulary(vocab)
    data = BidirectionalLMDataset(train_path, vocab)
    data_gen = data.iter_batches(batch_size, steps)
    return data_gen

def train():
    options_file = 'tests/fixtures/model/options.json'
    train_data = './tests/fixtures/train/data.txt'
    vocab_path = './tests/fixtures/train/vocab.txt'
    with open(options_file, 'r') as fin:
        options = json.load(fin)
    lr = 0.2
    epoch = 2

    lm = LanguageModel(options=options, training=True)
    opt = nn.Adagrad(lm.trainable_params(), learning_rate=lr)

    data = get_data(options, train_data, vocab_path)
    dataset = create_elmo_dataset(batch_size=options['batch_size'], data_file_path='tests/fixtures/train.mindrecord')
    
    train_one_step = ElmoTrainOnestepWithLoss(lm, opt)
    model = Model(train_one_step)
    model.train(epoch, dataset)
train()      