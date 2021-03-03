import glob
import random
from mindspore.log import logging

class LMDataset(object):
    """
    Hold a language model dataset.

    A dataset is a list of tokenized files. Each file contains one sentence per line.
    Each sentence is pre-tokenized and white space jointed
    """
    def __init__(self, filepattern, vocab, reverse=False, test=False, shuffle_on_load=False):
        self._vocab = vocab
        self._all_shards = glob.glob(filepattern)
        logging.info('Found %d shards at %s' % (len(self._all_shards), filepattern))
        self._shards_to_choose = []

        self._reverse = reverse
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._use_char_inputs = hasattr(vocab, 'encode_chars')

        self._ids = self._load_random_shard()

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self._all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_random_shard(self):
        if self._test:
            if len(self._all_shards) == 0:
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            shard_name = self._choose_random_shard()
        
        ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(ids)
        return ids
    
    def _load_shard(self, shard_name):
        logging.info('Loading data from: %s' % shard_name)
        with open(shard_name) as f:
            sentences_raw = f.readlines()

        if self._reverse:
            sentences = []
            for sentence in sentences_raw:
                splitted = sentence.split()
                splitted.reverse()
                sentences.append(' '.join(splitted))
        else:
            sentences = sentences_raw
        
        if self._shuffle_on_load:
            random.shuffle(sentences)
        
        ids = [self.vocab.encode(sentence, self._reverse) for sentence in sentences]
        if self._use_char_inputs:
            chars_ids = [self.vocab.encode_chars(sentence, self._reverse) for sentence in sentences]
        else:
            chars_ids = [None] * len(ids)
        logging.info('Loaded %d sentences.' % len(ids))
        return list(zip(ids, chars_ids))
    
    def get_sentence(self):
        while True:
            if self._i == self._nids:
                self._ids = self._load_random_shard()
            ret = self._ids[self._i]
            self._i += 1
            yield ret

    @property
    def max_word_length(self):
        if self._use_char_inputs:
            return self._vocab.max_word_length
        else:
            return None
    
    def iter_batches(self, batch_size, num_steps):
        pass
    
    @property
    def vocab(self):
        return self._vocab