import numpy as np

class Vocabulary:
    """
    A token vocabulary. Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.
    """
    def __init__(self, filename, validata_file=False):
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1
        self.load_file(filename)
        if validata_file:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has <S>, </S>, <UNK> tokens")

    def load_file(self, filename):
        with open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

    @property
    def bos(self):
        return self._bos
    
    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk
    
    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, ids):
        return ' '.join([self.id_to_word(cur_id) for cur_id in ids])
        
    def encode(self, sentence, reverse=False, split=True):
        if split:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self.eos] + word_ids + [self.bos], dtype=np.int32)
        else:
            return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class UnicodeCharsVocabulary(Vocabulary):
    def __init__(self, filename, max_word_length, **kwargs):
        super().__init__(filename, **kwargs)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256
        self.eos_char = 257
        self.bow_char = 258
        self.eow_char = 259
        self.pad_char = 260

        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length], dtype=np.int32)

        self.bos_chars = self._make_bos_eos(self.bos_char)
        self.eos_chars = self._make_bos_eos(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars

    def _make_bos_eos(self, c):
        r = np.zeros([self._max_word_length], dtype=np.int32)
        r[:] = self.pad_char
        r[0] = self.bow_char
        r[1] = c
        r[2] = self.eow_char
        return r

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char
        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, char_id in enumerate(word_encoded, start=1):
            code[k] = char_id
        code[len(word_encoded) + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)
    
    def encode_chars(self, sentence, reverse=False, split=True):
        if split:
            chars_ids = [self.word_to_char_ids(cur_word) for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word) for cur_word in sentence]
        
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])