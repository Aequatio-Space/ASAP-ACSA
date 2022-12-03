import collections
import jieba
import re
import json
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None, dict_path = None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # The index for the padding word is 0
        if dict_path is not None:
            #it can only load full dict.
            f = open(dict_path,'r')
            self._token_freqs = [tuple(line.split()) for line in list(f)]
            self._token_freqs = [(item[0],int(item[-1])) for item in self._token_freqs]
        else:
            # Sort according to frequencies
            counter = count_corpus(tokens)
            self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<pad>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        print(len(self.token_to_idx))
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    @property
    def unk(self):  # Index for the unknown token
        return 0
    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def clean_str(string):
    for ch in '!"#$&*+，、-.:；。 <=>？@[]《》^！_\|·~‘’':
        string = string.replace(ch, "")
    return string
def read_words(data_file):  #@save
    with open(data_file) as f:
        lines = f.readlines()
        lines.pop(0)
        #移除标题
    return [list(jieba.cut(clean_str(re.sub("\t\d\n","",line)))) for line in lines]

# data_file = "train_SENT.tsv"
# tokens = read_words(data_file)
# vocab = Vocab(reserved_tokens=['<unk>'],dict_path="word_freq.txt")
# vocab = Vocab(tokens,reserved_tokens=['<unk>'])
# ASAP_dict = json.dumps(vocab.token_to_idx,sort_keys=False,indent=4,separators=(',',': '))
# #保存
# fileObject = open('idx_to_token_bug.txt', 'w')
# for token in vocab.idx_to_token:
#     fileObject.write(str(token))
#     fileObject.write('\n')
# fileObject.close()
# fileObject = open('token_to_idx.json','w')
# fileObject.write(ASAP_dict)
# fileObject.close()
# fileObject = open('word_freq.txt','w')
# for freq in vocab._token_freqs:
#     fileObject.write(f"{freq[0]} {freq[1]}")
#     fileObject.write('\n')
# fileObject.close()