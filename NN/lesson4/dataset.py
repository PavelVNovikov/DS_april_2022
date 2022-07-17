import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class DatasetSeq(Dataset):
    def __init__(self, data_dir, train_lang='en'):

        with open(data_dir + train_lang + '.train', 'r') as f:
            train = f.read().split('\n\n')

        # delete extra tag markup
        train = [x for x in train if not '_ ' in x]

        self.target_vocab = {}
        self.word_vocab = {}
        self.char_vocab = {}

        self.encoded_sequences = []
        self.encoded_targets = []
        self.encoded_char_sequences = []
        n_word = 1
        n_target = 1
        n_char = 1
        for line in train:
            sequence = []
            target = []
            chars = []
            for item in line.split('\n'):
                if item != '':
                    word, label = item.split(' ')

                    if self.word_vocab.get(word) is None:
                        self.word_vocab[word] = n_word
                        n_word += 1
                    if self.target_vocab.get(label) is None:
                        self.target_vocab[label] = n_target
                        n_target += 1
                    for char in word:
                        if self.char_vocab.get(char) is None:
                            self.char_vocab[char] = n_char
                            n_char += 1
                    sequence.append(self.word_vocab[word])
                    target.append(self.target_vocab[label])
                    chars.append([self.char_vocab[char] for char in word])
            self.encoded_sequences.append(sequence)
            self.encoded_targets.append(target)
            self.encoded_char_sequences.append(chars)

    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, index):
        return {
            'data': self.encoded_sequences[index], # [1, 2, 3, 4, 6] len=5
            'char': self.encoded_char_sequences[index],# [[1,2,3], [4,5], [1,2], [2,6,5,4], []] len=5
            'target': self.encoded_targets[index], #  (1)
        }
# seq1 = [1, 2, 3] -> [1, 2, 3, 0]
# seq2 = [7, 5, 4, 2]

def collate_fn(batch):
    data = []
    target = []
    for item in batch:
        data.append(torch.as_tensor(item['data']))
        target.append(torch.as_tensor(item['target']))
    data = pad_sequence(data, batch_first=True, padding_value=0)
    target = pad_sequence(target, batch_first=True, padding_value=0)

    return {'data': data, 'target': target}


def collate_fn_char(input_data):
    data = []
    chars = []
    targets = []
    max_len = 0
    for item in input_data:
        if len(item['data']) > max_len:
            max_len = len(item['data'])
        data.append(torch.as_tensor(item['data']))
        chars.append(item['char'])
        targets.append(torch.as_tensor(item['target']))
    chars_seq = [[torch.as_tensor([0]) for _ in range(len(input_data))] for _ in range(max_len)]
    for j in range(len(input_data)):
        for i in range(max_len):
            if len(chars[j]) > i:
                chars_seq[i][j] = torch.as_tensor(chars[j][i])
    for j in range(max_len):
        chars_seq[j] = pad_sequence(chars_seq[j], batch_first=True, padding_value=0)
    data = pad_sequence(data, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return {'data': data, 'chars': chars_seq, 'target': targets}
