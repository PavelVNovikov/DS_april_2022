import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


data_dir = '/raid/home/bgzhestkov/nn_reload2/lesson4/'
train_lang = 'en'


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

dataset = DatasetSeq(data_dir)

#hyper params
vocab_len = len(dataset.word_vocab) + 1
n_classes = len(dataset.target_vocab) + 1
n_chars = len(dataset.char_vocab) + 1
cuda_device = 10
batch_size = 100
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'


#model

class CharModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_dim: int,
                 ):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, emb_dim)

        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x): # B x T
        x = self.char_emb(x)  # B x T x V
        _, out = self.rnn(x)

        return out # B x 1 x V


class POS_predictorV2Chars(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_dim: int,
                 n_classes: int,
                 n_chars: int,
                 char_emb_dim: int,
                 char_hidden_dim: int,
                 ):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim)

        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_size=emb_dim+char_emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.char_model = CharModel(n_chars, char_emb_dim, char_hidden_dim)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, char_seq): # B x T
        x = self.word_emb(x)  # B x T x V
        chars = [self.char_model(item.to(x.device)).squeeze().unsqueeze(1) for item in char_seq] # T x 1
        chars = torch.cat(chars, dim=1)
        rnn_out, hidden, cell_state = self.rnn(torch.cat([x, chars], dim=-1))
        out = self.classifier(rnn_out)

        return out


def collate_fn(input_data):
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


model = POS_predictorV2Chars(vocab_len, 200, 256, n_classes, n_chars, 32, 64)
model.train()
model = model.to(device)
#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler


#loss
loss_func = nn.CrossEntropyLoss()
#dataloder
for epoch in range(20):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        #
        optim.zero_grad()
        data = batch['data'].to(device)  # B x T
        pred = model(data, batch['chars'])
        loss = loss_func(pred.view(-1, n_classes), batch['target'].view(-1).to(device))
        loss.backward()
        # if step % 5:
        optim.step()
        #
        if step % 50:
            print(loss)
    print(epoch)
    torch.save({'model': model.state_dict()}, '/raid/home/bgzhestkov/nn_reload2/epoch_%d.pth.tar' % epoch)

    # tensor(0.0063, device='cuda:10', grad_fn= < NllLossBackward >)
    # tensor(0.0123, device='cuda:10', grad_fn= < NllLossBackward >)
    # tensor(0.0066, device='cuda:10', grad_fn= < NllLossBackward >)