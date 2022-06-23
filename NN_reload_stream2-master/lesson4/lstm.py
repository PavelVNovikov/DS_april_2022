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
            #'char': torch.tensor(self.encoded_char_sequences[index]),# [[1,2,3], [4,5], [1,2], [2,6,5,4], []] len=5
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

class POS_predictor(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_dim: int,
                 n_classes: int,
                 ):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim)

        self.hidden_dim = hidden_dim
        self.rnncell = nn.GRUCell(input_size=emb_dim, hidden_size=hidden_dim)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x): # B x T
        x = self.word_emb(x)  # B x T x V
        hidden_state = torch.zeros((x.size(0), self.hidden_dim)).to(x.device)
        x = x.transpose(0, 1)   # T x B x V
        rnn_out = []
        for i in range(x.size(0)):
            inp = x[i] # B x V
            hidden_state = self.rnncell(inp, hidden_state)
            rnn_out.append(hidden_state)
        rnn_out = torch.stack(rnn_out)   # T x B x Hid
        rnn_out = rnn_out.transpose(0, 1)

        out = self.classifier(rnn_out)

        return out


class POS_predictorV2(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_dim: int,
                 n_classes: int,
                 ):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim)

        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x): # B x T
        x = self.word_emb(x)  # B x T x V
        rnn_out, _ = self.rnn(x)
        out = self.classifier(rnn_out)

        return out


def collate_fn(batch):
    data = []
    target = []
    for item in batch:
        data.append(torch.as_tensor(item['data']))
        target.append(torch.as_tensor(item['target']))
    data = pad_sequence(data, batch_first=True, padding_value=0)
    target = pad_sequence(target, batch_first=True, padding_value=0)
    return {'data': data, 'target': target}


model = POS_predictorV2(vocab_len, 200, 256, n_classes)
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
        optim.zero_grad()
        data = batch['data'].to(device)  # B x T
        pred = model(data)
        loss = loss_func(pred.view(-1, n_classes), batch['target'].view(-1).to(device))
        loss.backward()
        # if step % 5:
        optim.step()

        if step % 50:
            print(loss)
    print(epoch)
    torch.save({'model': model.state_dict()}, '/raid/home/bgzhestkov/nn_reload2/epoch_%d.pth.tar' % epoch)