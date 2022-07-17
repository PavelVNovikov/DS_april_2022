import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lesson4.dataset import DatasetSeq, collate_fn

data_dir = '/raid/home/bgzhestkov/nn_reload3/lesson4/'
train_lang = 'en'

dataset = DatasetSeq(data_dir)

#hyper params
vocab_len = len(dataset.word_vocab) + 1
n_classes = len(dataset.target_vocab) + 1
n_chars = len(dataset.char_vocab) + 1
cuda_device = 10
batch_size = 100
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'

ds_item = dataset[156]
#model
decode_words = [k for k in dataset.word_vocab]
print([decode_words[i] for i in ds_item['data']])


#using GRUCell
class POS_predictor(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_dim: int,
                 n_classes: int,
                 ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru_cell = nn.GRUCell(emb_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes, bias=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):  # B x T
        b, t = x.size()
        emb_x = self.emb(x) # B x T x V
        hidden = torch.zeros((b, self.hidden_dim))
        gru_out = []
        for i in range(t):
            hidden = self.gru_cell(emb_x[:, i, :], hidden) # B x Hid
            gru_out.append(hidden.unsqueeze(1)) # B x 1 x Hid
        gru_out = torch.cat(gru_out, dim=1) # B x T x Hid
        pred = self.classifier(torch.dropout(gru_out, 0.1, self.training))

        return pred

#usng GRU

# emb = [
#     [1.2,3.4,1.2],
#     [7.2,6.4,4.7],
#     [2.8,3.4,9.2],
# ]
# seq = [2, 1, 2]
# [[2.8,3.4,9.2], [7.2,6.4,4.7], [2.8,3.4,9.2]]

class POS_predictorV2(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_dim: int,
                 n_classes: int,
                 ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, n_classes, bias=True)

    def forward(self, x): # B x T
        emb_x = self.emb(x)  # B x T x V
        gru_out, _ = self.gru(emb_x)
        pred = self.classifier(torch.dropout(gru_out, 0.1, self.training))

        return pred

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
    torch.save({'model': model.state_dict()}, '/raid/home/bgzhestkov/nn_reload3/epoch_%d.pth.tar' % epoch)

# inference
sequence = [2,36,2,14,4,24]
with torch.no_grad():
    model.eval()
    predict = model(torch.tensor(sequence).unsqueeze(0).to(device)) # 1 x T x N_classes
    labels = pred.argmax(-1)


#example
phrase = 'He ran quickly after the red bus and caught it'
words = phrase.split(' ')
tokens = [dataset.word_vocab[w] for w in words]

start = datetime.datetime.now()
with torch.no_grad():
    model.eval()
    predict = model(torch.tensor(tokens).unsqueeze(0).to(device)) # 1 x T x N_classes
    labels = torch.argmax(predict, dim=-1).squeeze().cpu().detach().tolist()
    end = datetime.datetime.now() - start

target_labels = list(dataset.target_vocab.keys())
print([target_labels[l] for l in labels])