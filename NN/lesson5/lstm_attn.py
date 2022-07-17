import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy

from lesson5.data import DatasetSeq2Seq, collate_fn_seq2seq
from lesson5.attn import ScaledDotProductAttention

data_dir = '/raid/home/bgzhestkov/nn_reload2/lesson5/dataset_text.txt'

dataset = DatasetSeq2Seq(data_dir)

#hyper params
input_vocab_len = len(dataset.input_sequnces_vocab)
output_vocab_len = len(dataset.output_sequnces_vocab)

cuda_device = 10
batch_size = 32
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'


class Encoder(nn.Module):
    def __init__(self, vocab_len: int, emb_size: int = 256, hidden_size: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_len, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        emb = self.embed(x)
        encoded, context = self.rnn(emb)

        return encoded, context

class Decoder(nn.Module):
    def __init__(self,
                 vocab_len: int,
                 eos_id: int,
                 emb_size: int = 256,
                 hidden_size: int = 256,
                 ):
        super().__init__()
        self.embed = nn.Embedding(vocab_len, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, vocab_len)
        self.eos_id = eos_id
        self.attn = ScaledDotProductAttention()

    def forward(self, hidden_state, sequence, encoded):
        if self.training:
            emb = self.embed(sequence) # истиная последовательность символом (таргет без еоса) B x
            # hid B x T x V - все хиддены декодера, _ 1 x B x V - последний хидден декодера (ht на схеме)
            hid, _ = self.rnn(emb, hidden_state)
            attn_res, attn_diag = self.attn(hid, encoded, encoded)
            out = self.classifier(torch.cat((hid, attn_res), dim=-1))

            return out
        else:
            predicts = []
            i = 0
            predicted_token = sequence # первый токен bos
            while predicted_token != self.eos_id and i < 20:
                emb = self.embed(predicted_token)
                hidden_state, _ = self.rnn(emb, hidden_state)
                attn_res, attn_diag = self.attn(hidden_state, encoded, encoded)
                out = self.classifier(torch.cat((hidden_state, attn_res), dim=-1))
                predicted_token = torch.argmax(out, dim=-1)
                predicts.append(predicted_token)
                i += 1

            return torch.cat(predicts, dim=1)


class DateNormalizer(nn.Module):
    def __init__(self, input_vocab_len, output_vocab_len, emb_size, hidden_size, eos_id):
        super().__init__()
        self.encoder = Encoder(input_vocab_len, emb_size, hidden_size)
        self.decoder = Decoder(output_vocab_len, eos_id, emb_size, hidden_size)

    def forward(self, x, sequence):
        encoded, context = self.encoder(x)
        out = self.decoder(context, sequence, encoded)

        return out

model = DateNormalizer(input_vocab_len, output_vocab_len, 256, 256, dataset.output_sequnces_vocab['#']).to(device)
#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler


#loss
loss_func = nn.CrossEntropyLoss()
#dataloder
for epoch in range(20):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn_seq2seq,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        input_seq = batch['data'].to(device)
        target = batch['target'].to(device)
        optim.zero_grad()
        predict = model(input_seq, target[:, :-1])

        loss = loss_func(predict.reshape(-1, output_vocab_len), target[:, 1:].reshape(-1))
        loss.backward()
        optim.step()
        if (step % 10 == 0):
            test = torch.tensor([[1, 19, 16, 27, 21, 29, 21, 22, 30, 27, 31, 32, 4, 18, 18, 4, 5, 6, 7, 4, 8, 3, 3, 28, 4, 23, 11,
                      2]]).to(device)
            bos_tensor = torch.tensor([[dataset.output_sequnces_vocab['~']]]).to(device)
            result = [3, 4, 4, 8, 6, 7, 8, 6, 10, 10, 9, 2]
            with torch.no_grad():
                model.eval()
                test_predict = model(test, bos_tensor)
                model.train()
            print(loss, test_predict)
            decode = list(dataset.output_sequnces_vocab.keys())
            out_str = ''
            for i in test_predict.squeeze().cpu().detach().tolist():
                out_str += decode[i]
            print(out_str)

    print('Epoch {} finished'.format(epoch))