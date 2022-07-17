import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy
import matplotlib.pyplot as plt


#hyper params
num_epoch = 20
cuda_device = -1
batch_size = 128
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'

#model
#linear encoder


class Encoder(nn.Module):
    # 28*28 -> hidden -> out
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.activation(self.linear1(x)))
        x = self.dropout2(self.activation(self.linear2(x)))
        x = self.activation(self.linear3(x))

        return x


class Decoder(nn.Module):
    # encoder_out -> hidden -> 28*28
    def __init__(self, latent_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.activation(self.linear1(x)))
        x = self.dropout2(self.activation(self.linear2(x)))
        x = self.activation(self.linear3(x))

        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, dec_hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, enc_hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, dec_hidden_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def collate_fn(data):
    pics = []
    target = []
    for item in data:

        pics.append(numpy.array(item[0]))
        target.append(item[1])
    return {
        'data': torch.from_numpy(numpy.array(pics)).float() / 255,
        'target': torch.from_numpy(numpy.array(target)),
    }


# model
model = AutoEncoder(28*28, 300, 280, 64)
model.train()
model.to(device)
# result = model(test_tersor)

#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler

#dataset
dataset = datasets.MNIST('/Users/a14419009/Repos/NN_reload_stream2', download=False)


#loss
loss_func = nn.MSELoss()

#dataloder
for epoch in range(5):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        data = batch['data'].to(device).view(batch['data'].size(0), -1)
        optim.zero_grad()
        predict = model(data)
        loss = loss_func(predict, data)
        loss.backward()
        optim.step()
        if (step % 100 == 0):
            print(loss)
    print(f'epoch: {epoch}')

test = dataset.data[784].view(1, -1).float() / 255
predict = model(test)

plt.imshow(test[0].view(28, 28).detach().numpy())
plt.show()

plt.imshow(predict[0].view(28, 28).detach().numpy())
plt.show()
