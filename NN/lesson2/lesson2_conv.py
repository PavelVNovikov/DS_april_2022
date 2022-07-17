import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy
import matplotlib.pyplot as plt



#hyper params
num_epoch = 1
cuda_device = -1
batch_size = 140
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'
input_d = 28*28
hidden_d = 512
out_d = 10

class Encoder(nn.Module):
    # #conv2d -> maxpool2d -> conv2d -> maxpool2d -> conv2d
    def __init__(self, in_chan, hidden_chan):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, hidden_chan, kernel_size=5, stride=1, padding=2)  # 28*28
        self.pool1 = nn.MaxPool2d(2, 2)  # 14*14
        self.conv2 = nn.Conv2d(hidden_chan, hidden_chan, kernel_size=3, stride=1, padding=1)  # 14*14
        self.pool2 = nn.MaxPool2d(2, 2)  # 7*7
        self.conv3 = nn.Conv2d(hidden_chan, 1, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, x):  # 7*7
        x = self.activation(self.pool1(self.conv1(x)))
        x = self.activation(self.pool2(self.conv2(x)))
        x = self.activation(self.conv3(x))

        return x


class Decoder(nn.Module):
    # conv2d -> upsampling2d -> conv2d -> upsampling2d -> conv2d
    def __init__(self, in_chan, hidden_chan):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_chan, kernel_size=5, stride=1, padding=2)  # 7*7
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)  # > 14 x 14
        self.conv2 = nn.Conv2d(hidden_chan, hidden_chan, kernel_size=3, stride=1, padding=1)  # > 14 x 14
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)  # 28 x 28
        self.conv3 = nn.Conv2d(hidden_chan, in_chan, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, x):  # 28*28
        x = self.activation(self.upsample1(self.conv1(x)))
        x = self.activation(self.upsample2(self.conv2(x)))
        x = self.activation(self.conv3(x))

        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(input_dim, hidden_dim)

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
model = AutoEncoder(1, 50)
model.train()
model.to(device)

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# dataset
dataset = datasets.MNIST('/Users/a14419009/Repos/NN_reload_stream2', download=False)

# loss
loss_func = nn.MSELoss()

# dataloder
for epoch in range(1):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    print(f'epoch: {epoch + 1}')
    for step, batch in enumerate(dataloader):
        data = batch['data'].to(device).unsqueeze(1)
        optim.zero_grad()
        predict = model(data)
        loss = loss_func(predict, data)
        loss.backward()
        optim.step()
        if (step % 100 == 0):
            print(loss)

test = dataset.data[100].unsqueeze(0).unsqueeze(0).float() / 255
predict = model(test)

plt.imshow(test[0][0].view(28, 28).detach().numpy())
plt.show()

plt.imshow(predict[0][0].squeeze().detach().numpy())
plt.show()# tensor(0.2437, grad_fn=<NllLossBackward>)