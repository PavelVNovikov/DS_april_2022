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
batch_size = 128
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'

#model

# conv autoencoder
class Encoder(nn.Module):
    # 28*28 -> hidden -> out
    def __init__(self, in_chan, hidden_ch, out_channels):
        super().__init__()
        #conv2d -> maxpool2d -> conv2d -> maxpool2d -> conv2d
        self.conv1 = nn.Conv2d(in_chan, hidden_ch, kernel_size=5, stride=1, padding=2) # 28 x28
        self.pool1 = nn.MaxPool2d(2, 2) # 14 x 14
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1)  # 14 x 14
        self.pool2 = nn.MaxPool2d(2, 2)  # 7 x 7
        self.conv3 = nn.Conv2d(hidden_ch, out_channels, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, x): # -> 7x7
        x = self.activation(self.pool1(self.conv1(x)))
        x = self.activation(self.pool2(self.conv2(x)))
        x = self.activation(self.conv3(x))

        return x

class Decoder(nn.Module):
    #conv2d -> upsampling2d -> conv2d -> upsampling2d -> conv2d
    def __init__(self, in_chan, hidden_ch, out_chan):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, hidden_ch, kernel_size=3, stride=1, padding=1)  # 7 x 7
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)  # 14 x 14
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1)  # 14 x 14
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)  # 28 x 28
        self.conv3 = nn.Conv2d(hidden_ch, out_chan, kernel_size=5, stride=1, padding=2)

        self.activation = nn.ReLU()

    def forward(self, x): # -> 28 x 28
        x = self.activation(self.upsample1(self.conv1(x)))
        x = self.activation(self.upsample2(self.conv2(x)))
        x = self.activation(self.conv3(x))

        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_ch, enc_hidden_ch, dec_hidden_ch, latent_ch):
        super().__init__()
        self.encoder = Encoder(input_ch, enc_hidden_ch, latent_ch)
        self.decoder = Decoder(latent_ch, dec_hidden_ch, input_ch)

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
model = AutoEncoder(1, 200, 190, 1)
model.train()
model.to(device)
# result = model(test_tersor)

#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler

#dataset
dataset = datasets.MNIST('/Users/a14419009/Repos/NN_reload_stream2', download=False)

#
# import matplotlib.pyplot as plt
# plt.imshow(dataset.data[0].detach().numpy())
# plt.show()
#loss
loss_func = nn.MSELoss()
#dataloder

for epoch in range(2):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        data = batch['data'].to(device).unsqueeze(1)
        data_noized = torch.clamp(data + 0.5 * torch.normal(torch.zeros_like(data), torch.ones_like(data)), 0., 1.)
        optim.zero_grad()
        predict = model(data_noized)
        loss = loss_func(predict, data)
        loss.backward()
        optim.step()
        if (step % 100 == 0):
            print(loss)
    print(f'epoch: {epoch}')


test = dataset.data[784].unsqueeze(0).unsqueeze(0).float() / 255
test = torch.clamp(test + 0.5 * torch.normal(torch.zeros_like(test), torch.ones_like(test)), 0., 1.0)
predict = model(test)

plt.imshow(test[0][0].view(28, 28).detach().numpy())
plt.show()

plt.imshow(predict[0][0].detach().numpy())
plt.show()
