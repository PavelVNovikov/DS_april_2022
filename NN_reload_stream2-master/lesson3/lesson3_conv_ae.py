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


class Encoder(nn.Module):
    # 28*28 -> hidden -> out
    def __init__(self, ):
        super().__init__()
        #conv2d -> maxpool2d -> conv2d -> maxpool2d -> conv2d

    def forward(self, x):


class Dencoder(nn.Module):
    #conv2d -> upsampling2d -> conv2d -> upsampling2d -> conv2d
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()


    def forward(self, x):



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

# result = model(test_tersor)

#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler

#dataset
dataset = datasets.MNIST('.', download=False)

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
        data_noized = torch.clamp(data + 0.5*torch.rand_like(data), 0., 1.)
        optim.zero_grad()
        predict = model(data)
        loss = loss_func(predict, data)
        loss.backward()
        optim.step()
        if (step % 50 == 0):
            print(loss)
    print(f'epoch: {epoch}')

# test = dataset.data[65].unsqueeze(0).unsqueeze(0).float() / 255
# predict = model(test)
# import matplotlib.pyplot as plt
# plt.imshow(predict[0][0].detach().numpy())
# plt.show()
#
# plt.imshow(test[0].view(28,28).detach().numpy())
# plt.show()