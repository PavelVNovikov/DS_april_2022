import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy


#hyper params
num_epoch = 20
cuda_device = -1
batch_size = 140
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'
input_d = 28*28
hidden_d = 512
out_d = 10

#model
class MyModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear3 = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.sigmoid(self.linear3(x))

        return x


def collate_fn(data):
    pics = []
    targets = []
    # data = [(pic, target), ....]
    for item in data:
        pics.append(numpy.array(item[0]))
        targets.append(item[1])

    return {
        'data': torch.from_numpy(numpy.array(pics)) / 255, # нормирование в диапазон [0, 1]
        'target': torch.from_numpy(numpy.array(targets))
    }


# init model
model = MyModel(input_d, hidden_d, out_d)
model = model.to(device)

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
criterion = nn.CrossEntropyLoss()

# train loop
for epoch in range(num_epoch):
    #dataloder
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=collate_fn,
                             drop_last=True,
                             )
    optim.zero_grad()
    for i, batch in enumerate(data_loader):
        # batch['data'] - > B x W x H -> B x W*H
        data = batch['data'].to(device).float()
        predict = model(data.view(data.size(0), -1))
        loss = criterion(predict, batch['target'].to(device))
        loss.backward()
        if i % 2 == 0:
            optim.step()
            optim.zero_grad()
        if i % 100:
            print(loss)
        # сохранение модели каждые n шагов

model.train()
with torch.no_grad():
    model.eval()
    pred = model(data) # B x W*H    1 x W*H
#
# tensor(0.2040, grad_fn=<NllLossBackward>)
# tensor(0.2921, grad_fn=<NllLossBackward>)
# tensor(0.2963, grad_fn=<NllLossBackward>)
# tensor(0.2437, grad_fn=<NllLossBackward>)