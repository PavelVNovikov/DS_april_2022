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
input_ch = 1
hidden_ch = 256
out_d = 10

#model
class MyModelCNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 n_classes: int,
                 ):
        super().__init__()
        # TODO change architecture
        # TODO use pooling
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2, stride=2) # 14 * 14
        # TODO add batchnorm after each conv
        # self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0, stride=1)
        self.linear1 = nn.Linear(14*14, n_classes, bias=True)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.linear1(x.view(x.size(0), -1)))

        return x


def collate_fn(data):
    pics = []
    targets = []
    # data = [(pic, target), ....]
    for item in data:
        pics.append(numpy.array(item[0]))
        targets.append(item[1])

    return {
        'data': torch.from_numpy(numpy.array(pics)) / 255,
        'target': torch.from_numpy(numpy.array(targets))
    }


# init model
model = MyModelCNN(input_ch, hidden_ch, out_d)
model = model.to(device)
model.train()
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
    for i, batch in enumerate(data_loader):
        optim.zero_grad()
        data = batch['data'].to(device).float()
        predict = model(data.unsqueeze(1))
        loss = criterion(predict, batch['target'].to(device))
        loss.backward()
        optim.step()
        if i % 100:
            print(loss)

# # SAVING MODEL
# torch.save(model.state_dict(), '<path to save>')
# sd = torch.load('<path to save>', map_location='cpu')
# model.cpu().load_state_dict(sd)
# #to device after load weights on cpu
# model.to(device)

# # inference
# with torch.no_grad():
#     predict = model(data)

#
# tensor(0.2040, grad_fn=<NllLossBackward>)
# tensor(0.2921, grad_fn=<NllLossBackward>)
# tensor(0.2963, grad_fn=<NllLossBackward>)
# tensor(0.2437, grad_fn=<NllLossBackward>)