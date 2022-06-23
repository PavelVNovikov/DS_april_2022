import torch
import torch.nn as nn


tensor = torch.tensor([1, 2, 3, 4, 5]).float()
wieghts = nn.Parameter(torch.rand(5), requires_grad=True)

# tensor = tensor.detach()
# tensor.detach_()

linear = wieghts @ tensor.t()

activation = nn.ReLU()

activated_linear = activation(linear) # torch.relu(linear)

linear = nn.Linear(5, 1, bias=True)
linear(tensor)

#TODO реализовать forward и backward
class CustomLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

seq = nn.Sequential(
    {
        'linear1': nn.Linear(5, 1, bias=True),
        'relu1': nn.ReLU(),
        'linear2': nn.Linear(5, 1, bias=True),
        'relu2': nn.ReLU(),
        'linear3': nn.Linear(5, 1, bias=True),
        'relu3': nn.ReLU(),
    }
)

predict = seq(tensor)
loss = torch.mean((target - predict) ** 2)
loss.backward()

