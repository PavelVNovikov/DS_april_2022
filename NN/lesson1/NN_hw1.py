# Написать на PyTorch forward и backward полносвязного слоя без использования autograd
# Написать 1-2 адаптивных оптимизатора
# Решить задачу нахождения корней квадратного уравнения методом градиентного спуска

import torch
import torch.nn as nn


tensor = torch.tensor([1, 2, 3, 4, 5]).float()
wieghts = nn.Parameter(torch.rand(5), requires_grad=True)

print(tensor)
print(wieghts)
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
        self.l1 = nn.Linear(5,10,bias=True)
        self.l2 = nn.Linear(10,10,bias=True)
        self.l3 = nn.Linear(10,5,bias=True)

    def forward(self, x):
        x = nn.ReLU(self.l1(x))
        x = nn.ReLU(self.l2(x))
        x = self.l3(x)
        return x

    def backward(self, grad):
        pass

net = CustomLinear()
print(net)

# seq = nn.Sequential(
#     {
#         'linear1': nn.Linear(5, 1, bias=True),
#         'relu1': nn.ReLU(),
#         'linear2': nn.Linear(5, 1, bias=True),
#         'relu2': nn.ReLU(),
#         'linear3': nn.Linear(5, 1, bias=True),
#         'relu3': nn.ReLU(),
#     }
# )

# predict = seq(tensor)
# loss = torch.mean((target - predict) ** 2)
# loss.backward()

