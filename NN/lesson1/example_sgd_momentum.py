# velocity = momentum (0.9-0.99)* velocity - lr*gradient
# w = w + velocity (скорость-вектор)

import torch
from torch.optim.optimizer import Optimizer


class SDGMomentum:
    def __init__(self, model_weights, momentum: float = 0.99, lr: float = 0.001):
        self.momentum = momentum
        self.lr = lr
        self.velocity = torch.zeros_like(model_weights)
        self.model = model_weights

    def step(self, grad):
        self.velocity = self.momentum * self.velocity - self.lr * grad
        self.model += self.velocity


model = torch.rand((5, 5))
optim = SDGMomentum(model)
optim.step(torch.rand((5, 5)) * 0.1)

# tt = t



# x ** 2 - 5 * x + 3 = 0

def grad_f(x):
    return 2 * (x ** 2 - 5 * x + 3) * (2 * x - 5)

def solver(init_x, lr, num_iter: int = 1000, eps: float = 0.0001):
    x = init_x
    for _ in range(num_iter):
        grad = grad_f(x)
        if grad < eps:
            return x
        x -= lr * grad

    return x

print(solver(60,0.1))