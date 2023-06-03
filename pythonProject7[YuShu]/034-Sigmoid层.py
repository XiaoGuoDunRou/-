import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):       # 前向传播
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):   # 反向传播
        dx = dout * (1.0 - self.out) * self.out
        return dx
