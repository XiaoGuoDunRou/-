import numpy as np


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
'''
Relu类有实例变量mask。这个变量mask是由True/False构成的NumPy数
组，它会把正向传播时的输入 x的元素中小于等于0的地方保存为 True，其
他地方（大于0的元素）保存为 False
'''

x = np.array( [[1.0, -0.5], [-2.0, 3.0]] )
print(x)

mask = (x <= 0)
print(mask)