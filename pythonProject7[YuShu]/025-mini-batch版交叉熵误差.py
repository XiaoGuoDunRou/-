import numpy as np



'''
y是神经网络的输出， t是监督数据。 y的维度为1时，即求单个
数据的交叉熵误差时，需要改变数据的形状
'''
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size



'''
监督数据是标签形式（非one-hot表示，而是像“2”“7”这样的
标签）时，交叉熵误差可通过如下代码实现
'''
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

