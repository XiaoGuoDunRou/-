import numpy as np

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # np.zeros_like(x)会生成一个形状和x相同、所有元素都为0的数组。

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
    return grad


def function_2(x):
    return x[0]**2 + x[1]**2   # 或者return np.sum(x**2)

numerical_gradient(function_2, np.array([3.0, 4.0]))

numerical_gradient(function_2, np.array([0.0, 2.0]))

numerical_gradient(function_2, np.array([3.0, 0.0]))
