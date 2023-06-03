import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])      # 输入
    w = np.array([0.5, 0.5])    # 权重
    b = -0.7                    # 偏置
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
print(AND(1,2))