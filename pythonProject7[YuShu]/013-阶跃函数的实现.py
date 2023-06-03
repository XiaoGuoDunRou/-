import numpy as np

# 简单地实现阶跃函数
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

# 改为支持NumPy数组的实现
def step_function(x):
    y = x > 0
    return y.astype(np.int)
'''
astype()方法转换NumPy数组的类型。 astype()方
法通过参数指定期望的类型，这个例子中是 np.int型。 Python中将布尔型
转换为 int型后， True会转换为1， False会转换为0。以上就是阶跃函数的
实现中所用到的NumPy的“技巧”
'''
