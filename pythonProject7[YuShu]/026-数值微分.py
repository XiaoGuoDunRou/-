






# 不好的实现示例  函数 numerical_diff(f, x)的名称来源于数值微分
def numerical_diff1(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) / h

'''
数值微分含有误差。为了减小这个误差，我们可以计算
函数f在(x + h)和(x − h)之间的差分
'''
def numerical_diff2(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


# y = 0.01x2 + 0.1x
def function_1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt
x = np.arange(0.0, 20.0, 0.1) # 以0.1为单位，从0到20的数组x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff1(function_1, 5))
print(numerical_diff1(function_1, 10))

print()

print(numerical_diff2(function_1, 5))
print(numerical_diff2(function_1, 10))
