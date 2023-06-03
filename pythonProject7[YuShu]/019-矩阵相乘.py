import numpy as np

'''
它 们 的 乘 积 可 以 通 过 NumPy 的
np.dot()函数计算（乘积也称为点积）。 np.dot()接收两个NumPy数组作为参
数，并返回数组的乘积。这里要注意的是， np.dot(A, B)和 np.dot(B, A)的
值可能不一样。和一般的运算（+或*等）不同，矩阵的乘积运算中，操作数（A、
B）的顺序不同，结果也会不同。
'''
A = np.array([[1,2], [3,4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))

C = np.array([[1,2,3], [4,5,6]])
D = np.array([[1,2], [3,4], [5,6]])
print(np.dot(C, D))
