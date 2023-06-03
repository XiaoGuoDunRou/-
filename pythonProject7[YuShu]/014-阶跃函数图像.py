import numpy as np
import matplotlib.pylab as plt

'''
阶跃函数以0为界，输出从0切换为1（或者从1切换为0）。
它的值呈阶梯式变化，所以称为阶跃函数
'''

def step_function(x):
    return np.array(x > 0, dtype=np.int64)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴的范围
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("阶跃函数") # 标题
plt.legend()  # 图例显示
plt.show()