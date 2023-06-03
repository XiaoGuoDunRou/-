import numpy as np
import matplotlib.pyplot as plt

# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int64)

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)

# 阶跃函数
y1 = step_function(x)
plt.plot(x, y1, label="阶跃函数")

# sigmoid
y2 = sigmoid(x)
plt.plot(x, y2, linestyle = "--", label="sigmoid函数")

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.xlabel("x") # x轴标签
plt.ylabel("y") # y轴标签
plt.title('阶跃函数 & sigmoid函数') # 标题
plt.legend()  # 图例显示
plt.show()
