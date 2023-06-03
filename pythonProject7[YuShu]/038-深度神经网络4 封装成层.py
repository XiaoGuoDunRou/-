from collections import OrderedDict

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np



# 网页例子，网页地址： https://blog.csdn.net/weixin_49374896/article/details/123227406?ops_request_misc=&request_id=&biz_id=102&utm_term=class%20TwoLayerNet:&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-123227406.142^v88^insert_down28v1,239^v2^insert_chatgpt&spm=1018.2226.3001.4187
# 1.简单的层
# 层的实现中有两个共通的方法（接口）forward()和backward()。forward()对应正向传播，backward()对应反向传播。
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):                # 正向传播
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):               # 反向传播
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy



# 买苹果的简单例子来看一下如何使用层   从此处开始

''' 例子图片展示 '''
img1=cv.imread('E:\RuanJian\Python\Python\python_picture/apple.png',1)
cv.imshow('image',img1)  #图像采用的B G R 存储
cv.waitKey(0)      #等待时间，毫秒级，0表示任意键终止
cv.destroyAllWindows()  #破坏我们创建的所有窗口
''' 例子图片展示结束 '''
''' 
使用层的方法是：
初始化层
按照正向传播顺序一次调用所有层的.forward方法
按照反向传播顺序一次调用所有层的.backward方法，并获取梯度
'''

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)
# 买苹果的简单例子来看一下如何使用层   从此处结束


# 加法层的实现 开始
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
# 加法层的实现 结束


# 2.更进一步 神经网络常用层的实现
# 神经网络中不会单独使用那么简单的乘法层或加法层。而把它们两者合在一起当成一个层就是神经网络中最基础的Affine层啦。
''' 例子图片展示 '''
img2=cv.imread('E:\RuanJian\Python\Python\python_picture/Affine1.png',1)
img3=cv.imread('E:\RuanJian\Python\Python\python_picture/Affine2.png',1)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img2[:,:,::-1])
axes[0].set_title("仿射变换")
axes[1].imshow(img3[:,:,::-1])
axes[1].set_title("实现了批处理的Affine层")
plt.show()

''' 例子图片展示结束 '''




# 激活函数层的实现Sigmoid层
class Sigmoid:
    def __init__(self):
        self.out = None

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        out = self.sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

# ReLU层, ReLU也是一个常用的激活函数。
'''Relu实现的关键是保存记录“值小于等于零的输入的坐标”：正向传播时，把这些位置变为一，其余不变；反向传播时，把这些位置变为零，其余不变。'''
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # 记录值小于等于零的输入的坐标
        self.mask = (x <= 0)
        out = x.copy()
        # 正向传播时，把这些位置变为一，其余不变
        out[self.mask] = 0

        return out

    def backward(self, dout):
        # 反向传播时，把这些位置变为零，其余不变
        dout[self.mask] = 0
        dx = dout

        return dx

# SoftmaxWithLoss层
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def softmax(x):
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x

        return y

    def cross_entropy_error(y, t):
        delta = 1e-7
        return -np.sum(t * np.log(y + delta))

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:                           # 监督数据是标签形式的情况
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
# 仿射变换
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
# 3.使用层组装神经网络
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()    # ”有顺序的字典“，它在提供像字典一样的键值对存储访问的同时，还提供了顺序访问能力。
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):

        # !!!!!!!!!!!!!!!!
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)

        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        # 反转list
        layers.reverse()
        # !!!!!!!!!!!!!!!!!!!!!!!!
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


