import cv2 as cv
''' 例子图片展示 '''
img1=cv.imread('E:\RuanJian\Python\Python\python_picture\SGD.PNG')
cv.imshow('image',img1)  #图像采用的B G R 存储
cv.waitKey(0)      #等待时间，毫秒级，0表示任意键终止
cv.destroyAllWindows()  #破坏我们创建的所有窗口
''' 例子图片展示结束 '''


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
           params[key] -= self.lr * grads[key]
           #  lr表示learning rate（学习率）