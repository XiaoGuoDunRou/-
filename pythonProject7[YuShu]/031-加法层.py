

class AddLayer:
    def __init__(self):
        pass                # pass语句表示“什么也不运行”

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

