# this is homework_01
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable

x_tensor = torch.linspace(0, 6*np.pi, 10000)        #创建一个输入数据集，[0-6*pi]
x_data = torch.unsqueeze(x_tensor, dim=1)           #增加维度，将x的形状从[10000]变成二维[10000,1]
y_data = torch.sin(x_data)                          #待拟合的标签值，此处可以添加部分噪声
# plt.figure("查看数据集是否正确")
# plt.scatter(x_data, y_data)
# plt.ion()
# plt.show()
# 定义超参数
D_in = 1                                            #输入维度1
D_out = 1                                           #输出维度1
H1 = 100                                            #中间隐藏层  100个神经元
train_step = 100000                                 #训练次数10e5次
#定义新激活函数swish(x) = x * sigmod(x)，作为一个类
class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()
    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
swish = Act_op()                                    #类的实例化
'''
对于下面的神经网络定义层的类，初始化输入参数为D_in,H1,H2,D_out
H1代表第一个隐藏层的神经元个数
H2代表第二个隐藏层的神经元个数
目前程序中使用的神经网络只有H1一个隐藏层，调试过程中曾经尝试过增加隐藏层的个数
clamp(min=0)是relu激活函数比较简洁的表达方式
swish(x)方法表示 x*sigmod(x)这个激活函数
'''
class TwolayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, D_out):
        super(TwolayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)
    def forward(self, X):
        y_pred = self.linear2(swish(self.linear1(X)))
        return y_pred
net = TwolayerNet(D_in, H1, D_out)              #类的实例化
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
loss_func = torch.nn.MSELoss()                      #优化器为adam，损失loss函数为MESloss
plt.figure("regression")                            #新建一张画布，打印数据点和预测值
plt.ion()                                           #交互模式
if torch.cuda.is_available():                       #将模型搬移到GPU执行
    print("GPU1")
    net = net.cuda()
else:
    print("CPU")

for step in range(train_step):
    if torch.cuda.is_available():                   #将数据点搬移到GUP执行
        inputs = Variable(x_data).cuda()
        target = Variable(y_data).cuda()
    else:
        inputs = Variable(x_data)
        target = Variable(y_data)
    # 调用搭建好的神经网络模型，得到预测值
    prediction = net(inputs)
    # 用定义好的损失函数，得出预测值和真实值的loss
    loss = loss_func(prediction, target)
    # 每次都需要把梯度将为0
    optimizer.zero_grad()
    # 误差反向传递
    loss.backward()
    # 调用优化器进行优化,将参数更新值施加到 net 的 parameters 上
    optimizer.step()
    if step % 100 == 0:
        # 清除当前座标轴
        plt.cla()
        plt.scatter(x_data.data.numpy(), y_data.data.numpy())
        # r- 是红色 lw 是线宽
        plt.plot(x_data.data.numpy(), prediction.data.cpu().numpy(), 'r-', lw=5)
        '''
        给图形添加标签，0.5， 0 表示X轴和Y轴坐标，
        'Loss=%.4f'%loss.data.numpy()表示标注的内容，
        .4f表示保留小数点后四位
        fontdict是设置字体大小和颜色
        '''
        plt.text(0.5, 0, 'Loss=%.4f'%loss.data.cpu().numpy(), fontdict={'size':20, 'color': 'red'})
        # 间隔多久再次进行绘图
        plt.pause(0.1)
    if step % 1000 == 0:
        print(str(step/1000)+'   Loss=%.4f'%loss.data.cpu().numpy())
