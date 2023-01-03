
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models #torchvision 提供的现有模型

#sequential方法写网络
class Net_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()  #在这输出可以得到线性层输入
        self.linear1 = nn.Linear(64*4*4, 64)
        self.linear2 = nn.Linear(64, 10)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

##使用sequential写法，效果相同

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = nn.sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        x = self.module1(x)
        return x  



module = Net_CIFAR10()
print(module) #输出网络结构
input = torch.ones((64, 3, 32, 32)) #创建输入
output = module(input)
print(output.shape)#输出的结构

writer = SummaryWriter('logs_Net_CIFA')  #将网络写入tensorboard
writer.add_graph(module, input)

writer.close()
