
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


'''model_vgg = torchvision.models.vgg16()
output = model_vgg(data)'''

#普通方法写方法写网络
class Net1(nn.Module):
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



module = Net1()
Device = torch.device("cuda")

#模型部署到GPU
if torch.cuda.is_available():
    #module = module.cuda()效果相同
    module =module.to(Device) 

#损失函数实例化
loss_fn = nn.CrossEntropyLoss()

#将损失函数， 模型, 数据部署到GPU上训练
if torch.cuda.is_available():
    #loss_fn = loss_fn.cuda()效果相同
    loss_fn = loss_fn.to(Device)