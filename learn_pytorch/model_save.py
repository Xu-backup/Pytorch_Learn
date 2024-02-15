import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


model_vgg = torchvision.models.vgg16()
#保存模型 方式一
torch.save(model_vgg, "vgg16_method1.pth")

#加载模型 方式一 同样需要恢复网络结构
model = torch.load("vgg16_method1.pth")##结构和参数

#保存 方式二（官方推荐）
torch.save(model_vgg.state_dict(),"vgg16_method2.pth") ##只有参数

#加载 方式二
model2_vgg = torchvision.models.vgg16()         ##先恢复网络
model2_vgg.load_state_dict(torch.load("vgg16_method2.pth"),strict=False)  ##再回复参数