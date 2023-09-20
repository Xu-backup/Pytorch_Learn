import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset  #引入Dataset类
from torchvision import transforms
from torch.utils.data import DataLoader       #用来读取定义好的数据集
import torch.optim as optim
import scipy.io as scio
from tqdm import tqdm
from Unet import Unet
from cal_loss import Myloss
import math
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, root_dir, Gnum, transform):
        super().__init__()
        self.root_dir = root_dir
        self.Gnum = Gnum
        self.imgs_path = os.listdir(self.root_dir)
        self.transform = transform
        if(Gnum<len(self.imgs_path)):
            self.imgs_path = os.listdir(self.root_dir)[0:Gnum]
        
    def __getitem__(self, index):
        self.path = os.path.join(self.root_dir, self.imgs_path[index])
        input = scio.loadmat(self.path)['Estimate']
        output = scio.loadmat(self.path)['input']
        input = self.transform(input).to(torch.float32)
        output = self.transform(output).to(torch.float32)
        return input ,output
    def __len__(self):
        
        return len(self.imgs_path)

def train_model(model, device, train_loader, optimizer, epoch, mask, loss_list):
    #模型训练
    train_loss = 0.0
    #增加进度条
    closs = Myloss().to(device)
    train_bar = tqdm(train_loader)
    model.train()
    for batch_index, (data, target) in enumerate(train_bar):  #data为一个batch
        #部署到device上
        data, target = data.to(device), target.to(device)
        #初始化梯度
        optimizer.zero_grad()
        #训练后的结果
        output = model(data)
        #计算损失，多分类问题的损失（mse计算） (8,8,128,128)位像素MSE之和
        loss = F.mse_loss(output, target, reduction='sum').cuda()
        
        #loss = closs(output, target, mask)
        train_loss += (loss.item()/len(train_loader.dataset))
        assert math.isnan(train_loss) != True 
        #反向传播（将梯度返回model，优化器利用梯度优化参数）
        loss.backward()
        #参数优化(根据模型中的梯度，进行调优)
        optimizer.step()
        #进度条描述
        train_bar.set_description(f"Epoch {epoch}:")
    if len(loss_list) == 0 or train_loss < min(loss_list):
        torch.save(model.state_dict(),"PyTorch\\DeepCUP\\UNet_flying_Best_train.pth")
    #train_loss = train_loss/len(train_loader.dataset)
    print(f"Train Average Loss:{train_loss:.4f}")
    loss_list.append(train_loss)

def test_model(model, device, test_loader, mask, loss_list):
    #模型验证
    model.eval()

    #测试损失
    test_loss = 0.0
    closs = Myloss().to(device)

    with torch.no_grad(): #不计算梯度，不反向传播
        for data, target in test_loader:
            #部署到device
            data, target = data.to(device), target.to(device)
            #data, target = data.cuda(), target.cuda() 数据部署到GPU

            #测试数据
            output = model(data)
            #计算损失
            loss = F.mse_loss(output, target,reduction="sum").cuda()
            #loss = closs(output, target, mask)
            test_loss += (loss.item()/len(test_loader.dataset))
    
    
    if len(loss_list) == 0 or test_loss < min(loss_list):
        torch.save(model.state_dict(),"PyTorch\\DeepCUP\\UNet_flying_Best_test.pth")
    print(f"Test Average Loss:{test_loss:.4f}")
    loss_list.append(test_loss)

def make_dataset(root_dir, Gnum, test_scale, transform):
    dataset = MyDataset(root_dir,Gnum,transform)
    l = len(dataset)
    test_size = int(l*test_scale)
    train_size = l - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset



def main():
    root_dir = 'PyTorch\\Vedio_Process\\Result_flying\\'
    Gnum = 10000
    Batch_Size = 8
    percent = 0.1
    ##划分数据集
    pipeline = transforms.Compose([
        transforms.ToTensor(),  #将图片转换为tensor
        #transforms.Normalize((0.5,),(0.5,)) #正则化降低模型复杂度，解决过拟合
    ])
    train_dataset, test_dataset = make_dataset(root_dir, Gnum, percent, transform=pipeline)

    train_loader = DataLoader(train_dataset,batch_size=Batch_Size,shuffle=True,drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=Batch_Size,shuffle=True,drop_last=True, num_workers=4, pin_memory=True)

    epoch = 30
    in_h = 135
    in_w = 128
    #Device = torch.device("cpu")
    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #选择训练的设备是GPU还是CPU。
    loss_list = []
    loss2_list = []
    
    model = Unet(in_h, in_w, Batch_Size).to(Device)
    #迁移学习
    #model = model.to(Device)
    #model.load_state_dict(torch.load("PyTorch\\DeepCUP\\UNet.pth",map_location=Device)) 
    

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1) ##调节模型参数的优化器
    #params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params_to_optimize, lr=0.1)

    mask = scio.loadmat('PyTorch/Vedio_Process/mat/mask.mat')
    mask = mask["f"]
 
    for i in range(epoch):
        train_model(model, Device, train_loader, optimizer, i, mask, loss_list)
        test_model(model, Device, test_loader, mask, loss2_list)

    plt.figure(1,figsize=(12, 6))
    plt.plot(range(1, epoch+1), loss_list, c='red', label="Train")
    plt.plot(range(1, epoch+1), loss2_list, c='blue', label="Test")
    plt.xlabel("epoch", fontdict={'size': 16})
    plt.ylabel("loss", fontdict={'size': 16})
    plt.title("epoch-loss", fontdict={'size': 20})
    plt.show()

if __name__ == '__main__':
    main()