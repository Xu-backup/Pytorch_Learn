import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import math

##将tensor转为numpy计算TSC
def TSC(imgs_np, mask, inter=1, frame_num=8):
    imgs_np = imgs_np.copy()
    output = np.zeros((128+inter*(frame_num-1),128))
    for i in range(frame_num):
        imgs_np[i,:,:] = imgs_np[i,:,:]*mask
    for i in range(frame_num):                 
        output[i*inter:i*inter+128, 0:128] += imgs_np[i,:,:]
    ymax = np.max(output)
    ymin = np.min(output)
    if(ymax-ymin!=0):
        output = (output-ymin)/(ymax-ymin)
    return output
'''

'''

class Myloss(nn.Module):
    def __init__(self,a=1,b=0.1,c=0.01):
        super().__init__()
        ##定义不同部分的比例
        self.a = a
        self.b = b
        self.c = c
        self.Mse = nn.MSELoss(reduction='sum')

    
    def forward(self, output, target, mask):
        temp = output.cpu().detach().numpy()
        temp_t = target.cpu().numpy()
        loss = self.Mse(output, target)*self.a
        trans = transforms.ToTensor()
        for i in range(temp.shape[0]):
            temp2 = TSC(temp[i,:,:,:], mask)
            temp3 = TSC(temp_t[i,:,:,:], mask)
            temp2 = trans(temp2)
            temp3 = trans(temp3)
            loss += self.Mse(temp2,temp3)*self.b
        return loss