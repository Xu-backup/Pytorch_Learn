import math
import os
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import scipy.io as scio
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from evaluate import ssim, psnr
from Unet import Unet 
import time



def norm(np_mat):
    ymax = np.max(np_mat)
    ymin = np.min(np_mat)
    if(ymax-ymin!=0):
        output = (np_mat-ymin)/(ymax-ymin)
    else:
        output = np_mat/255
    return output


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    pipeline = transforms.Compose([
        transforms.ToTensor(),  #将图片转换为tensor
        #transforms.Normalize((0.5,),(0.5,)) #正则化降低模型复杂度，解决过拟合
    ])
    model = Unet(135,128,1).to(device)
    # load model weights
    weights_path = "D:\\SourceCode\\PyTorch\\DeepCUP\\UNet_6w_pos_30.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # load image5
    '''mat_path1 = "D:\\SourceCode\\PyTorch\\Vedio_Process\\mat\\E1.mat"
    mat_path2 = "D:\\SourceCode\\PyTorch\\Vedio_Process\\mat\\I1.mat"
    input =  scio.loadmat(mat_path1)['y']
    target = scio.loadmat(mat_path2)['input']'''

    mat_path1 = "Vedio_Process\\Result_Video_UCF\\G10.mat"
    input =  scio.loadmat(mat_path1)['Estimate']
    cv2.imwrite(f"D:\\SourceCode\\PyTorch\\DeepCUP\\result\\Estimate.jpg", input*255)
    target = scio.loadmat(mat_path1)['input']

    ##归一化
    input = input.astype(np.float32)
    input = pipeline(input)
    
    # expand batch dimension 输入图片的维度会有batch维度 用于BN层
    input = torch.unsqueeze(input, dim=0)
    print(input.shape)


    # 重构部分
    model.eval()
    with torch.no_grad():
        # predict class
        output = model(input.to(device))
    print(output.shape)
    
    #反归一化
    output = torch.squeeze(output, dim=0).cpu()
    output = np.array(output)
    output = (norm(output)*255).astype("uint8")
    target = (norm(target)*255).astype("uint8")

    p = 0
    s = 0
    for i in range(8):
        p += psnr(output[i,:,:], target[:,:,i])
        s += SSIM(output[i,:,:], target[:,:,i])
        #p1 = psnr(output[i,:,:], output[i+1,:,:])

    print(f"{p/8}") 
    print(f"{s/8}")
    for i in range(8):   
        cv2.imwrite(f"D:\\SourceCode\\PyTorch\\DeepCUP\\result\\{i}.jpg",output[i,:,:])
        cv2.imwrite(f"D:\\SourceCode\\PyTorch\\DeepCUP\\result\\true_{i}.jpg", target[:,:,i])
    #scio.savemat("D:\\SourceCode\\PyTorch\\DeepCUP\\result\\result.mat",{'output':output, 'target':target})
    
    ##计算500组的平均值
    '''Gnum = 500

    root_path = 'PyTorch/DeepCUP/test_fly'
    root_list = os.listdir(root_path)
    mat_list = []
    
    for i in root_list:
        mat_list.append(os.path.join(root_path, i))

    mat_list = mat_list[:Gnum]
    avg_p = 0
    avg_s = 0
    t = 0
    for j in mat_list:
        input =  scio.loadmat(j)['Estimate']
        target = scio.loadmat(j)['input']
        input = input.astype(np.float32)
        input = pipeline(input)
        input = torch.unsqueeze(input, dim=0)
        start = time.time()
        model.eval()
        with torch.no_grad():
        # predict class
            output = model(input.to(device))
        end = time.time()
        t += end-start
        output = torch.squeeze(output, dim=0).cpu()
        output = np.array(output)
        output = (norm(output)*255).astype("uint8")
        target = (norm(target)*255).astype("uint8")
        p = 0
        s = 0
        for i in range(8):
            p += psnr(output[i,:,:], target[:,:,i])
            s += ssim(output[i,:,:], target[:,:,i])
        avg_p += p/len(mat_list)/8
        avg_s += s/len(mat_list)/8
    
    print(f"avg_p:{avg_p}")
    print(f"avg_s:{avg_s}")
    print(f"时间：{t/500}")'''


    

 
    

if __name__ == '__main__':
    main()