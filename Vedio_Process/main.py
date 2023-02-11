import cv2
import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm

def frame_process(img):
    img = cv2.resize(img, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

##合并数据集
def es_merge(imgs_np, mask, inter=1, frame_num=8):
    imgs_np = imgs_np.copy()
    output = np.zeros((128+inter*(frame_num-1),128))
    for i in range(frame_num):
        imgs_np[:,:,i] = imgs_np[:,:,i]*mask
    for i in range(frame_num):                 
        output[i*inter:i*inter+128, 0:128] += imgs_np[:,:,i]
    ymax = np.max(output)
    ymin = np.min(output)
    output = (output-ymin)/(ymax-ymin)
    return output

##处理单个视频
def vedio_process(vedio_path, save_path, mask, start_group, frame_num=8):
    
    cap = cv2.VideoCapture(vedio_path)
    while(cap.isOpened()):
        imgs = np.zeros((128,128,frame_num))
        ##一次读取8帧图像并各自处理
        flag = 0
        for i in range(frame_num):
            flag, frame = cap.read()
            if not flag:
                break
            imgs[:,:,i] = frame_process(frame)
        if not flag:
            break

        ##保存原始图像
        '''for i in range(frame_num):
            img_path = os.path.join(group_dir,f"{i}.png")
            cv2.imwrite(img_path, imgs[:,:,i])'''
        ##生成模拟图像
        E = es_merge(imgs, mask)

        ##保存合成图像
        E_path = os.path.join(save_path, f"G{int(start_group)}.mat")
        scio.savemat(E_path,{"Estimate":E, 'input':imgs})
        start_group += 1
    cap.release()
    return start_group


def main():

    ##保存地址
    result_dir = "PyTorch/Vedio_Process/Result_Video_UCF"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    Gnum = 10000
    start_group = 0
    frame_num = 8

    ##读取mat文件
    mask = scio.loadmat('PyTorch/Vedio_Process/mat/mask.mat')
    mask = mask["f"]

    scio.savemat(os.path.join(result_dir,'mask.mat'),{'mask':mask})

    ##读取视频文件
    vedio_root = "PyTorch/UCF-101_dataset"
    assert os.path.exists(vedio_root), "path does not exist."
    root_list = os.listdir(vedio_root)
    
    ##获取所有视频的路径
    vedio_path = []
    for i in root_list:
        vedio_dir = os.path.join(vedio_root, i)
        vedio_list = os.listdir(vedio_dir)
        for j in vedio_list:
            vedio_path.append(os.path.join(vedio_dir, j))
    print(len(vedio_path))
    
    ##制作Gnum组数据
    vedio_path = tqdm(vedio_path)
    for i in vedio_path:
        start_group = vedio_process(i, result_dir, mask, start_group)
        #设置
        vedio_path.set_description(f'Group{start_group} start')
        if(start_group>Gnum):
            break

if __name__ == '__main__':
    main()

