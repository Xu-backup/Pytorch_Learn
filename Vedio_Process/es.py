import numpy as np
import os
import cv2
import scipy.io as scio

result_dir = "./Result_Video/"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
Gnum = 10
frame_num = 8
inter = 1

##读取mat文件
input = scio.loadmat('PyTorch\Vedio_Process\mat\I1.mat')
input=np.array(input['input'])

y = scio.loadmat('PyTorch\Vedio_Process\mat\E1.mat')
y=np.array(y['y'])

mask = scio.loadmat('PyTorch\Vedio_Process\mat\mask.mat')
mask = mask["f"]

##编码Code
output = np.zeros((128+inter*(frame_num-1),128))  
for i in range(frame_num):
    input[:,:,i] = input[:,:,i]*mask   

##偏置叠加
for i in range(frame_num):                 
    output[i*inter:i*inter+128, 0:128] += input[:,:,i]

'''for i in range(Gnum):
    output = np.zeros((128+inter*(frame_num-1),128))
    temp_dir_name = os.path.join(result_dir, f"group {int(i)}")
    for g in range(frame_num):
        img_path = os.path.join(temp_dir_name,f"{g}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output[g*inter:g*inter+128, 0:128] += img
    
    ymax = np.max(output)
    ymin = np.min(output)
    output = (output-ymin)/(ymax-ymin)
    path = os.path.join(temp_dir_name,f"E{i}.jpg")
    cv2.imwrite(path, output)'''

ymax = np.max(output)
ymin = np.min(output)
output = (output-ymin)/(ymax-ymin)

print((output==y).all())

print("hello")