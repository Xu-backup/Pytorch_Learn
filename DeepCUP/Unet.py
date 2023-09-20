import torch
import torch.nn as nn
import scipy.io as scio
import math

class BasicBlock(nn.Module):
    """Basic Block for resnet
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
            # nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block):
        super().__init__()
        self.in_channels = 256
        self.conv_x = self._make_layer(block, 256, 1)

    def _make_layer(self, block, out_channels, stride):

        layers = []
        for stride in range(15):
            layers.append(block(self.in_channels, out_channels, stride)) ##这里的stride值没有用到

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv_x(x)
        return output
def resnet():

    return ResNet(BasicBlock)

class Unet(nn.Module):
    def __init__(self,img_W, img_H, batch_size):
        super(Unet, self).__init__()
        self.imgW = img_W
        self.imgH = img_H
        self.batchSize = batch_size

        # 权值初始化,正态分布
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.1)
#输入为（1，1，135，128）
        # 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3),
                      stride=(1, 1), padding_mode="zeros",padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
#（1，64，67，64）
        # 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                      stride=(1, 1), padding_mode="zeros", padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
#（1，128，33，32）
        # 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                      stride=(1, 1), padding_mode="zeros", padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
#（1，256，16，16）
        # resnet
        # middle_net = resnet()
        self.conv4 = resnet()


        # 上采样1 ##这是只有最后一次生成了八个图片  2*2上采样， channel减半 长宽翻倍
        self.upConv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2),
                               stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2),
                           stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 2),
                               stride=(2, 2)),
            nn.BatchNorm2d(1),  
            nn.Sigmoid()
        )


        #上采样2 ##这是八个并行的组件
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2),
                               stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2),
                           stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 2),
                               stride=(2, 2)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        # 上采样3
        self.upConv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2),
                               stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2),
                           stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 2),
                               stride=(2, 2)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        self.upConv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2),
                               stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2),
                           stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 2),
                               stride=(2, 2)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        self.upConv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2),
                               stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2),
                           stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 2),
                               stride=(2, 2)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        self.upConv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2),
                               stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2),
                           stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 2),
                               stride=(2, 2)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        self.upConv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2),
                               stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2),
                           stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 2),
                               stride=(2, 2)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        self.upConv8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2),
                               stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2),
                           stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2, 2),
                               stride=(2, 2)),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


    def forward(self,  x):
        x = x.reshape(self.batchSize, 1, self.imgW, self.imgH) 
        # print("输入：",x.shape)  (8,1,135,128)
        x = x.to(torch.float32)

        x1 = self.conv1(x) 
        # print("x1:",x1.shape) (8,64,67,64)


        x2 = self.conv2(x1) 
        # print("x2:", x2.shape) (8,128,33,32)

        x3 = self.conv3(x2)
        # print("x3:", x3.shape) (8,256,16,16)

        x4 = self.conv4(x3)
        # print("x4:", x4.shape)



        up1 = self.upConv1(x4)
        # print("up1:", up1.shape)

        up2 = self.upConv2(x4)
#         # print("up2:", up2.shape)

        up3 = self.upConv3(x4)
#         # print("up3:", up3.shape)

        up4 = self.upConv4(x4)
#         # print("up4:", up4.shape)

        up5 = self.upConv5(x4)
#         # print("up5:", up5.shape)

        up6 = self.upConv6(x4)
#         # print("up6:", up6.shape)

        up7 = self.upConv7(x4)
#         # print("up7:", up7.shape)

        up8 = self.upConv8(x4)
# print("up8:", up8.shape)

        decompressed = torch.cat([up1,up2,up3,up4,up5,up6,up7,up8],1)
        assert math.isnan(decompressed[0,0,0,0]) != True
        return decompressed

if __name__ == '__main__':
    x = torch.ones((8, 1, 135, 128))
    x = torch.randint_like(x, 1, 10)
    print(x.shape)
    my_net = Unet(135,128,8)
    output = my_net.forward(x)
    print('shuchu',output.shape)
