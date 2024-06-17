import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.deform_conv import DeformConv2d as DW

class ConvBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1) -> None:
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride=stride,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)


class UABBlock(nn.Module):
    def __init__(self,inchannels,outchannels,deep) -> None:
        super().__init__()
        self.conv1=ConvBlock(inchannels,outchannels,stride=1)
        self.layers1 = nn.ModuleList()
        self.layers1.append(ConvBlock(outchannels, outchannels, stride=1))
        for _ in range(deep - 1):
            # 假设我们希望每一层都有不同的输出通道
            self.layers1.append(ConvBlock(outchannels, outchannels, stride=2))
        self.pab=PBABlock(outchannels,outchannels)
        self.layers2 = nn.ModuleList()
        for _ in range(deep):
            # 假设我们希望每一层都有不同的输出通道
            self.layers2.append(ConvBlock(2*outchannels, outchannels, stride=1))
        self.up=nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.deep=deep
    def forward(self,x):
        x=self.conv1(x)
        skip=x
        em=[]
        for layer in self.layers1:
            x=layer(x)
            em.append(x)
        upsample=0
        for layer in self.layers2:
            x=torch.cat((x,em[self.deep-1-upsample]),dim=1)
            x=layer(x)
            if upsample < self.deep-1:
                x=self.up(x)
            upsample= upsample+1
        return x+skip

class DeFormConv2d(nn.Module):
    def __init__(self,channel) -> None:
        super().__init__()
        self.offset=nn.Conv2d(channel,18,3,1,1)
        self.dcn=DW(channel,channel,3,1,1)
    def forward(self,x):
        return self.dcn(x,self.offset(x))
    
class PBABlock(nn.Module):
    def __init__(self,inchannels,outchannels) -> None:
        super().__init__()
        self.pre=nn.Conv2d(inchannels,outchannels,1)
        self.maxpool=nn.MaxPool2d(3,2,1)
        self.dwconv=nn.Sequential(
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=2, padding=1, groups=outchannels),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(),
        )
        self.dconv=nn.Sequential(
            nn.Conv2d(outchannels,outchannels,1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(),
            nn.ConvTranspose2d(outchannels,outchannels,4,2,1),
            nn.BatchNorm2d(outchannels),
            nn.Sigmoid()
        )
        self.cmsa=nn.Sequential(
            nn.Conv2d(outchannels,outchannels//4,1),
            nn.BatchNorm2d(outchannels//4),
            nn.ReLU(),
            DeFormConv2d(outchannels//4),
            nn.BatchNorm2d(outchannels//4),
            nn.ReLU(),
            nn.Conv2d(outchannels//4,outchannels,1),
            nn.BatchNorm2d(outchannels),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.pre(x)
        x1=self.maxpool(x)
        x2=self.dwconv(x)
        out=self.dconv(x1+x2)*x
        out=out*self.cmsa(out)
        return out

class PABUNet(nn.Module):
    def __init__(self,channels=[1,16,32,64,128],deep=[5,4,3,2,1],mode='train') -> None:
        super().__init__()
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)
        self.uab1=UABBlock(channels[0],channels[1],deep[0])
        self.pab1_1=PBABlock(channels[1]+channels[2],channels[1])
        self.pab1_2=PBABlock(2*channels[1]+channels[2],channels[1])
        self.pab1_3=PBABlock(3*channels[1]+channels[2],channels[1])
        self.uab2=UABBlock(channels[1],channels[2],deep[1])
        self.pab2_1=PBABlock(channels[1]+channels[2]+channels[3],channels[2])
        self.pab2_2=PBABlock(2*channels[2]+channels[1]+channels[3],channels[2])
        self.uab3=UABBlock(channels[2],channels[3],deep[2])
        self.pab3_1=PBABlock(channels[4]+channels[2]+channels[3],channels[3])
        self.uab4=UABBlock(channels[3],channels[4],deep[3])
        self.uab5=UABBlock(channels[4],channels[4],deep[4])
        self.duab4=UABBlock(2*channels[4]+channels[3],channels[3],deep[3])
        self.duab3=UABBlock(3*channels[3]+channels[2],channels[2],deep[2])
        self.duab2=UABBlock(4*channels[2]+channels[1],channels[1],deep[1])
        self.duab1=UABBlock(5*channels[1],channels[1],deep[0])
        self.conv1=nn.Conv2d(channels[1],1,1)
        self.conv2=nn.Conv2d(channels[1],1,1)
        self.conv3=nn.Conv2d(channels[1],1,1)
        self.conv4=nn.Sequential(
            nn.Conv2d(channels[4]+channels[3]+channels[2]+2*channels[1],channels[1],3,1,1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1],channels[1],1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.finalconv=nn.Conv2d(channels[1],1,1)
        self.convout=nn.Conv2d(4,1,1)
        self.mode=mode
    def forward(self,x):
        x1=self.uab1(x)
        x2=self.uab2(self.pool(x1))
        x3=self.uab3(self.pool(x2))
        x4=self.uab4(self.pool(x3))
        x5=self.uab5(self.pool(x4))
        x1_1=self.pab1_1(torch.cat((x1,self.up(x2)),1))
        x2_1=self.pab2_1(torch.cat((x2,self.down(x1_1),self.up(x3)),1))
        x3_1=self.pab3_1(torch.cat((x3,self.down(x2_1),self.up(x4)),1))
        x1_2=self.pab1_2(torch.cat((x1,x1_1,self.up(x2_1)),1))
        x2_2=self.pab2_2(torch.cat((x2,x2_1,self.down(x1_2),self.up(x3_1)),1))
        x1_3=self.pab1_3(torch.cat((x1,x1_1,x1_2,self.up(x2_2)),1))
        dx4=self.duab4(torch.cat((self.up(x5),self.down(x3_1),x4),1))
        dx3=self.duab3(torch.cat((self.up(dx4),self.down(x2_2),x3,x3_1),1))
        dx2=self.duab2(torch.cat((self.up(dx3),x2,x2_1,x2_2,self.down(x1_3)),1))
        dx1=self.duab1(torch.cat((self.up(dx2),x1,x1_1,x1_2,x1_3),1))
        output1=self.conv1(x1_1).sigmoid()
        output2=self.conv2(x1_2).sigmoid()
        output3=self.conv3(x1_3).sigmoid()
        out=self.conv4(torch.cat((self.up_16(x5),self.up_8(dx4),self.up_4(dx3),self.up(dx2),dx1),1))
        out=self.finalconv(out).sigmoid()
        
        if self.mode=='train':
            return [output1,output2,output3,out]
        else:
            return out

