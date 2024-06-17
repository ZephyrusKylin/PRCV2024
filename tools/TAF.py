import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
from tools.FGSA import FA

class FusionAttention(nn.Module):
    def __init__(self,in_channels,out_channels,reduction_ratio=2) -> None:
        super(FusionAttention,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels//reduction_ratio,1,1,0),
            nn.BatchNorm2d(out_channels//reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(out_channels//reduction_ratio,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
        )
        self.fc2=nn.Sequential(
            nn.Conv2d(in_channels,out_channels//reduction_ratio,1,1,0),
            nn.BatchNorm2d(out_channels//reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(out_channels//reduction_ratio,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x1=self.GAP(x)
        x1=self.fc1(x1)
        x2=self.fc2(x)
        out=self.sigmoid(x1+x2)
        return out

class ChannelAttention(nn.Module):
    def __init__(self,in_channels,reduction=2) -> None:
        super(ChannelAttention,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Conv2d(in_channels,in_channels//reduction,1,bias=False),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction,in_channels,1,bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=self.avg_pool(x)
        out=self.fc(avg_out)
        out=self.sigmoid(out)*x+x
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7) -> None:
        super(SpatialAttention,self).__init__()
        assert kernel_size in (3,7), 'kernel_size must be 3 or 7'
        padding = 3 if kernel_size==7 else 1
        self.conv=nn.Sequential(
            nn.Conv2d(2,1,kernel_size,stride=1,padding=padding,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_=torch.max(x,dim=1,keepdim=True)
        out=self.conv(torch.cat([avg_out,max_out],dim=1))
        out=self.sigmoid(out)*x+x
        return out

class L2HAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(2,1,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self,high,low):
        low_avg=torch.mean(low,dim=1,keepdim=True)
        low_max,_=torch.max(low,dim=1,keepdim=True)
        out=self.conv(torch.cat([low_avg,low_max],dim=1))
        out=self.sigmoid(out)*high+high
        return out

class TAF(nn.Module):
    # high_channel:高分辨图通道数，low_channel:低分辨图通道数
    def __init__(self,high_channel,mid_channel,low_channel,filter=True) -> None:
        super().__init__()
        self.high=None
        if high_channel != mid_channel:
            self.high=nn.Sequential(
                nn.Conv2d(high_channel,mid_channel,1),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(True)
            )
        self.low=None
        if low_channel != mid_channel:
            self.low=nn.Sequential(
                nn.Conv2d(low_channel,mid_channel,1),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(True)
            )
        #modified 2024.4.18
        # self.pa=nn.Sequential(
        #     nn.Conv2d(mid_channel,mid_channel//4,1,1,0),
        #     nn.ReLU(),
        #     nn.Conv2d(mid_channel//4,mid_channel,1,1,0)
        # )
        #
        self.ca=ChannelAttention(in_channels=mid_channel)
        self.sa=SpatialAttention()
        self.att1=FusionAttention(in_channels=mid_channel,out_channels=mid_channel)
        self.att2=FusionAttention(in_channels=mid_channel,out_channels=mid_channel)
        # self.filter=
        self.filter=filter
        self.F1=L2HAttention()
        self.F2=L2HAttention()
        self.fa=FA()
    def forward(self,high,mid,low,flag=False,str1=None,str2=None):
        if self.high is not None:
            high=self.high(high)
        high=self.sa(high)
        if self.low is not None:
            low=self.low(low)
        low=self.ca(low)
        # low=self.ca(low)
        if flag:
            if not os.path.exists(str1+'mid/'):
                os.makedirs(str1+'mid/')
            mid=self.fa(mid,str1+'mid/'+str2)
        low=F.interpolate(low,size=(mid.size(2),mid.size(3)),mode='bilinear')
        if flag:
            if not os.path.exists(str1+'low/'):
                os.makedirs(str1+'low/')
            low=self.fa(low,str1+'low/'+str2)
        high=F.interpolate(high,size=(mid.size(2),mid.size(3)),mode='bilinear')
        if self.filter:
            mid=self.F1(mid,low)
        # atten1=self.att1(mid+low)
        # mid=low*atten1+mid*(1-atten1)
        mid=mid+low
        if self.filter:
            high=self.F2(high,mid)
        # atten2=self.att2(mid+high)
        # mid=mid*atten2+high*(1-atten2)
        mid=mid+high
        return mid

class DAF(nn.Module):
    # high_channel:高分辨图通道数，low_channel:低分辨图通道数
    def __init__(self,mid_channel,low_channel,filter=True) -> None:
        super().__init__()
        self.low=None
        if low_channel != mid_channel:
            self.low=nn.Sequential(
                nn.Conv2d(low_channel,mid_channel,1),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(True)
            )
        self.ca=ChannelAttention(in_channels=mid_channel)
        self.att=FusionAttention(in_channels=mid_channel,out_channels=mid_channel)
        # self.filter=filter
        self.filter=False
        self.Filter=L2HAttention()
    def forward(self,high,low):
        if self.low is not None:
            low=self.low(low)
        low=self.ca(low)
        low=F.interpolate(low,size=(high.size(2),high.size(3)),mode='bilinear')
        if self.filter:
            high=self.Filter(high,low)
        # atten=self.att(high+low)
        # high=low*atten+high*(1-atten)
        high=low+high
        return high



    