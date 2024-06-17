import torch
import torch.nn as nn


class BiLocalChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(BiLocalChaFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)

        out = 2 * xl * topdown_wei + 2* xh * bottomup_wei
        out = self.post(out)
        return out


class AsymBiCSFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AsymBiCSFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),# 创建自适应平均池化层，将输入特征图池化为大小为(1, 1)
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            # nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            # nn.BatchNorm2d(self.bottleneck_channels),
            # nn.ReLU(True),

            # nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            # nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        avg_out=torch.mean(xl,dim=1,keepdim=True)
        max_out,_=torch.max(xl,dim=1,keepdim=True)
        xl1=torch.cat([avg_out,max_out],dim=1)
        bottomup_wei = self.bottomup(xl1)
        # bottomup_wei = self.bottomup(xl)
        xs = 2 * xl * topdown_wei + 2 * xh * bottomup_wei
        out = self.post(xs)
        return out

class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),# 创建自适应平均池化层，将输入特征图池化为大小为(1, 1)
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        # self.post = nn.Sequential(
        #     nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
        #     nn.BatchNorm2d(self.out_channels),
        #     nn.ReLU(True),
        # )

        # self.trans=nn.Conv2d(2,1,1,1,0)

    def forward(self, xh, xl):
        
        # high_max,_=torch.max(xh,dim=1)
        # high_max=high_max.unsqueeze(1)
        # high_avg=torch.mean(xh,dim=1).unsqueeze(1)
        # high_filter=self.trans(torch.cat([high_max,high_avg],dim=1))
        # xl=xl*high_filter
        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        # print(xh.shape,xl.shape,topdown_wei.shape,bottomup_wei.shape)
        out = 2 * xl * topdown_wei + 2 * xh * bottomup_wei
        # xs=xs*high_filter
        # out = self.post(xs)
        return out

class BiGlobalChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(BiGlobalChaFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * xl * topdown_wei + 2 * xh * bottomup_wei
        out = self.post(xs)
        return out

class AddFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AddFuseReduce, self).__init__()
        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.conv=nn.Conv2d(self.high_channels,self.out_channels,1)
    def forward(self,high,low):
        high=self.conv(high)
        out=high+low
        return out

class HighfilterLow(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(HighfilterLow,self).__init__()
        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )
        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )
        self.trans=nn.Conv2d(2,1,1,1,0)
        # self.post = nn.Sequential(
        #     nn.Conv2d(2, self.out_channels, 3, 1, 1),
        #     nn.BatchNorm2d(self.out_channels),
        #     nn.ReLU(True),
        # )
    def forward(self,high,low):
        high_max,_=torch.max(high,dim=1,keepdim=True)
        # high_max=high_max.unsqueeze(1)
        # print(high_max.shape)
        high_avg=torch.mean(high,dim=1,keepdim=True)
        # print(high_avg.shape)
        # exit()
        high_filter=self.trans(torch.cat([high_max,high_avg],dim=1))
        low=low*high_filter
        high=self.feature_high(high)
        out=low+high
        # out=torch.cat([high,low],dim=1)

        out=self.post(out)
        return out

class SAFA(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(SAFA,self).__init__()
        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )
        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )
    def forward(self,high,low):
        high=self.feature_high(high)
        mid=high+low
        w=[]
        for i in range(8):
            channel=high.shape[1]
            assert channel%8==0
            channel_interval=channel/8
            a=int(i*channel_interval)
            b=int((i+1)*channel_interval)
            high1=high[:,a:b,:,:]
            low1=low[:,a:b,:,:]
            mid1=mid[:,a:b,:,:]
            weight=high1*low1
            weight=weight.softmax(dim=1)*mid1
            w.append(weight)
        out = torch.cat([w[i] for i in range(8)], dim=1)
        out=self.post(out)
        return out   
    
class MSCAM(nn.Module):
    def __init__(self,in_channels,out_channels,reduction_ratio=4) -> None:
        super(MSCAM,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Conv2d(in_channels,out_channels//reduction_ratio,1,1,0),
            nn.BatchNorm2d(out_channels//reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(out_channels//reduction_ratio,out_channels,1,1,0),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self,x):
        x1=self.GAP(x)
        x1=self.fc(x1)
        x2=self.fc(x)
        out=(x1+x2).sigmoid()
        # out=x*out
        return out

class MSCAM1(nn.Module):
    def __init__(self,in_channels,out_channels,reduction_ratio=4) -> None:
        super(MSCAM1,self).__init__()
        self.GAP=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Conv2d(in_channels,out_channels//reduction_ratio,1,1,0),
            nn.ReLU(),
            nn.Conv2d(out_channels//reduction_ratio,out_channels,1,1,0),
        )
    def forward(self,high,low):
        x1=self.GAP(high)
        x1=self.fc(x1)
        x2=self.fc(low)
        out=high*x1.sigmoid()+low*x2.sigmoid()
        return out        

class AFF(nn.Module):
    def __init__(self,in_low_channels,in_high_channels,out_channels,reduction_ratio=4) -> None:
        super(AFF,self).__init__()
        self.cam=MSCAM(in_low_channels,out_channels,reduction_ratio)
        self.trans=nn.Sequential(
            nn.Conv2d(2,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.up=nn.ConvTranspose2d(in_high_channels,in_low_channels,4,2,1)
    def forward(self,high,low,vc=0):
        # high_max,_=torch.max(high,dim=1,keepdim=True)
        # # high_max=high_max.unsqueeze(1)
        # high_avg=torch.mean(high,dim=1,keepdim=True)
        # high_filter=self.trans(torch.cat([high_max,high_avg],dim=1))
        # low=low*high_filter
        #初始
        # high=self.up(high)
        fusion_init=low+high+vc
        # fusion_init=fusion_init*high_filter
        weight=self.cam(fusion_init)
        # low=low*high_filter
        #修改
        # out=
        out=(low+vc)*weight+high*(1-weight)
        return out

class AFF2(AFF):
    def __init__(self, in_low_channels, in_high_channels, out_channels, reduction_ratio=4) -> None:
        super().__init__(in_low_channels, in_high_channels, out_channels, reduction_ratio)
        self.cam=MSCAM(in_low_channels,out_channels,reduction_ratio)
        self.trans=nn.Sequential(
            nn.Conv2d(2,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.up=nn.ConvTranspose2d(in_high_channels,in_low_channels,4,2,1)
    def forward(self,high,low,vc=0):
        high_max,_=torch.max(high,dim=1,keepdim=True)
        # high_max=high_max.unsqueeze(1)
        high_avg=torch.mean(high,dim=1,keepdim=True)
        high_filter=self.trans(torch.cat([high_max,high_avg],dim=1))
        # low=low*high_filter
        #初始
        # high=self.up(high)
        fusion_init=low+high+vc
        # fusion_init=fusion_init*high_filter
        weight=self.cam(fusion_init)
        low=low*high_filter
        #修改
        # out=
        out=(low+vc)*weight+high*(1-weight)
        return out

class AFF1(nn.Module):
    def __init__(self,in_low_channels,in_high_channels,out_channels,reduction_ratio=4) -> None:
        super(AFF1,self).__init__()
        self.cam=MSCAM(in_low_channels,in_low_channels,reduction_ratio)
        self.trans=nn.Sequential(
            nn.Conv2d(2,1,1,1,0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.up=nn.Sequential()
        if in_high_channels != in_low_channels:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels=in_high_channels,out_channels=in_low_channels,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(in_low_channels),
                nn.ReLU(inplace=True)
            )
        self.post=nn.Sequential()
        if out_channels != in_low_channels:
            self.post = nn.Sequential(
                nn.Conv2d(in_channels=in_low_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        # self.up=nn.ConvTranspose2d(in_high_channels,in_low_channels,4,2,1)
    def forward(self,high,low,vc=0):
        high_max,_=torch.max(high,dim=1,keepdim=True)
        # high_max=high_max.unsqueeze(1)
        high_avg=torch.mean(high,dim=1,keepdim=True)
        high_filter=self.trans(torch.cat([high_max,high_avg],dim=1))
        # low=low*high_filter
        #初始
        high=self.up(high)
        fusion_init=low+high+vc
        fusion_init=fusion_init*high_filter
        weight=self.cam(fusion_init)
        # low=low*high_filter
        #修改
        # out=
        out=(low+vc)*weight+high*(1-weight)
        out=self.post(out)
        return out
    
class iAFF(nn.Module):
    def __init__(self,in_channels,out_channels,reduction_ratio) -> None:
        super(iAFF,self).__init__()
        self.cam=MSCAM(in_channels,out_channels,reduction_ratio)
    def forward(self,low,high):
        mid=low+high
        weight1=self.cam(mid)
        fusion_init=low*weight1+high*(1-weight1)
        weight=self.cam(fusion_init)
        out=low*weight+high*(1-weight)
        return out

class ChannelAttention(nn.Module):
    def __init__(self,in_channels,reduction=4) -> None:
        super(ChannelAttention,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(
            nn.Conv2d(in_channels,in_channels//reduction,1,bias=False),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction,in_channels,1,bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=self.fc(self.avg_pool(x))
        max_out=self.fc(self.max_pool(x))
        return self.sigmoid(avg_out+max_out)
    
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7) -> None:
        super(SpatialAttention,self).__init__()
        assert kernel_size in (3,7), 'kernel_size must be 3 or 7'
        padding = 3 if kernel_size==7 else 1
        self.fc=nn.Sequential(
            nn.Conv2d(2,1,kernel_size,stride=1,padding=padding,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_=torch.max(x,dim=1,keepdim=True)
        return self.fc(torch.cat([avg_out,max_out],dim=1))
    
class CBAM(nn.Module):
    def __init__(self,in_channels,out_channels,reduction=4,kernel_size=7) -> None:
        super(CBAM,self).__init__()
        self.stem = nn.Sequential()
        if in_channels != out_channels:
            self.stem=nn.Conv2d(in_channels,out_channels,1)
        self.ca=ChannelAttention(in_channels=out_channels,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
    def forward(self,x):
        x=self.stem(x)
        out=x*self.ca(x)
        result=out*self.sa(out)
        return result

class SCAF(nn.Module):
    def __init__(self,in_low_channel,in_high_channel,out_channel,reduction=4) -> None:
        super().__init__()
        assert in_low_channel == out_channel, 'in_low_channel must equal out_channel'
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channels=in_high_channel)
        self.up = nn.ConvTranspose2d(in_high_channel, out_channel,4, 2, 1)
        self.pa = nn.Sequential(
            nn.Conv2d(in_low_channel, in_low_channel//reduction, 1, 1, 0),
            nn.BatchNorm2d(in_low_channel//reduction),
            nn.ReLU(True),
            nn.Conv2d(in_low_channel//reduction, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid(),
        )
    def forward(self, high, low):
        high = self.ca(high) * high
        high = self.up(high)
        low = self.pa(low)
        low = self.sa(low)
        return low+high



import math
class ECABlock(nn.Module):
    def __init__(self,channels,b=1,gamma=2) -> None:
        super(ECABlock,self).__init__()
        t=int(abs((math.log(channels,2)+b)/gamma))
        k=t if t%2 else t+1
        self.avg_pooling=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=k,padding=int(k/2),bias=False)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        avg_out=self.avg_pooling(x).squeeze(-1).transpose(0,2,1)
        out=self.conv(avg_out).transpose(0,2,1).unsqueeze(-1)
        out=self.sigmoid(out)
        return out

class ECABlock1(nn.Module):
    def __init__(self,channels,b=1,gamma=2) -> None:
        super(ECABlock,self).__init__()
        t=int(abs((math.log(channels,2)+b)/gamma))
        k=t if t%2 else t+1
        self.avg_pooling=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=k,padding=int(k/2),bias=False)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        x=self.sigmoid(x)
        avg_out=self.avg_pooling(x).squeeze(-1).transpose(0,2,1)
        out=self.conv(avg_out).transpose(0,2,1).unsqueeze(-1)
        return out
    
class MCAF(nn.Module):
    def __init__(self,high_channels,low_channels,out_channels,reduction=16) -> None:
        super(MCAF,self).__init__()
        assert low_channels==out_channels,'low_channels must equal out_channels'
        self.high_channels=high_channels
        self.low_channels=low_channels
        if high_channels!=low_channels:
            self.conv=nn.Sequential(
                nn.Conv2d(high_channels,out_channels,1,1,0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        self.high_process=nn.Sequential(
            nn.Conv2d(out_channels,out_channels//reduction,kernel_size=3,stride=1,dilation=1,padding=1 ,bias=False),
            nn.BatchNorm2d(out_channels//reduction),
            nn.ReLU(),
            nn.Conv2d(out_channels//reduction,out_channels,kernel_size=3,stride=1,dilation=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )
        self.low_process=ECABlock1(out_channels)
    def forward(self,high,low):
        if self.high_channels!=self.low_channels:
            high=self.conv(high)
        high=self.high_process(high)
        low_out=low*self.low_process(low)
        out=low+high*low_out
        return out

import torch.nn.functional as F    
class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        print(out_normal.shape)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff1 = self.conv.weight.sum(2).sum(2)
            kernel_diff2 = kernel_diff1[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff2, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

x=torch.ones(1,3,5,5)
conv=Conv2d_cd(3,3)
y=conv(x)
# print(y)