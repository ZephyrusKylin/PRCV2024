import torch
import torch.nn as nn

class TridentConvFirst(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations):
        super(TridentConvFirst, self).__init__()
        self.dilations = dilations
        self.shared_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.shared_weight, a=0, nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        outputs = []
        for dilation in self.dilations:
            weight = self.shared_weight
            bias = self.bias
            output = nn.functional.conv2d(x, weight, bias, stride=1, 
                                          padding=dilation, dilation=dilation)
            output = self.relu(self.bn(output))
            outputs.append(output)
        return outputs

class TridentConv(TridentConvFirst):
    def __init__(self, in_channels, out_channels, kernel_size, dilations):
        super().__init__(in_channels, out_channels, kernel_size, dilations)
    def forward(self, x):
        outputs = []
        for idx in range(len(self.dilations)):
            weight = self.shared_weight
            bias = self.bias
            output = nn.functional.conv2d(x[idx], weight, bias, stride=1, 
                                          padding=self.dilations[idx], dilation=self.dilations[idx])
            output = self.relu(self.bn(output))
            outputs.append(output)
        return outputs

class TridentResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1,2,3]):
        super(TridentResidualBlock, self).__init__()
        self.conv1 = TridentConv(in_channels, out_channels, kernel_size=3, dilations=dilations)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = TridentConv(out_channels, out_channels, kernel_size=3, dilations=dilations)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bns = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual=[]
        for i in range(len(x)):
            residual_i = self.shortcut(x[i])
            residual_i = self.bns(residual_i)
            residual.append(residual_i)
        out = []
        conv1_results = self.conv1(x)
        out = self.conv2(conv1_results)
        for i in range(len(residual)):
            out[i]=out[i]+residual[i]
        return out

class TridentEncoder(nn.Module):
    def __init__(self,inchanns=1,channels=[16,32,64,128],layers=[2,2,2,2],dilations=[1,2,3]) -> None:
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(inchanns,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3,2,1)
        )
        self.first=TridentConvFirst(16,channels[0],3,dilations)
        self.stage1=self.make_layer(TridentResidualBlock,channels[0],channels[0],layers[0],dilations)
        
        self.downsample1=nn.Sequential(
            nn.Conv2d(channels[0],channels[1],3,2,1,bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
        ) 
        self.stage2=self.make_layer(TridentResidualBlock,channels[1],channels[1],layers[1],dilations)
        
        self.downsample2=nn.Sequential(
            nn.Conv2d(channels[1],channels[2],3,2,1,bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
        )
        self.stage3=self.make_layer(TridentResidualBlock,channels[2],channels[2],layers[2],dilations)
        
        self.downsample3=nn.Sequential(
            nn.Conv2d(channels[2],channels[3],3,2,1,bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
        )
        self.stage4=self.make_layer(TridentResidualBlock,channels[3],channels[3],layers[3],dilations)
    
    def forward(self,x):
        out=[]
        x=self.stem(x)

        x=self.first(x)
        # print(x[0].shape)
        x1=self.stage1(x)
        # for i in range(len(x1)):
        #     print(i,x1[i].shape)
        out.append(self.stage1(x))
        # print('out0',out[0][0].shape)

        for i in range(len(x1)):
            x1[i]=self.downsample1(x1[i])
        x2=self.stage2(x1)

        out.append(self.stage2(x1))
        # print('out1',out[1][0].shape)

        for i in range(len(x2)):
            x2[i]=self.downsample2(x2[i])
        x3=self.stage3(x2)
        out.append(self.stage3(x2))
        # print('out2',out[2][0].shape)

        for i in range(len(x3)):
            x3[i]=self.downsample3(x3[i])
        x4=self.stage4(x3)
        out.append(self.stage4(x3))
        # print('out3',out[3][0].shape)

        return out

    def make_layer(self,block,in_channel,out_channel,block_num,dilations):
        layer=[]
        layer.append(block(in_channel,out_channel,dilations))
        for _ in range(block_num-1):
            layer.append(block(out_channel,out_channel,dilations))
        return nn.Sequential(*layer)        
    
class TridentEncoder1(nn.Module):
    def __init__(self,inchanns=1,channels=[16,32,64],layers=[2,2,4],dilations=[1,2,3]) -> None:
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(inchanns,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3,2,1)
        )
        self.first=TridentConvFirst(16,channels[0],3,dilations)
        self.stage1=self.make_layer(TridentResidualBlock,channels[0],channels[0],layers[0],dilations)
        
        self.downsample1=nn.Sequential(
            nn.Conv2d(channels[0],channels[1],3,2,1,bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
        ) 
        self.stage2=self.make_layer(TridentResidualBlock,channels[1],channels[1],layers[1],dilations)
        
        self.downsample2=nn.Sequential(
            nn.Conv2d(channels[1],channels[2],3,2,1,bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
        )
        self.stage3=self.make_layer(TridentResidualBlock,channels[2],channels[2],layers[2],dilations)
        
        # self.downsample3=nn.Sequential(
        #     nn.Conv2d(channels[2],channels[3],3,2,1,bias=False),
        #     nn.BatchNorm2d(channels[3]),
        #     nn.ReLU(inplace=True),
        # )
        # self.stage4=self.make_layer(TridentResidualBlock,channels[3],channels[3],layers[3],dilations)
    
    def forward(self,x):
        out=[]
        x=self.stem(x)

        x=self.first(x)
        mid=x
        # print(x[0].shape)
        x1=self.stage1(x)
        # for i in range(len(x1)):
        #     print(i,x1[i].shape)
        out.append(x1)
        # print('out0',out[0][0].shape)

        for i in range(len(x1)):
            mid[i]=self.downsample1(x1[i])
        x2=self.stage2(mid)

        out.append(x2)
        # print('out1',out[1][0].shape)

        for i in range(len(x2)):
            mid[i]=self.downsample2(x2[i])
        x3=self.stage3(mid)
        out.append(x3)
        # print('out2',out[2][0].shape)

        # for i in range(len(x3)):
        #     x3[i]=self.downsample3(x3[i])
        # x4=self.stage4(x3)
        # out.append(self.stage4(x3))
        # print('out3',out[3][0].shape)

        return out

    def make_layer(self,block,in_channel,out_channel,block_num,dilations):
        layer=[]
        layer.append(block(in_channel,out_channel,dilations))
        for _ in range(block_num-1):
            layer.append(block(out_channel,out_channel,dilations))
        return nn.Sequential(*layer)        
    
class TridentEncoder2(nn.Module):
    def __init__(self,inchanns=1,channels=[16,32,64,128],layers=[2,2,2,2],dilations=[1,2,3]) -> None:
        super().__init__()
        self.first=TridentConvFirst(inchanns,channels[0],3,dilations)
        self.stage1=self.make_layer(TridentResidualBlock,channels[0],channels[0],layers[0],dilations)
        
        self.downsample1=nn.Sequential(
            nn.Conv2d(channels[0],channels[1],3,2,1,bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
        ) 
        self.stage2=self.make_layer(TridentResidualBlock,channels[1],channels[1],layers[1],dilations)
        
        self.downsample2=nn.Sequential(
            nn.Conv2d(channels[1],channels[2],3,2,1,bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
        )
        self.stage3=self.make_layer(TridentResidualBlock,channels[2],channels[2],layers[2],dilations)
        
        self.downsample3=nn.Sequential(
            nn.Conv2d(channels[2],channels[3],3,2,1,bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
        )
        self.stage4=self.make_layer(TridentResidualBlock,channels[3],channels[3],layers[3],dilations)
    
    def forward(self,x):
        out=[]

        x=self.first(x)
        # print(x[0].shape)
        x1=self.stage1(x)
        # for i in range(len(x1)):
        #     print(i,x1[i].shape)
        out.append(self.stage1(x))
        # print('out0',out[0][0].shape)

        for i in range(len(x1)):
            x1[i]=self.downsample1(x1[i])
        x2=self.stage2(x1)

        out.append(self.stage2(x1))
        # print('out1',out[1][0].shape)

        for i in range(len(x2)):
            x2[i]=self.downsample2(x2[i])
        x3=self.stage3(x2)
        out.append(self.stage3(x2))
        # print('out2',out[2][0].shape)

        for i in range(len(x3)):
            x3[i]=self.downsample3(x3[i])
        x4=self.stage4(x3)
        out.append(self.stage4(x3))
        # print('out3',out[3][0].shape)

        return out

    def make_layer(self,block,in_channel,out_channel,block_num,dilations):
        layer=[]
        layer.append(block(in_channel,out_channel,dilations))
        for _ in range(block_num-1):
            layer.append(block(out_channel,out_channel,dilations))
        return nn.Sequential(*layer)        


# model=TridentEncoder1()
# x=torch.randn(2,1,512,512)
# out1=model(x)
# for i in range(3):
#     for j in range(3):
#         print(out1[i][j].shape)