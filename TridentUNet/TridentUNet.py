import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from tools.FGSA import FA
from tools.swin import *
from tools.trident import *
from tools.common import *
from tools.fusion import *
from tools.common import _FCNHead
from tools.frenquacy_resnet import spctral_residual

class TridentUNet(nn.Module):
    def __init__(self,inchans=1,channels=[16,32,64,128],layers=[2,2,2,2],dilations=[1,2,3]) -> None:
        super().__init__()
        self.encoder=TridentEncoder(inchans,channels,layers,dilations)
        self.sr=spctral_residual()
        self.conv1=nn.Sequential(
            nn.Conv2d(channels[0]*3,channels[0],1,1,0,bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(channels[1]*3,channels[1],1,1,0,bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(channels[2]*3,channels[2],1,1,0,bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(channels[3]*3,channels[3],1,1,0,bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU()
        )
        # self.vit1=SwinTransformerBlock(c1=channels[0],c2=channels[0],num_heads=4,num_layers=1)
        # self.vit2=SwinTransformerBlock(c1=channels[1],c2=channels[1],num_heads=4,num_layers=1)
        # self.vit3=SwinTransformerBlock(c1=channels[2],c2=channels[2],num_heads=4,num_layers=1)
        # self.vit4=SwinTransformerBlock(c1=channels[3],c2=channels[3],num_heads=4,num_layers=1)
        # self.botteneck=self.make_layer(ResidualBlock,
        #                            channels[0]+channels[1]+channels[2]+channels[3]*2,
        #                            channels[3],1)
        # 修改后
        self.vit1=SwinViT(img_dim=256,in_channels=channels[0],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=8)
        self.vit2=SwinViT(img_dim=128,in_channels=channels[1],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=4)
        self.vit3=SwinViT(img_dim=64,in_channels=channels[2],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=2)
        self.vit4=SwinTransformerBlock(c1=channels[3],c2=channels[3],num_heads=4,num_layers=1)
        self.botteneck=self.make_layer(ResidualBlock,
                                   3*channels[2]+channels[3]*2,
                                   channels[3],1)
        
        self.downsample2 = nn.Upsample(scale_factor=1/2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1/4, mode='bilinear', align_corners=True)
        self.downsample8 = nn.Upsample(scale_factor=1/8, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion1=AsymBiChaFuseReduce(in_high_channels=channels[3],in_low_channels=channels[2],
                                         out_channels=channels[2])
        self.decoder3=self.make_layer(ResidualBlock,channels[2],channels[2],2)
        self.fusion2=AsymBiChaFuseReduce(in_high_channels=channels[2],in_low_channels=channels[1],
                                         out_channels=channels[1])
        self.decoder2=self.make_layer(ResidualBlock,channels[1],channels[1],2)
        self.fusion3=AsymBiChaFuseReduce(in_high_channels=channels[1],in_low_channels=channels[0],
                                         out_channels=channels[0])
        self.decoder1=self.make_layer(ResidualBlock,channels[0],channels[0],2)
        self.decoder0=self.make_layer(ResidualBlock,channels[0],channels[0],2)
        # self.head=_FCNHead(channels[0],1)
        self.newhead=nn.Sequential(
            nn.Conv2d(channels[0],1,1),
            nn.Sigmoid()
        )
        self.FA = FA()
        
    def make_layer(self,block,inchans,outchans,layers):
        layer=[]
        layer.append(block(inchans,outchans))
        for _ in range(layers-1):
            layer.append(block(outchans,outchans))
        return nn.Sequential(*layer)
    
    def forward(self,x):
        fm=self.encoder(x)
        x1=self.conv1(torch.cat([fm[0][0],fm[0][1],fm[0][2]],1))
        x1=self.sr(x1)+x1
        x2=self.conv2(torch.cat([fm[1][0],fm[1][1],fm[1][2]],1))
        x2=self.sr(x2)+x2
        x3=self.conv3(torch.cat([fm[2][0],fm[2][1],fm[2][2]],1))
        x3=self.sr(x3)+x3
        x4=self.conv4(torch.cat([fm[3][0],fm[3][1],fm[3][2]],1))
        x4=self.sr(x4)+x4

        # out=self.botteneck(torch.cat([self.downsample8(self.vit1(x1)),
        #                               self.downsample4(self.vit2(x2)),
        #                               self.downsample2(self.vit3(x3)),
        #                               self.vit4(x4),x4],dim=1))
        # 修改后
        out=self.botteneck(torch.cat([self.vit1(x1),
                                      self.vit2(x2),
                                      self.vit3(x3),
                                      self.vit4(x4),x4],dim=1))
        
        
        
        # out=self.up(out)
        # out=(out.sigmoid()>0).float()
        # out=out*(out.sigmoid()>0.5).float()
        out=self.up(out)
        out=self.fusion1(out,x3)
        out=self.decoder3(out)

        # out=self.up(out)
        # out=(out.sigmoid()>0).float()
        # out=out*(out.sigmoid()>0.4).float()
        out=self.up(out)
        out=self.fusion2(out,x2)
        out=self.decoder2(out)

        # out=self.up(out)
        # out=(out.sigmoid()>0).float()
        # out=out*(out.sigmoid()>0.3).float()
        out=self.up(out)
        # out=self.FA(out)
        out=self.fusion3(out,x1)
        # out=self.FA(out)
        out=self.decoder1(out)
        # out=self.FA(out)

        # out=(out.sigmoid()>0).float()
        # out=out*(out.sigmoid()>0.2).float()
        out=self.up(out)
        # out=self.FA(out)
        out=self.decoder0(out)
        # out=self.FA(out)
        # out=self.head(out)
        out=self.newhead(out)
        return out
    

class TridentUNet1(nn.Module):
    def __init__(self,inchans=1,channels=[16,32,64],layers=[2,2,4],dilations=[1,2,3]) -> None:
        super().__init__()
        self.encoder=TridentEncoder1(inchans,channels,layers,dilations)
        self.conv1=nn.Sequential(
            nn.Conv2d(channels[0]*3,channels[0],1,1,0,bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(channels[1]*3,channels[1],1,1,0,bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(channels[2]*3,channels[2],1,1,0,bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )
        
        # self.vit1=SwinTransformerBlock(c1=channels[0],c2=channels[0],num_heads=4,num_layers=1)
        # self.vit2=SwinTransformerBlock(c1=channels[1],c2=channels[1],num_heads=4,num_layers=1)
        # self.vit3=SwinTransformerBlock(c1=channels[2],c2=channels[2],num_heads=4,num_layers=1)
        # self.vit4=SwinTransformerBlock(c1=channels[3],c2=channels[3],num_heads=4,num_layers=1)
        # self.botteneck=self.make_layer(ResidualBlock,
        #                            channels[0]+channels[1]+channels[2]+channels[3]*2,
        #                            channels[3],1)
        # 修改后
        self.vit1=SwinViT(img_dim=256,in_channels=channels[0],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=4)
        self.vit2=SwinViT(img_dim=128,in_channels=channels[1],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=2)
        self.vit3=SwinViT(img_dim=64,in_channels=channels[2],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=1)
        self.botteneck=self.make_layer(ResidualBlock,
                                   4*channels[2],
                                   channels[2],1)
        
        self.downsample2 = nn.Upsample(scale_factor=1/2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1/4, mode='bilinear', align_corners=True)
        self.downsample8 = nn.Upsample(scale_factor=1/8, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.fusion1=AsymBiChaFuseReduce(in_high_channels=channels[3],in_low_channels=channels[2],
        #                                  out_channels=channels[2])
        # self.decoder3=self.make_layer(ResidualBlock,channels[2],channels[2],2)
        self.fusion2=AsymBiChaFuseReduce(in_high_channels=channels[2],in_low_channels=channels[1],
                                         out_channels=channels[1])
        self.decoder2=self.make_layer(ResidualBlock,channels[1],channels[1],2)
        self.fusion3=AsymBiChaFuseReduce(in_high_channels=channels[1],in_low_channels=channels[0],
                                         out_channels=channels[0])
        self.decoder1=self.make_layer(ResidualBlock,channels[0],channels[0],2)
        self.decoder0=self.make_layer(ResidualBlock,channels[0],channels[0],2)
        self.head=_FCNHead(channels[0],1)
        self.FA = FA()
        
    def make_layer(self,block,inchans,outchans,layers):
        layer=[]
        layer.append(block(inchans,outchans))
        for _ in range(layers-1):
            layer.append(block(outchans,outchans))
        return nn.Sequential(*layer)
    
    def forward(self,x):
        fm=self.encoder(x)
        x1=self.conv1(torch.cat([fm[0][0],fm[0][1],fm[0][2]],1))      
        x2=self.conv2(torch.cat([fm[1][0],fm[1][1],fm[1][2]],1))
        x3=self.conv3(torch.cat([fm[2][0],fm[2][1],fm[2][2]],1))
        out=self.botteneck(torch.cat([self.vit1(x1),
                                      self.vit2(x2),
                                      self.vit3(x3),x3],dim=1))
        
        # out=self.up(out)
        # out=(out.sigmoid()>0).float()
        # out=out*(out.sigmoid()>0.4).float()
        out=self.up(out)
        out=self.fusion2(out,x2)
        out=self.decoder2(out)

        # out=self.up(out)
        # out=(out.sigmoid()>0).float()
        # out=out*(out.sigmoid()>0.3).float()
        out=self.up(out)
        out=self.fusion3(out,x1)
        out=self.decoder1(out)

        # out=(out.sigmoid()>0).float()
        # out=out*(out.sigmoid()>0.2).float()
        out=self.up(out)
        out=self.decoder0(out)
        # out=self.FA(out)
        out=self.head(out)
        return out
    
class ResMTUNet(nn.Module):
    def __init__(self,inchans=1,channels=[32,64,128,256],layer=[2,2,2,2],ds=True,mode='train') -> None:
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(inchans,16,3,1,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,channels[0],1,1,0,bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1=self.make_layer(ResidualBlock,channels[0],channels[0],layer[0],stride=2)
        self.stage2=self.make_layer(ResidualBlock,channels[0],channels[1],layer[1],stride=2)
        self.stage3=self.make_layer(ResidualBlock,channels[1],channels[2],layer[2],stride=2)
        self.stage4=self.make_layer(ResidualBlock,channels[2],channels[3],layer[3],stride=2)

        self.vit1=SwinViT(img_dim=256,in_channels=channels[0],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=8)
        self.vit2=SwinViT(img_dim=128,in_channels=channels[1],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=4)
        self.vit3=SwinViT(img_dim=64,in_channels=channels[2],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=2)
        self.vit4=SwinTransformerBlock(c1=channels[3],c2=channels[3],num_heads=4,num_layers=1)
        self.botteneck=self.make_layer(ResidualBlock,
                                   3*channels[2]+channels[3]*2,
                                   channels[3],1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.fusion1=AsymBiChaFuseReduce(in_high_channels=channels[3],in_low_channels=channels[2],
                                         out_channels=channels[2])
        self.decoder3=self.make_layer(ResidualBlock,channels[2],channels[2],2)
        self.fusion2=AsymBiChaFuseReduce(in_high_channels=channels[2],in_low_channels=channels[1],
                                         out_channels=channels[1])
        self.decoder2=self.make_layer(ResidualBlock,channels[1],channels[1],2)
        self.fusion3=AsymBiChaFuseReduce(in_high_channels=channels[1],in_low_channels=channels[0],
                                         out_channels=channels[0])
        self.decoder1=self.make_layer(ResidualBlock,channels[0],channels[0],2)
        self.fusion4=AsymBiChaFuseReduce(in_high_channels=channels[0],in_low_channels=channels[0],
                                         out_channels=channels[0])
        self.decoder0=self.make_layer(ResidualBlock,channels[0],channels[0]//2,2)
        self.head=_FCNHead(channels[0]//2,1)
        self.ds=ds
        self.mode=mode
        # if self.ds:
        #     self.conv3=nn.Conv2d(channels[2],1,1,1,0)
        #     self.conv2=nn.Conv2d(channels[1],1,1,1,0)
        #     self.conv1=nn.Conv2d(channels[0],1,1,1,0)
        #     self.out=nn.Conv2d(4,1,1,1,0)
    def make_layer(self,block,inchans,outchans,layers,stride=1):
        layer=[]
        layer.append(block(inchans,outchans,stride))
        for _ in range(layers-1):
            layer.append(block(outchans,outchans,1))
        return nn.Sequential(*layer)
    def forward(self,x):
        x0=self.stem(x)
        x1=self.stage1(x0)
        x2=self.stage2(x1)
        x3=self.stage3(x2)
        x4=self.stage4(x3)
        bn=self.botteneck(torch.cat([self.vit1(x1),
                                      self.vit2(x2),
                                      self.vit3(x3),
                                      self.vit4(x4),x4],dim=1))
        dx3=self.decoder3(self.fusion1(self.up(bn),x3))
        dx2=self.decoder2(self.fusion2(self.up(dx3),x2))
        dx1=self.decoder1(self.fusion3(self.up(dx2),x1))
        dx0=self.decoder0(self.fusion4(self.up(dx1),x0))
        out=self.head(dx0)
        if self.ds:
            # gt3=self.up8(self.conv3(dx3))
            # gt2=self.up4(self.conv2(dx2))
            # gt1=self.up(self.conv1(dx1))
            # d0 = self.out(torch.cat((gt3, gt2, gt1, out), 1))
            if self.mode=='train':
                return out
                # return [torch.sigmoid(gt3),torch.sigmoid(gt2),torch.sigmoid(gt1),torch.sigmoid(d0),torch.sigmoid(out)]
            else:
                return out
        else:
            return out

                





