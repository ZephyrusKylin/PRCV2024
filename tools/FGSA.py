import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class ChannelPool(nn.Module):
    def __init__(self, pool_types=['avg', 'max']):
        super(ChannelPool, self).__init__()
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = torch.mean(x,1).unsqueeze(1)
                channel_att_raw = avg_pool
            elif pool_type=='max':
                max_pool = torch.max(x,1)[0].unsqueeze(1)
                channel_att_raw = max_pool

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        return channel_att_sum

class FA(nn.Module):
    def __init__(self, group=1, fre_interval=1, pool_types=['avg', 'max'], visualization=True, reduction_ratio=4, init_values=0.):
        super(FA, self).__init__()
        self.compress = ChannelPool(pool_types)
        self.group = group
        self.fre_interval = fre_interval
        self.Visualization = visualization
    def min_max(self, x, to_min, to_max):
        x_min = torch.min(x)
        x_max = torch.max(x)
        return to_min + ((to_max - to_min) / (x_max - x_min)) * (x - x_min)
     
    def forward(self,x,img_dir):
        B, C, H1, W1 = x.shape
        group = self.group
        K = int(C/group)
        out_feature = None
        for i in range(group):
            x_i = x[:, i*K:(i+1)*K, :, :]                             # BKHW
            x_compress_i = self.compress(x_i).squeeze(1)              # B1HW -> BHW
            enhance_feature = []
            for b in range(B):
                feature_map = x_compress_i[b]
                f = torch.fft.fft2(feature_map)                           #二维傅里叶变换
                shift2center = torch.fft.fftshift(f)                      #傅里叶变换的结果进行频率域的中心化处理
                magnitude_spectrum = torch.abs(shift2center)              # 振幅谱
                phase_spectrum = torch.angle(shift2center)                # 相位谱
                '''可视化特征和频谱'''
                if self.Visualization:
                    # 可视化特征图
                    feature_map_copy = feature_map.cpu().detach()
                    plt.imshow(feature_map_copy, cmap='viridis')  # cmap参数可以指定使用的颜色映射，这里使用了viridis色彩映射
                    plt.colorbar()  # 添加颜色条
                    # print(img_dir+'.png')
                    plt.savefig(img_dir+'.png')  # 保存图像到指定文件夹内
                    plt.close()
                    # 可视化振幅谱
                #     magnitude_spectrum_copy = magnitude_spectrum.cpu().detach()
                #     phase_spectrum_copy = phase_spectrum.cpu().detach()
                #     # plt.subplot(1, 2, 1)
                #     plt.imshow(torch.log(magnitude_spectrum_copy + 1), cmap='gray')
                #     plt.title('Magnitude Spectrum')
                #     plt.axis('off')
                #     plt.savefig('/home/wangzhang/BasicIRSTD-main/Magnitude_Sperctrum/Magnitude_Spectrum_Before.png')
                #     plt.close()

                #     H, W = magnitude_spectrum.shape
                # center_h = int(H/2)
                # center_w = int(W/2)
                # R = min(center_h, center_w)
                # q = self.fre_interval     # 频率间隔
                # mask = torch.zeros((H, W), dtype=torch.int64)
                # for h in range(H):
                #     for w in range(W):
                #         h1 = h - center_h
                #         w1 = w - center_w
                #         r = np.sqrt(h1*h1+w1*w1)
                #         r_floor = np.floor(r)
                #         if r_floor > R:
                #             r_floor = R
                #         label = int(np.ceil(r_floor/q))
                #         mask[h, w] = label
                # '''可视化label'''
                # if self.Visualization:
                #     mask_copy = mask.cpu().detach()
                #     plt.imshow(mask_copy, cmap='viridis')  # cmap参数可以指定使用的颜色映射，这里使用了viridis色彩映射
                #     plt.colorbar()  # 添加颜色条
                #     plt.savefig('/home/wangzhang/BasicIRSTD-main/Magnitude_Sperctrum/mask.png')  # 保存图像到指定文件夹内
                #     plt.close()
                
                # mask_Label = mask.unique()
                # assert(len(mask_Label)==int(R/q)+1)     
                # fre_avg_pool = []
                # for label in mask_Label:
                #     fre_avg_pool.append(torch.mean(magnitude_spectrum[mask==label]))
                # fre_avg_pool = torch.tensor(fre_avg_pool)
                # # print(fre_avg_pool.shape)
                # # exit()
                # # fre_avg_pool = F.normalize(fre_avg_pool, dim=0)
                # fre_att = self.fc[i](fre_avg_pool.to(x.device))
                # magnitude_spectrum1 = magnitude_spectrum.clone()
                # for index in range(len(fre_att)):
                #     label = index
                #     attn = fre_att[index]
                #     magnitude_spectrum1[mask==label]=(magnitude_spectrum*attn)[mask==label]

                # '''可视化频谱'''
                # if self.Visualization:
                #     magnitude_spectrum1_copy = self.min_max(magnitude_spectrum1.cpu().detach(), 0, 1000)
                #     # plt.subplot(1, 2, 1)
                #     plt.imshow(torch.log(magnitude_spectrum1_copy + 1), cmap='gray')
                #     plt.title('Magnitude Spectrum')
                #     plt.axis('off')
                #     plt.savefig('/home/wangzhang/BasicIRSTD-main/Magnitude_Sperctrum/Magnitude_Spectrum_After.png')
                #     plt.close()
        return x