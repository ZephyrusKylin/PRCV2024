from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from TridentUNet.TridentUNet import *
# from TridentUNet.PABUNet import *
from DNANet.DNANet import *
from DNANet.model_ISTDUNet import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')
        elif model_name == 'Trid':
            self.model = TridentUNet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        # elif model_name == 'PAB':
        #     if mode == 'train':
        #         self.model = PABUNet(mode='train')
        #     else:
        #         self.model = PABUNet(mode='test')
    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
