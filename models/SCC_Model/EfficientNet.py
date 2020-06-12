import torch.nn as nn
import torch
from torchvision import models

from misc.layer import convDU, convLR

import torch.nn.functional as F
from misc.utils import *

import pdb

from efficientnet_pytorch import EfficientNet

# model_path = '../PyTorch_Pretrained/resnet101-5d3b4d8f.pth'

GlobalParams = collections.namedtuple('GlobalParams', [
    'num_classes'])

class EfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNet, self).__init__() 
        self.seen = 0

        self.res = EfficientNet.from_pretrained('efficientnet-b0') #perlu tambahin function from_pretrained?
        
        self.convDU = convDU(in_out_channels=64,kernel_size=(1,9))
        self.convLR = convLR(in_out_channels=64,kernel_size=(9,1))

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self.output_layer = nn.Linear(out_channels, self._global_params.num_classes)

        # import IPython; IPython.embed()

    def forward(self,x):
        x = self.res.extract_features(x)

        # pdb.set_trace()

        x = self.convDU(x)
        x = self.convLR(x)
        x = self.output_layer(x)

        x = F.upsample(x,scale_factor=2)
        return x