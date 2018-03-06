import torch

import torch.nn as nn
import numpy    as np

from torch.autograd import Function, Variable

class alexAdviser(nn.Module):
    def __init__(self, num_classes = 34, weights = None, weights_path = None):
        super(alexAdviser, self).__init__()

        # Normalization layers
        norm1 = nn.LocalResponseNorm(5, 0.0001, 0.75, 1)
        norm2 = nn.LocalResponseNorm(5, 0.0001, 0.75, 1)

        # conv layers
        conv1 = nn.Conv2d(3, 96, (11, 11), (4,4))
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d( (3,3), (2,2), (0,0), ceil_mode=True)

        conv2 = nn.Conv2d(96, 256, (5, 5), (1,1), (2,2), 1,2)
        relu2 = nn.ReLU()
        pool2 = nn.MaxPool2d( (3,3), (2,2), (0,0), ceil_mode=True)

        conv3 = nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1))
        relu3 = nn.ReLU()

        conv4 = nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1),1,2)
        relu4 = nn.ReLU()

        conv5 = nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1),1,2)
        relu5 = nn.ReLU()
        pool5 = nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)

        # inference layers
        fc6     = nn.Linear(9216,4096)
        relu6   = nn.ReLU()

        fc7     = nn.Linear(4096,4096)
        relu7   = nn.ReLU()

        drop6   = nn.Dropout(0.5)
        drop7   = nn.Dropout(0.5)

        infer   = nn.Linear(4096,num_classes)


        if weights != None:

            conv1.weight.data.copy_(weights['model_state_dict']['conv4.0.weight'])
            conv1.bias.data.copy_(weights['model_state_dict']['conv4.0.bias'])
            conv2.weight.data.copy_(weights['model_state_dict']['conv4.4.weight'])
            conv2.bias.data.copy_(weights['model_state_dict']['conv4.4.bias'])
            conv3.weight.data.copy_(weights['model_state_dict']['conv4.8.weight'])
            conv3.bias.data.copy_(weights['model_state_dict']['conv4.8.bias'])
            conv4.weight.data.copy_(weights['model_state_dict']['conv4.10.weight'])
            conv4.bias.data.copy_(weights['model_state_dict']['conv4.10.bias'])
            conv5.weight.data.copy_(weights['model_state_dict']['conv5.0.weight'])
            conv5.bias.data.copy_(weights['model_state_dict']['conv5.0.bias'])


            fc6.weight.data.copy_(weights['model_state_dict']['infer.0.weight'])
            fc6.bias.data.copy_(weights['model_state_dict']['infer.0.bias'])
            fc7.weight.data.copy_(weights['model_state_dict']['infer.3.weight'])
            fc7.bias.data.copy_(weights['model_state_dict']['infer.3.bias'])

        self.conv   = nn.Sequential( conv1, relu1, pool1, norm1,
                                    conv2, relu2, pool2, norm2,
                                    conv3, relu3,
                                    conv4, relu4,
                                    conv5,  relu5,  pool5)

        self.infer  = nn.Sequential( fc6,    relu6,  drop6,
                                    fc7,    relu7,  drop7,
                                    infer)

        self.init_weights()

    def init_weights(self):

        self.infer[0].weight.data.normal_(0.0, 0.01)
        self.infer[0].bias.data.fill_(0)
        self.infer[3].weight.data.normal_(0.0, 0.01)
        self.infer[3].bias.data.fill_(0)
        self.infer[6].weight.data.normal_(0.0, 0.01)
        self.infer[6].bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.infer(x)
