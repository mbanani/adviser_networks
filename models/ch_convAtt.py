import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from IPython import embed
import torch.nn.functional as F


class ch_convAtt(nn.Module):
    def __init__(self, weights, num_classes = 12):
        super(ch_convAtt, self).__init__()

        # # Normalization layers
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

        # FC layers
        fc6     = nn.Linear(9216,4096)
        relu6   = nn.ReLU()
        drop6 = nn.Dropout(0.5)

        fc7     = nn.Linear(4096,4096)
        relu7   = nn.ReLU()
        drop7 = nn.Dropout(0.5)

        #Keypoint Stream
        kp_map_1    = nn.Conv2d(34, 12, kernel_size=3, stride=1, dilation=1,padding =1)
        kp_relu_1   = nn.ReLU()
        kp_pool     = nn.AvgPool2d(3, stride=3)
        kp_map_2    = nn.Conv2d(12, 1, kernel_size=3, stride=1, dilation=1,padding =0)
        kp_relu_2   = nn.ReLU()
        kp_class    = nn.Linear(34,34)

        # Fused layer
        fc8     = nn.Linear(4096 + 384, 4096)
        relu8   = nn.ReLU()
        drop8   = nn.Dropout(0.5)

        # Prediction layers
        azim        = nn.Linear(4096, num_classes * 360)
        elev        = nn.Linear(4096, num_classes * 360)
        tilt        = nn.Linear(4096, num_classes * 360)

        assert weights != None, "Error: Current model assumes pretrained Image-Stream weights"

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


        # Define Network
        self.conv4 = nn.Sequential( conv1, relu1, pool1, norm1,
                                    conv2, relu2, pool2, norm2,
                                    conv3, relu3,
                                    conv4, relu4)

        self.conv5 = nn.Sequential( conv5,  relu5,  pool5)

        self.map_linear  = nn.Sequential( kp_map_1,
                                        kp_relu_1,
                                        kp_pool,
                                        kp_map_2,
                                        kp_relu_2
                                        )
        self.cls_linear  = nn.Sequential( kp_class )

        self.infer = nn.Sequential(fc6, relu6, drop6, fc7, relu7, drop7)
        self.fusion = nn.Sequential(fc8, relu8, drop8)


        self.azim = nn.Sequential(azim)
        self.elev = nn.Sequential(elev)
        self.tilt = nn.Sequential(tilt)

        if weights == None:
            self.init_weights()


    def init_weights(self, weights = None):


        if weights == None:
            self.infer[0].weight.data.normal_(0.0, 0.01)
            self.infer[0].bias.data.fill_(0)
            self.infer[3].weight.data.normal_(0.0, 0.01)
            self.infer[3].bias.data.fill_(0)

        # Intialize weights for KP stream

        self.map_linear[0].weight.data.normal_(0.0, 0.01)
        self.map_linear[3].weight.data.normal_(0.0, 0.01)
        self.cls_linear[0].weight.data.normal_(0.0, 0.01)
        self.fusion[0].weight.data.normal_(0.0, 0.01)
        self.azim[0].weight.data.normal_(0.0, 0.01)
        self.elev[0].weight.data.normal_(0.0, 0.01)
        self.tilt[0].weight.data.normal_(0.0, 0.01)

        self.map_linear[0].bias.data.fill_(0)
        self.map_linear[3].bias.data.fill_(0)
        self.cls_linear[0].bias.data.fill_(0)
        self.fusion[0].bias.data.fill_(0)
        self.azim[0].bias.data.fill_(0)
        self.elev[0].bias.data.fill_(0)
        self.tilt[0].bias.data.fill_(0)

        # # Xavier initialization -- produces worse results for some reason :/ 
        # nn.init.xavier_uniform(self.map_linear[0].weight)
        # nn.init.xavier_uniform(self.map_linear[3].weight)
        # nn.init.xavier_uniform(self.cls_linear[0].weight)
        # nn.init.xavier_uniform(self.fusion[0].weight)
        # nn.init.xavier_uniform(self.azim[0].weight)
        # nn.init.xavier_uniform(self.elev[0].weight)
        # nn.init.xavier_uniform(self.tilt[0].weight)

        # nn.init.xavier_uniform(self.map_linear[0].bias)
        # nn.init.xavier_uniform(self.map_linear[3].bias)
        # nn.init.xavier_uniform(self.cls_linear[0].bias)
        # nn.init.xavier_uniform(self.fusion[0].bias)
        # nn.init.xavier_uniform(self.azim[0].bias)
        # nn.init.xavier_uniform(self.elev[0].bias)
        # nn.init.xavier_uniform(self.tilt[0].bias)


    def forward(self, images, kp_map, kp_class):
        # images    : 3x227x227
        # kp_map    : 46x46
        # kp_class  : 34

        # Image Stream
        images = self.conv4(images)                         # 384x13x13

        # Keypoint Stream
        # KP map scaling performed in dataset class
        kp_class    = self.cls_linear(kp_class)             # 34
        kp_class    = kp_class.unsqueeze(2).unsqueeze(3)    # 34x1x1
        kp_map      = kp_map.unsqueeze(1)                   # 1x13x13

        # Outer product of Map and class
        kp_map  = kp_map * kp_class                         # 34 x 46 x 46

        # Convolve kp_map
        # Concatenate the two keypoint feature vectors
        kp_map  = self.map_linear(kp_map)                   # 1 x 13 x 13
        kp_map = kp_map.view(kp_map.size(0), 13*13)         # 13*13
        kp_map = F.softmax(kp_map, dim=-1)                  # 13*13
        kp_map = kp_map.view(kp_map.size(0), 13*13)         # 1 x 13 x 13


        # Attention -> Elt. wise product, then summation over x and y dims
        kp_map  = kp_map * images                           # 384x13x13
        kp_map  = kp_map.sum(3).sum(2)                      # 384

        # Continue from conv4
        images = self.conv5(images)                         # 256x6x6
        images = images.view(images.size(0), -1)            # 9216
        images = self.infer(images)                         # 4096

        # Concatenate fc7 and attended features
        images = torch.cat([images, kp_map], dim = 1)       # 4480 (4096+384)
        images = self.fusion(images)                        # 4096

        # Final inference
        azim = self.azim(images)                            # num_classes * 360
        elev = self.elev(images)                            # num_classes * 360
        tilt = self.tilt(images)                            # num_classes * 360

        return azim, elev, tilt
