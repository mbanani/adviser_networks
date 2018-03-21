import torch
import torch.nn as nn
import torchvision.models   as models
import torch.nn.functional  as F

from torch.autograd import Variable
from IPython        import embed


class ch_alexBird(nn.Module):
    def __init__(self, alexBird, num_classes = 500, num_kps = 15):
        super(ch_alexBird, self).__init__()

        # Image Scream
        self.alexnet_conv4  = alexBird.alexnet_conv4
        self.alexnet_conv5  = alexBird.alexnet_conv5
        temp_mods           = list(alexBird.alexnet_fc.children())
        temp_mods.append(nn.Dropout(0.5))
        self.alexnet_fc = nn.Sequential(*temp_mods)

        #Keypoint Stream
        kp_map_1    = nn.Conv2d(num_kps, num_kps, kernel_size=3, stride=1, dilation=1,padding =1)
        kp_relu_1   = nn.ReLU()
        kp_pool     = nn.AvgPool2d(3, stride=3)
        kp_map_2    = nn.Conv2d(num_kps, 1, kernel_size=3, stride=1, dilation=1,padding =0)
        kp_relu_2   = nn.ReLU()
        kp_class    = nn.Linear(num_kps,num_kps)

        # Fused layer
        fc8    = nn.Linear(4096 + 256, 4096)
        relu8  = nn.ReLU()
        drop8  = nn.Dropout(0.5)
        fc9    = nn.Linear(4096, num_classes)

        # Attention Stream
        self.map_linear  = nn.Sequential( kp_map_1,
                                        kp_relu_1,
                                        kp_pool,
                                        kp_map_2,
                                        kp_relu_2
                                        )

        self.cls_linear  = nn.Sequential( kp_class )

        self.fusion = nn.Sequential(fc8, relu8, drop8, fc9)

        self.init_weights()

    def init_weights(self):

        # Intialize weights for KP stream
        self.map_linear[0].weight.data.normal_(0.0, 0.01)
        self.map_linear[3].weight.data.normal_(0.0, 0.01)
        self.cls_linear[0].weight.data.normal_(0.0, 0.01)
        self.fusion[0].weight.data.normal_(0.0, 0.01)
        self.fusion[3].weight.data.normal_(0.0, 0.01)

        self.map_linear[0].bias.data.fill_(0)
        self.map_linear[3].bias.data.fill_(0)
        self.cls_linear[0].bias.data.fill_(0)
        self.fusion[0].bias.data.fill_(0)
        self.fusion[3].bias.data.fill_(0)

        # # Xavier initialization -- produces worse results for some reason :/
        # nn.init.xavier_uniform(self.map_linear[0].weight)
        # nn.init.xavier_uniform(self.map_linear[3].weight)
        # nn.init.xavier_uniform(self.cls_linear[0].weight)
        # nn.init.xavier_uniform(self.fusion[0].weight)
        # nn.init.xavier_uniform(self.fusion[3].weight)

        # nn.init.xavier_uniform(self.map_linear[0].bias)
        # nn.init.xavier_uniform(self.map_linear[3].bias)
        # nn.init.xavier_uniform(self.cls_linear[0].bias)
        # nn.init.xavier_uniform(self.fusion[0].bias)
        # nn.init.xavier_uniform(self.fusion[3].bias)


    def forward(self, images, kp_map, kp_class):
        # images    : 3x227x227
        # kp_map    : 46x46
        # kp_class  : num_kps

        # Image Stream
        images = self.alexnet_conv4(images)                 # 256x13x13

        # Keypoint Stream
        # KP map scaling performed in dataset class
        kp_class    = self.cls_linear(kp_class)             # num_kps
        kp_class    = kp_class.unsqueeze(2).unsqueeze(3)    # num_kps x 1 x 1
        kp_map      = kp_map.unsqueeze(1)                   # 1 x 13 x 13

        # Outer product of Map and class
        kp_map  = kp_map * kp_class                         # num_kps x 46 x 46

        # Convolve kp_map
        # Concatenate the two keypoint feature vectors
        kp_map  = self.map_linear(kp_map)                   # 1 x 13 x 13
        kp_map = kp_map.view(kp_map.size(0), 13*13)         # 13*13
        kp_map = F.softmax(kp_map, dim=-1)                  # 13*13
        kp_map = kp_map.view(kp_map.size(0), 1, 13, 13)     # 1 x 13 x 13


        # Attention -> Elt. wise product, then summation over x and y dims
        kp_map  = kp_map * images                           # 256 x 13 x 13
        kp_map  = kp_map.sum(3).sum(2)                      # 256

        # Continue from conv4
        images = self.alexnet_conv5(images)                 # 256x6x6
        images = images.view(images.size(0), -1)            # 9216
        images = self.alexnet_fc(images)                    # 4096

        # Concatenate fc7 and attended features
        images = torch.cat([images, kp_map], dim = 1)       # 4480 (4096+256)
        images = self.fusion(images)                        # 4096

        return images
