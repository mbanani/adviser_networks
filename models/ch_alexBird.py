import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from IPython import embed


class ch_alexBird(nn.Module):
    def __init__(self, num_classes = 500):
        super(ch_alexBird, self).__init__()
        # Define AlexNet
        alexnet = models.alexnet(pretrained=True)
        modules = list(alexnet.children())

        # Separate AlexNet to 3 parts
        alex_conv4 = list(modules[0].children())[0:10]
        alex_conv5 = list(modules[0].children())[10:]
        alex_fc = list(modules[1].children())[:-1]

        #Keypoint Stream
        kpm_dim = 46 * 46
        kpc_dim = 15
        kp_map   = nn.Linear(kpm_dim, kpm_dim)
        kp_class = nn.Linear(kpc_dim, kpc_dim)
        kp_fuse  = nn.Linear(kpc_dim + kpm_dim, 169)

        # Fused layer
        fc8    = nn.Linear(4096 + 256, 4096)
        relu8  = nn.ReLU()
        drop8  = nn.Dropout(0.5)
        fc9    = nn.Linear(4096, num_classes)


        self.alexnet_conv4  = nn.Sequential(*alex_conv4)
        self.alexnet_conv5  = nn.Sequential(*alex_conv5)
        self.alexnet_fc     = nn.Sequential(*alex_fc)

        self.map_linear  = nn.Sequential( kp_map )
        self.cls_linear  = nn.Sequential( kp_class )
        self.kp_softmax  = nn.Sequential( kp_fuse, nn.Softmax() )

        self.fusion = nn.Sequential(fc8, relu8, drop8, fc9)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.fusion[0].weight.data.normal_(0.0, 0.02)
        self.fusion[0].bias.data.fill_(0)
        self.fusion[3].weight.data.normal_(0.0, 0.02)
        self.fusion[3].bias.data.fill_(0)
        self.kp_softmax[0].weight.data.normal_(0.0, 0.02)
        self.kp_softmax[0].bias.data.fill_(0)
        self.map_linear[0].weight.data.normal_(0.0, 0.02)
        self.map_linear[0].bias.data.fill_(0)
        self.cls_linear[0].weight.data.normal_(0.0, 0.02)
        self.cls_linear[0].bias.data.fill_(0)
        self.kp_softmax[0].weight.data.normal_(0.0, 0.02)
        self.kp_softmax[0].bias.data.fill_(0)

        # nn.init.xavier_normal(LAYER WEIGHTS)

    def forward(self, image, kp_map, kp_class):
        # embed()
        image = self.alexnet_conv4(image)

        # Keypoint Stream
        # KP map scaling performed in dataset class
        # kp_map  = self.pool_map(kp_map)
        kp_map  = kp_map.view(kp_map.size(0), 46*46)
        kp_map = self.map_linear(kp_map)

        kp_class = self.cls_linear(kp_class)

        # Concatenate the two keypoint feature vectors
        # In deploy file, map over class
        kp_map = torch.cat([kp_map, kp_class], dim = 1)

        # Softmax followed by reshaping into a 13x13
        # Conv4 as shape batch * 384 * 13 * 13 -> PyTorch implementation has 256 x 13 x 13
        kp_map = self.kp_softmax(kp_map)
        kp_map = kp_map.view(kp_map.size(0),1, 13, 13)

        # Attention -> Elt. wise product, then summation over x and y dims
        kp_map   = kp_map * image
        kp_map   = kp_map.sum(2).sum(2)

        # Rest of Image stream
        image = self.alexnet_conv5(image)
        image = image.view(image.size(0), -1)
        image = self.alexnet_fc(image)

        # Concatenate fc7 and attended features
        image = torch.cat([image, kp_map], dim = 1)
        image = self.fusion(image)

        return image
