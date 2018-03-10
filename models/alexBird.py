import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class alexBird(nn.Module):
    def __init__(self, num_classes = 500):
        super(alexBird, self).__init__()
        # Define AlexNet
        alexnet = models.alexnet(pretrained=True)
        modules = list(alexnet.children())

        # Separate AlexNet to 3 parts
        alex_conv4 = list(modules[0].children())[0:10]
        alex_conv5 = list(modules[0].children())[10:]
        alex_fc = list(modules[1].children())[:-1]


        self.alexnet_conv4  = nn.Sequential(*alex_conv4)
        self.alexnet_conv5  = nn.Sequential(*alex_conv5)
        self.alexnet_fc     = nn.Sequential(*alex_fc)
        self.infer          = nn.Sequential(nn.Linear(4096, num_classes),
                                            nn.Softmax())

        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.infer[0].weight.data.normal_(0.0, 0.02)
        self.infer[0].bias.data.fill_(0)

    def forward(self, images):
        features = self.alexnet_conv4(images)
        features = self.alexnet_conv5(features)
        features = features.view(features.size(0), 256 * 6 * 6)
        features = self.alexnet_fc(features)

        return self.infer(features)
