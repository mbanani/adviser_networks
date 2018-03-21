import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from IPython import embed


class resBird(nn.Module):
    def __init__(self, num_classes = 500):
        super(resBird, self).__init__()
        # Define AlexNet
        resNet = models.resnet18(pretrained=True)

        self.conv1      = resNet.conv1
        self.bn1        = resNet.bn1
        self.relu       = resNet.relu
        self.maxpool    = resNet.maxpool
        self.layer1     = resNet.layer1
        self.layer2     = resNet.layer2
        self.layer3     = resNet.layer3
        self.layer4     = resNet.layer4
        self.avgpool    = resNet.avgpool
        self.fc         = nn.Sequential(nn.Linear(512, num_classes, bias = True))

    def init_weights(self):
        """Initialize the weights."""
        self.fc[0].weight.data.normal_(0.0, 0.02)
        self.fc[0].bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
