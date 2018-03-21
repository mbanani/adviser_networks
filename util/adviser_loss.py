"""
Multi-class loss for networks doing multi-class labeling for MSE, KL and BCE losses
"""

import torch

from torch      import nn
from IPython    import embed

import torch.nn.functional as F
import numpy               as np

class adviser_loss(nn.Module):
    def __init__(self, num_classes, loss = 'BCE', weights = None):
        super(adviser_loss, self).__init__()

        # self.num_classes = num_classes
        self.loss        = loss
        self.weights     = np.ones(3) if weights is None else weights

        if num_classes == 37:
            self.ranges = [0, 13, 26, 37]
        else:
            self.ranges = [0, 12, 24, 34]

        assert loss in ['BCE', 'KL', 'MSE']

    def forward(self, preds, labels, obj_classes):
        """
        :param preds:   Angle predictions (batch_size, 360 x num_classes)
        :param targets: Angle labels (batch_size, 360 x num_classes)
        :return: Loss. Loss is a variable which may have a backward pass performed.
        Apply Softmax over the preds, and then apply geometrics loss
        """
        # Set absolute minimum for numerical stability (assuming float16 - 6x10^-5)
        # preds = F.softmax(preds.float())
        labels      = labels.float()
        batch_size  = preds.size(0)
        loss        = torch.zeros(1)
        loss_0      = torch.zeros(1)
        loss_1      = torch.zeros(1)
        loss_2      = torch.zeros(1)
        weights     = torch.from_numpy(self.weights).float()


        if torch.cuda.is_available():
            obj_classes = obj_classes.cuda()
            weights = weights.cuda()
            loss_0    = loss_0.cuda()
            loss_1    = loss_1.cuda()
            loss_2    = loss_2.cuda()
            loss    = loss_2.cuda()

        loss      = torch.autograd.Variable(loss)
        loss_0    = torch.autograd.Variable(loss_0)
        loss_1    = torch.autograd.Variable(loss_1)
        loss_2    = torch.autograd.Variable(loss_2)
        losses    = [loss_0, loss_1, loss_2]
        weights = torch.autograd.Variable(weights)


        if self.loss == 'KL':
            for inst_id in range(batch_size):
                curr_class  = obj_classes[inst_id]
                start_index = self.ranges[curr_class]
                end_index   = self.ranges[curr_class + 1]

                losses[curr_class] += weights[curr_class] * F.kl_div(   F.softmax(preds[inst_id, start_index:end_index], dim=-1),
                                                                F.softmax(labels[inst_id, start_index:end_index], dim=-1),
                                                                size_average=False)


        elif self.loss == 'BCE':
            for inst_id in range(batch_size):
                curr_class  = obj_classes[inst_id]
                # embed()

                start_index = self.ranges[curr_class]
                end_index   = self.ranges[curr_class + 1]

                losses[curr_class] += weights[curr_class] * F.binary_cross_entropy(   F.softmax(preds[inst_id, start_index:end_index], dim=-1),
                                                            F.softmax(labels[inst_id, start_index:end_index], dim=-1),
                                                            size_average=False)


        elif self.loss == 'MSE':
            for inst_id in range(batch_size):
                curr_class  = obj_classes[inst_id]
                start_index = self.ranges[curr_class]
                end_index   = self.ranges[curr_class + 1]


                losses[curr_class] += weights[curr_class] * F.mse_loss( preds[inst_id, start_index:end_index],
                                                                labels[inst_id, start_index:end_index],
                                                                size_average=True).sqrt()

        loss = losses[0] + losses[1] + losses[2] #+ loss_1 + loss_2
        loss = loss / batch_size
        return loss, losses[0], losses[1], losses[2]
