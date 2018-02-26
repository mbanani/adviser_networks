import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from IPython import embed

# Imports for legacy code!
from functools                              import reduce
from torch.legacy.nn.Module                 import Module as LegacyModule
from torch.legacy.nn.utils                  import clear
from torch.nn._functions.thnn.normalization import CrossMapLRN2d

class clickhere_cnn(nn.Module):
    def __init__(self, weights_path = None, num_classes = 12):
        super(clickhere_cnn, self).__init__()

        # Normalization layers
        norm1 = Lambda(lambda x,lrn=SpatialCrossMapLRN_temp(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data)))
        norm2 = Lambda(lambda x,lrn=SpatialCrossMapLRN_temp(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data)))

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


        fc6     = nn.Linear(9216,4096)
        relu6   = nn.ReLU()
        fc7     = nn.Linear(4096,4096)
        relu7   = nn.ReLU()
        drop6 = nn.Dropout(0.5)
        drop7 = nn.Dropout(0.5)


        #Keypoint Stream
        kp_map      = nn.Linear(2116,2116)
        kp_class    = nn.Linear(34,34)
        kp_fuse     = nn.Linear(2150,169)

        # Fused layer
        fc8     = nn.Linear(4096 + 384, 4096)
        relu8   = nn.ReLU()
        drop8 = nn.Dropout(0.5)

        # Prediction layers
        azim        = nn.Linear(4096, num_classes * 360)
        elev        = nn.Linear(4096, num_classes * 360)
        tilt        = nn.Linear(4096, num_classes * 360)

        if weights_path:
            npy_dict = np.load(weights_path).item()

            state_dict = npy_dict
            # Convert parameters to torch tensors
            for key in npy_dict.keys():
                state_dict[key]['weight'] = torch.from_numpy(npy_dict[key]['weight'])
                state_dict[key]['bias']   = torch.from_numpy(npy_dict[key]['bias'])

            conv1.weight.data.copy_(state_dict['conv1']['weight'])
            conv1.bias.data.copy_(state_dict['conv1']['bias'])
            conv2.weight.data.copy_(state_dict['conv2']['weight'])
            conv2.bias.data.copy_(state_dict['conv2']['bias'])
            conv3.weight.data.copy_(state_dict['conv3']['weight'])
            conv3.bias.data.copy_(state_dict['conv3']['bias'])
            conv4.weight.data.copy_(state_dict['conv4']['weight'])
            conv4.bias.data.copy_(state_dict['conv4']['bias'])
            conv5.weight.data.copy_(state_dict['conv5']['weight'])
            conv5.bias.data.copy_(state_dict['conv5']['bias'])

            fc6.weight.data.copy_(state_dict['fc6']['weight'])
            fc6.bias.data.copy_(state_dict['fc6']['bias'])
            fc7.weight.data.copy_(state_dict['fc7']['weight'])
            fc7.bias.data.copy_(state_dict['fc7']['bias'])
            fc8.weight.data.copy_(state_dict['fc8']['weight'])
            fc8.bias.data.copy_(state_dict['fc8']['bias'])

            kp_map.weight.data.copy_(state_dict['fc-keypoint-map']['weight'])
            kp_map.bias.data.copy_(state_dict['fc-keypoint-map']['bias'])
            kp_class.weight.data.copy_(state_dict['fc-keypoint-class']['weight'])
            kp_class.bias.data.copy_(state_dict['fc-keypoint-class']['bias'])
            kp_fuse.weight.data.copy_(state_dict['fc-keypoint-concat']['weight'])
            kp_fuse.bias.data.copy_(state_dict['fc-keypoint-concat']['bias'])

            if num_classes == 3 and (state_dict['pred_azimuth']['weight'].size()[0] > 360*3):
                azim.weight.data.copy_( torch.cat([  state_dict['pred_azimuth'][  'weight'][360*4:360*5, :],  state_dict['pred_azimuth'][  'weight'][360*5:360*6, :], state_dict['pred_azimuth'][  'weight'][360*8:360*9, :] ], dim = 0) )
                elev.weight.data.copy_( torch.cat([  state_dict['pred_elevation']['weight'][360*4:360*5, :],  state_dict['pred_elevation']['weight'][360*5:360*6, :], state_dict['pred_elevation']['weight'][360*8:360*9, :] ], dim = 0) )
                tilt.weight.data.copy_( torch.cat([  state_dict['pred_tilt'][     'weight'][360*4:360*5, :],  state_dict['pred_tilt'][     'weight'][360*5:360*6, :], state_dict['pred_tilt'][     'weight'][360*8:360*9, :] ], dim = 0) )

                azim.bias.data.copy_(   torch.cat([  state_dict['pred_azimuth']['bias'][360*4:360*5], state_dict['pred_azimuth']['bias'][360*5:360*6], state_dict['pred_azimuth']['bias'][360*8:360*9] ], dim = 0) )
                elev.bias.data.copy_(   torch.cat([  state_dict['pred_elevation']['bias'][360*4:360*5], state_dict['pred_elevation']['bias'][360*5:360*6], state_dict['pred_elevation']['bias'][360*8:360*9] ], dim = 0) )
                tilt.bias.data.copy_(   torch.cat([  state_dict['pred_tilt']['bias'][360*4:360*5], state_dict['pred_tilt']['bias'][360*5:360*6], state_dict['pred_tilt']['bias'][360*8:360*9] ], dim = 0) )
            else:
                azim.weight.data.copy_( state_dict['pred_azimuth'  ]['weight'] )
                elev.weight.data.copy_( state_dict['pred_elevation']['weight'] )
                tilt.weight.data.copy_( state_dict['pred_tilt'     ]['weight'] )

                azim.bias.data.copy_( state_dict['pred_azimuth'  ]['bias'] )
                elev.bias.data.copy_( state_dict['pred_elevation']['bias'] )
                tilt.bias.data.copy_( state_dict['pred_tilt'     ]['bias'] )


        # Define Network
        self.conv4 = nn.Sequential( conv1, relu1, pool1, norm1,
                                    conv2, relu2, pool2, norm2,
                                    conv3, relu3,
                                    conv4, relu4)

        self.conv5 = nn.Sequential( conv5,  relu5,  pool5)

        self.pool_map    = nn.Sequential(nn.MaxPool2d( (5,5), (5,5), (1,1), ceil_mode=True))
        self.map_linear  = nn.Sequential( kp_map )
        self.cls_linear  = nn.Sequential( kp_class )
        self.kp_softmax  = nn.Sequential( kp_fuse, nn.Softmax() )

        self.infer = nn.Sequential(fc6, relu6, drop6, fc7, relu7, drop7)
        self.fusion = nn.Sequential(fc8, relu8, drop8)


        self.azim = nn.Sequential(azim)
        self.elev = nn.Sequential(elev)
        self.tilt = nn.Sequential(tilt)

        if weights_path == None:
            self.init_weights()


    def init_weights(self):

        self.infer[0].weight.data.normal_(0.0, 0.01)
        self.infer[0].bias.data.fill_(0)
        self.infer[3].weight.data.normal_(0.0, 0.01)
        self.infer[3].bias.data.fill_(0)

        # Intialize weights for KP stream
        self.map_linear[0].weight.data.normal_(0.0, 0.01)
        self.map_linear[0].bias.data.fill_(0)
        self.cls_linear[0].weight.data.normal_(0.0, 0.01)
        self.cls_linear[0].bias.data.fill_(0)
        self.kp_softmax[0].weight.data.normal_(0.0, 0.01)
        self.kp_softmax[0].bias.data.fill_(0)

        # Initialize weights for fusion and inference
        self.fusion[0].weight.data.normal_(0.0, 0.01)
        self.fusion[0].bias.data.fill_(0)

        self.azim[0].weight.data.normal_(0.0, 0.01)
        self.azim[0].bias.data.fill_(0)
        self.elev[0].weight.data.normal_(0.0, 0.01)
        self.elev[0].bias.data.fill_(0)
        self.tilt[0].weight.data.normal_(0.0, 0.01)
        self.tilt[0].bias.data.fill_(0)


    def forward(self, images, kp_map, kp_class):
        # Image Stream
        features_conv4 = self.conv4(images)
        features_conv5 = self.conv5(features_conv4)
        features_conv5 = features_conv5.view(features_conv5.size(0), -1)
        features_fc7   = self.infer(features_conv5)

        # Keypoint Stream
        # KP map scaling performed in dataset class
        # kp_map       = self.pool_map(kp_map)
        kp_map_flat  = kp_map.view(kp_map.size(0), -1)
        features_map = self.map_linear(kp_map_flat)
        features_cls = self.cls_linear(kp_class)

        # Concatenate the two keypoint feature vectors
        # In deploy file, map over class
        features_kp = torch.cat([features_map, features_cls], dim = 1)

        # Softmax followed by reshaping into a 13x13
        # Conv4 as shape batch * 384 * 13 * 13
        features_kp = self.kp_softmax(features_kp)
        features_kp = features_kp.view(kp_map.size(0),1, 13, 13)

        # Attention -> Elt. wise product, then summation over x and y dims
        attention_mul   = features_kp * features_conv4
        attention_kp    = attention_mul.sum(3).sum(2)

        # Concatenate fc7 and attended features
        features_fused = torch.cat([features_fc7, attention_kp], dim = 1)
        features_fused = self.fusion(features_fused)

        # Final inference
        azim = self.azim(features_fused)
        elev = self.elev(features_fused)
        tilt = self.tilt(features_fused)

        return azim, tilt, elev

class SpatialCrossMapLRN_temp(LegacyModule):

	def __init__(self, size, alpha=1e-4, beta=0.75, k=1, gpuDevice=0):
		super(SpatialCrossMapLRN_temp, self).__init__()

		self.size = size
		self.alpha = alpha
		self.beta = beta
		self.k = k
		self.scale = None
		self.paddedRatio = None
		self.accumRatio = None
		self.gpuDevice = gpuDevice

	def updateOutput(self, input):
		assert input.dim() == 4

		if self.scale is None:
			self.scale = input.new()

		if self.output is None:
			self.output = input.new()

		batchSize = input.size(0)
		channels = input.size(1)
		inputHeight = input.size(2)
		inputWidth = input.size(3)

		if input.is_cuda:
			self.output = self.output.cuda(self.gpuDevice)
			self.scale = self.scale.cuda(self.gpuDevice)

		self.output.resize_as_(input)
		self.scale.resize_as_(input)

		# use output storage as temporary buffer
		inputSquare = self.output
		torch.pow(input, 2, out=inputSquare)

		prePad = int((self.size - 1) / 2 + 1)
		prePadCrop = channels if prePad > channels else prePad

		scaleFirst = self.scale.select(1, 0)
		scaleFirst.zero_()
		# compute first feature map normalization
		for c in range(prePadCrop):
			scaleFirst.add_(inputSquare.select(1, c))

		# reuse computations for next feature maps normalization
		# by adding the next feature map and removing the previous
		for c in range(1, channels):
			scalePrevious = self.scale.select(1, c - 1)
			scaleCurrent = self.scale.select(1, c)
			scaleCurrent.copy_(scalePrevious)
			if c < channels - prePad + 1:
				squareNext = inputSquare.select(1, c + prePad - 1)
				scaleCurrent.add_(1, squareNext)

			if c > prePad:
				squarePrevious = inputSquare.select(1, c - prePad)
				scaleCurrent.add_(-1, squarePrevious)

		self.scale.mul_(self.alpha / self.size).add_(self.k)

		torch.pow(self.scale, -self.beta, out=self.output)
		self.output.mul_(input)

		return self.output

	def updateGradInput(self, input, gradOutput):
		assert input.dim() == 4

		batchSize = input.size(0)
		channels = input.size(1)
		inputHeight = input.size(2)
		inputWidth = input.size(3)

		if self.paddedRatio is None:
			self.paddedRatio = input.new()
		if self.accumRatio is None:
			self.accumRatio = input.new()
		self.paddedRatio.resize_(channels + self.size - 1, inputHeight, inputWidth)
		self.accumRatio.resize_(inputHeight, inputWidth)

		cacheRatioValue = 2 * self.alpha * self.beta / self.size
		inversePrePad = int(self.size - (self.size - 1) / 2)

		self.gradInput.resize_as_(input)
		torch.pow(self.scale, -self.beta, out=self.gradInput).mul_(gradOutput)

		self.paddedRatio.zero_()
		paddedRatioCenter = self.paddedRatio.narrow(0, inversePrePad, channels)
		for n in range(batchSize):
			torch.mul(gradOutput[n], self.output[n], out=paddedRatioCenter)
			paddedRatioCenter.div_(self.scale[n])
			torch.sum(self.paddedRatio.narrow(0, 0, self.size - 1), 0, out=self.accumRatio)
			for c in range(channels):
				self.accumRatio.add_(self.paddedRatio[c + self.size - 1])
				self.gradInput[n][c].addcmul_(-cacheRatioValue, input[n][c], self.accumRatio)
				self.accumRatio.add_(-1, self.paddedRatio[c])

		return self.gradInput

	def clearState(self):
		clear(self, 'scale', 'paddedRatio', 'accumRatio')
		return super(SpatialCrossMapLRN_temp, self).clearState()


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))
