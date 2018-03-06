import os,sys, math
import torch

import numpy                    as np
import torchvision.transforms   as transforms

from datasets                   import advisee_dataset
from util                       import Paths

root_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_root = '/z/home/mbanani/datasets/pascal3d'

def get_data_loaders(dataset, batch_size, num_workers, model, num_classes = 12, flip = False, valid = 0.0, temperature = 1.0):

    image_size      = 227

    csv_train       = os.path.join(root_dir, 'data/pascal3d_train.csv')
    csv_test        = os.path.join(root_dir, 'data/pascal3d_valid.csv')
    keypoint_train  = os.path.join(root_dir, 'data/caffe_chcnn_pascal_train.npy')
    keypoint_test   = os.path.join(root_dir, 'data/caffe_chcnn_pascal_test.npy')

    kp_dict_train   = np.load(keypoint_train).item()
    kp_dict_test    = np.load(keypoint_test).item()
    dataset_root       = '/z/home/mbanani/datasets/pascal3d'


    if dataset == "advisee_full":
        train_set       = Adviser_Dataset(csv_train, dataset_root= dataset_root, transform = alex_transform, im_size = image_size, old_kp_dict = kp_dict_train, temperature = temperature)
        test_set        = Adviser_Dataset(csv_test,  dataset_root= dataset_root, transform = alex_transform, im_size = image_size, old_kp_dict = kp_dict_test, test_set = True)
    elif dataset == "advisee_test":
        train_set       = Adviser_Dataset(csv_test, dataset_root= dataset_root, transform = alex_transform, im_size = image_size, old_kp_dict = kp_dict_test, temperature = temperature)
        test_set        = train_set.generate_validation(0.3)
    else:
        print "Error: Dataset argument not recognized."
        exit()



    if valid > 0.0:
        valid_set   = train_set.generate_validation(valid)

        print "Generated Validation Dataset - size : ", valid_set.num_instances
        valid_loader = torch.utils.data.DataLoader( dataset     = valid_set,
                                                    batch_size  = batch_size,
                                                    shuffle     = False,
                                                    pin_memory  = True,
                                                    num_workers = num_workers,
                                                    drop_last = False)
    else:
        valid_loader = None


    # if flip:
    #     train_set.augment()
    #     print "Augmented Training Dataset - size : ", train_set.num_instances

    train_loader = torch.utils.data.DataLoader( dataset=train_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                drop_last = True)

    test_loader  = torch.utils.data.DataLoader( dataset=test_set,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                drop_last = False)

    return train_loader, valid_loader, test_loader
