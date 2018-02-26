import os,sys, math
import torch
import numpy                    as np
import torchvision.transforms   as transforms
from datasets                   import pascal3d_kp
import torch.utils.data.distributed

from IPython import embed


root_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_root = '/z/home/mbanani/datasets/pascal3d'

# # The New transform for ImageNet Stuff
# new_transform   = transforms.Compose([
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                     std=(0.229, 0.224, 0.225))])


def get_data_loaders(dataset, batch_size, num_workers, model, num_classes = 12, flip = False, valid = 0.0, parallel = False):

    image_size = 227
    alex_transform  = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(    mean=(104/255., 116.6/255., 122.6/255.),
                                                std=(1./255., 1./255., 1./255.) ) ])


    if dataset == "pascalVehKP":
        csv_train = os.path.join(root_dir, 'data/veh_pascal3d_kp_train.csv')
        csv_test  = os.path.join(root_dir, 'data/veh_pascal3d_kp_valid.csv')

        train_set = pascal3d_kp(csv_train, dataset_root= dataset_root, transform = alex_transform, im_size = image_size, num_classes = num_classes)
        test_set  = pascal3d_kp(csv_test, dataset_root= dataset_root, transform = alex_transform, im_size = image_size, num_classes = num_classes)

    elif dataset == "pascalKP":
        csv_train = os.path.join(root_dir, 'data/pascal3d_kp_train.csv')
        csv_test  = os.path.join(root_dir, 'data/pascal3d_kp_valid.csv')

        train_set = pascal3d_kp(csv_train, dataset_root= dataset_root, transform = train_transform, im_size = image_size, num_classes = num_classes)
        test_set  = pascal3d_kp(csv_test,  dataset_root= dataset_root, transform = test_transform,  im_size = image_size, num_classes = num_classes)

    else:
        print "Error in load_datasets: Dataset name not defined."


    # Generate validation dataset
    if valid > 0.0:
        valid_set   = train_set.generate_validation(valid)

    # Augment Training
    if flip:
        train_set.augment()
        print "Augmented Training Dataset - size : ", train_set.num_instances

    # Parallelize model
    train_sampler = None

    # Generate data loaders
    train_loader = torch.utils.data.DataLoader( dataset     =train_set,
                                                batch_size  =batch_size,
                                                shuffle     = (train_sampler is None),
                                                sampler     = train_sampler,
                                                num_workers =num_workers,
                                                drop_last   = True)

    test_loader  = torch.utils.data.DataLoader( dataset=test_set,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                drop_last = False)

    if valid > 0.0:
        print "Generated Validation Dataset - size : ", valid_set.num_instances
        valid_loader = torch.utils.data.DataLoader( dataset     = valid_set,
                                                    batch_size  = batch_size,
                                                    shuffle     = False,
                                                    pin_memory  = True,
                                                    num_workers = num_workers,
                                                    drop_last = False)
    else:
        valid_loader = None

    return train_loader, valid_loader, test_loader
