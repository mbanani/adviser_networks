import torch
import numpy as np
import time
import pandas
import os

import numpy            as np
import torch.utils.data as data

from PIL            import Image
from torchvision    import transforms
import copy
import random

from IPython import embed

class advisee_dataset(data.Dataset):
    """
        Construct a Pascal Dataset.
        Inputs:
            csv_path    path containing instance data
            augment     boolean for flipping images
    """
    def __init__(self, kp_dict, dataset_root = None, im_size = 227, transform = None, temperature=1.0, test_set = False):

        start_time = time.time()

        assert kp_dict is not None

        # dataset parameters
        self.root           = dataset_root
        self.loader         = self.pil_loader
        self.num_classes    = 34
        self.temperature    = temperature


        # Load instance data from csv-file
        image_paths, bboxes, obj_class, geo_dists, flip, keys = self.dict_to_instances(kp_dict)

        print "csv file length: ", len(image_paths)

        self.im_paths       = image_paths
        self.bbox           = bboxes
        self.obj_cls        = obj_class
        self.labels         = geo_dists
        self.flip           = flip
        self.keys           = keys

        self.num_instances  = len(self.im_paths)
        self.im_size        = im_size

        assert transform   != None
        self.transform      = transform
        self.kp_dict        = kp_dict

        # # Set weights for loss
        # class_hist          = np.histogram(obj_cls, range(0, 13))[0]
        # mean_class_size     = np.mean([x for x in class_hist if x > 0])
        # self.loss_weights   = [mean_class_size / clsSize if clsSize >0 else 0 for clsSize in class_hist]
        # self.loss_weights   = np.asarray(self.loss_weights)
        self.loss_weights   = None

        # Print out dataset stats
        print "Dataset loaded in ", time.time() - start_time, " secs."
        print "Dataset size: ", self.num_instances

    def calculate_prob_vector(self, geo_dict):
        output      = np.zeros(self.num_classes)
        geo_list    = geo_dict.keys()
        geo_dists   = [geo_dict[geo_list[i]] for i in range(0, len(geo_list)) ]
        for i in range(0, len(geo_list)):
            # output[geo_list[i]] = -1.0 * geo_dists[i]/self.temperature
            output[geo_list[i]] = np.exp(-1.0 * geo_dists[i]/self.temperature)

        output = output / np.sum(output)

        return output

    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        # Load and transform image
        if self.root == None:
            im_path = self.im_paths[index]
        else:
            im_path = os.path.join(self.root, self.im_paths[index])

        bbox    = self.bbox[index]
        key     = self.keys[index]
        obj_cls = self.obj_cls[index]
        label   = self.labels[index]
        flip    = self.flip[index]

        # Load and transform image
        img = self.loader(im_path, bbox = bbox, flip = flip)
        if self.transform is not None:
            img = self.transform(img)

        # Load and transform label
        return img, label, obj_cls, key

    def __len__(self):
        return self.num_instances


    def pil_loader(self, path, bbox = None ,flip = False):
        # open path as file to avoid ResourceWarning
        # link: (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

                # Convert to BGR from RGB
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))

                if bbox != None: img = img.crop(box=bbox)

                # resize image using LANCZOS
                img = img.resize( (self.im_size, self.im_size), Image.LANCZOS)

                # flip image
                if flip: img = img.transpose(Image.FLIP_LEFT_RIGHT)

                return img

    def dict_to_instances(self, kp_dict):
        image_paths = []
        bboxes      = []
        obj_class   = []
        geo_dists   = []
        flip        = []
        dict_keys   = []

        for key in kp_dict.keys():
            im_path = '_'.join(key.split('_')[:-1])
            bb      = [int(i) for i in key.split('_')[-1].split('-')]
            obj_cls = kp_dict[key]['class']
            gdist   = self.calculate_prob_vector(kp_dict[key]['geo_dist'])
            # currently ignoring pred and label .. just retaining geo_dists

            image_paths.append(im_path)
            bboxes.append(bb)
            obj_class.append(obj_cls)
            geo_dists.append(gdist)
            dict_keys.append(key)

            if 'flip' in kp_dict[key].keys():
                flip.append(True)
            else:
                flip.append(False)

        return image_paths, bboxes, obj_class, geo_dists, flip, dict_keys

    def augment(self):
        print "Augment not implemented as a flipped image won't correspond to the same ground truth label. Exiting."
        exit()

    def generate_validation(self, ratio = 0.1):

        assert ratio > (2.*self.num_classes/float(self.num_instances)) and ratio < 0.5

        random.seed(a = 6306819796159687115)

        valid_class     = copy.deepcopy(self)

        # valid_size      = int(ratio * self.num_instances)
        # train_size      = self.num_instances - valid_size

        train_instances = [[], [], []]
        for i in range(0, self.num_instances):
            [self.obj_cls[i]].append(i)

        valid_instances =   random.sample(train_instances[0], int(len(train_instances[0]) * ratio)) + random.sample(train_instances[1], int(len(train_instances[1]) * ratio)) + random.sample(train_instances[2], int(len(train_instances[2]) * ratio))

        valid_size      = len(valid_instances)
        train_size      = self.num_instances - valid_size
        train_instances = range(0, self.num_instances)
        train_instances = [x for x in train_instances if x not in valid_instances]

        # assert train_size == len(train_instances) and valid_size == len(valid_instances)

        valid_class.im_paths        = [ self.im_paths[i]    for i in sorted(valid_instances) ]
        valid_class.keys            = [ self.keys[i]        for i in sorted(valid_instances) ]
        valid_class.bbox            = [ self.bbox[i]        for i in sorted(valid_instances) ]
        valid_class.obj_cls         = [ self.obj_cls[i]     for i in sorted(valid_instances) ]
        valid_class.labels          = [ self.labels[i]      for i in sorted(valid_instances) ]
        valid_class.flip            = [ self.flip[i]        for i in sorted(valid_instances) ]
        valid_class.num_instances   = len(valid_instances)

        self.im_paths            = [ self.im_paths[i]       for i in sorted(train_instances) ]
        self.keys                = [ self.keys[i]           for i in sorted(train_instances) ]
        self.bbox                = [ self.bbox[i]           for i in sorted(train_instances) ]
        self.obj_cls             = [ self.obj_cls[i]        for i in sorted(train_instances) ]
        self.labels              = [ self.labels[i]         for i in sorted(train_instances) ]
        self.flip                = [ self.flip[i]           for i in sorted(train_instances) ]
        self.num_instances       = len(train_instances)

        return valid_class
