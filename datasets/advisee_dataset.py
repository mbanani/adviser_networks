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
    def __init__(self, csv_path, dataset_root = None, im_size = 227, transform = None, old_kp_dict=None, temperature=1.0, test_set = False):

        start_time = time.time()

        assert old_kp_dict is not None

        # dataset parameters
        self.root           = dataset_root
        self.loader         = self.pil_loader

        included_classes = [4,5,8]

        # Load instance data from csv-file
        a_im_paths, a_bbox, a_obj_cls, a_vp_labels = self.csv_to_instances(csv_path)

        im_paths  = []
        bbox      = []
        obj_cls   = []
        vp_labels = []


        for i in range(0, len(a_im_paths)):
            if a_obj_cls[i] in included_classes:
                im_paths.append(a_im_paths[i])
                bbox.append(a_bbox[i])
                obj_cls.append(a_obj_cls[i])
                vp_labels.append(a_vp_labels[i])

        # Load kp_dictionary
        new_kp_dict     = dict()
        reversed_dict   = dict()

        for i in range(0, len(im_paths)):
            key = '_'.join([im_paths[i].split('/')[2].split('.')[0], 'bb'+ '-'.join([str(bbox[i][j]) for j in range(0,4)])])
            azim = np.zeros(360)
            elev = np.zeros(360)
            tilt = np.zeros(360)

            azim[vp_labels[i][0]] = 1.0
            elev[vp_labels[i][1]] = 1.0
            tilt[vp_labels[i][2]] = 1.0

            new_kp_dict[key] = {'label': (azim, elev, tilt), 'class' : obj_cls[i]}
            reversed_dict[key] = {'label': (azim, elev, tilt), 'class' : obj_cls[i]}


        num_skiped = 0
        old_keys = []
        skipped = []


        for key in old_kp_dict.keys():
            if key[-1] == 'r':
                reverse = True
                new_key = '_'.join(key.split('_')[1:-4])
            else:
                reverse = False
                new_key = '_'.join(key.split('_')[1:-3])
            old_keys.append(new_key)

            if new_key in new_kp_dict.keys():
                if reverse:
                    reversed_dict[new_key][old_kp_dict[key]['kpc']] = old_kp_dict[key]['pred_prob']
                else:
                    new_kp_dict[new_key][old_kp_dict[key]['kpc']] = old_kp_dict[key]['pred_prob']
            else:
                skipped.append(new_key)
                num_skiped += 1

        keys_not_in_data = [key for key in (new_kp_dict.keys() + skipped) if key not in set(old_keys) ]

        for key in keys_not_in_data:
            _ = new_kp_dict.pop(key)
            _ = reversed_dict.pop(key)


        # print 'Num total : ', len(new_kp_dict.keys())
        # print 'Num in caffe dict : ', len(old_kp_dict.keys())
        # print "num_skiped ", num_skiped
        # print 'Num in caffe dict : ', len(list(set(old_keys)))
        # print [key in (new_kp_dict.keys() + skipped) if key not in set(old_keys) ]

        from util import kp_dict as metric

        curr_metric = metric()
        curr_metric.keypoint_dict = new_kp_dict
        curr_metric.calculate_geo_performance()
        r_curr_metric = metric()
        r_curr_metric.keypoint_dict = reversed_dict
        r_curr_metric.calculate_geo_performance()

        if test_set == True:
            kp_dont_matter  = []
            counter = np.zeros(12)
            all_counter = np.zeros(12)

            for key in curr_metric.keypoint_dict.keys():
                perf = []
                for kp_k in curr_metric.keypoint_dict[key]['geo_dist'].keys():
                    perf.append(curr_metric.keypoint_dict[key]['geo_dist'][kp_k])
                all_counter[curr_metric.keypoint_dict[key]['class']] += 1
                if (max(perf) - min(perf)) <= np.pi/24:
                    counter[curr_metric.keypoint_dict[key]['class']] += 1
                    kp_dont_matter.append(key)

            print "Counter (matters) : ", counter
            print "Counter (All)     : ", all_counter

            for key in kp_dont_matter:
                curr_metric.keypoint_dict.pop(key)


        print "csv file length: ", len(im_paths)

        self.im_paths       = []
        self.keys           = []
        self.bbox           = []
        self.obj_cls        = []
        self.labels         = []
        self.flip           = []
        self.num_classes    = 34
        self.temperature    = temperature

        for i in range(0, len(im_paths)):
            key = '_'.join([im_paths[i].split('/')[2].split('.')[0], 'bb'+ '-'.join([str(bbox[i][j]) for j in range(0,4)])])

            if key in new_kp_dict.keys():
                self.keys.append(key)
                self.im_paths.append(im_paths[i])
                self.bbox.append(bbox[i])
                self.obj_cls.append(obj_cls[i])
                self.labels.append( self.calculate_prob_vector(curr_metric.keypoint_dict[key]['geo_dist']) )
                self.flip.append(False)

        self.performance_dict = curr_metric.keypoint_dict
        self.num_instances  = len(self.im_paths)
        self.im_size        = im_size
        self.reverse_dict   = r_curr_metric.keypoint_dict

        self.num_instances  = len(self.im_paths)
        assert transform   != None
        self.transform      = transform

        # Set weights for loss
        class_hist          = np.histogram(obj_cls, range(0, 13))[0]
        mean_class_size     = np.mean([x for x in class_hist if x > 0])
        self.loss_weights   = [mean_class_size / clsSize if clsSize >0 else 0 for clsSize in class_hist]
        self.loss_weights   = np.asarray(self.loss_weights)

        # Print out dataset stats
        print "Dataset loaded in ", time.time() - start_time, " secs."
        print "Dataset size: ", self.num_instances

    def calculate_prob_vector(self, geo_dict):
        output      = np.zeros(self.num_classes)
        geo_list    = geo_dict.keys()
        geo_dists   = [geo_dict[geo_list[i]] for i in range(0, len(geo_list)) ]
        for i in range(0, len(geo_list)):
            output[geo_list[i]] = -1.0 * geo_dists[i]/self.temperature

        # for i in range(0, len(geo_list)):
        #     output[geo_list[i]] = np.exp(-1.0 * geo_dists[i]/self.temperature)
        #
        # output = output / np.sum(output)

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

    """
        Loads images and applies the following transformations
            1. convert all images to RGB
            2. crop images using bbox (if provided)
            3. resize using LANCZOS to rescale_size
            4. convert from RGB to BGR
            5. (? not done now) convert from HWC to CHW
            6. (optional) flip image

        TODO: once this works, convert to a relative path, which will matter for
              synthetic data dataset class size.
    """
    def pil_loader(self, path, bbox = None ,flip = False):
        # open path as file to avoid ResourceWarning
        # link: (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

                # Convert to BGR from RGB
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))

                # crop (TODO verify that it's the correct ordering!)
                if bbox != None:
                    img = img.crop(box=bbox)

                # verify that imresize uses LANCZOS
                img = img.resize( (self.im_size, self.im_size), Image.LANCZOS)

                # flip image
                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                return img

    def csv_to_instances(self, csv_path):
        df   = pandas.read_csv(csv_path, sep=',')
        data = df.values

        data_split = np.split(data, [0, 1, 5, 6, 9], axis=1)
        del(data_split[0])

        image_paths = np.squeeze(data_split[0]).tolist()
        bboxes      = data_split[1].tolist()
        obj_class   = np.squeeze(data_split[2]).tolist()
        viewpoints  = data_split[3].tolist()

        return image_paths, bboxes, obj_class, viewpoints

    def augment(self):



        for i in range(0, self.num_instances):
            key = '_'.join([self.im_paths[i].split('/')[2].split('.')[0], 'bb'+ '-'.join([str(self.bbox[i][j]) for j in range(0,4)])])

            if key in self.reverse_dict.keys():
                self.keys.append(key)
                self.im_paths.append(self.im_paths[i])
                self.bbox.append(self.im_paths[i])
                self.obj_cls.append(self.im_paths[i])
                self.labels.append( self.calculate_prob_vector(self.reverse_dict[key]['geo_dist']) )
                self.flip.append(True)

        self.num_instances = len(self.im_paths)
        print "Augmented dataset. New size: ", self.num_instances

    def generate_validation(self, ratio = 0.1):
        assert ratio > (2.*self.num_classes/float(self.num_instances)) and ratio < 0.5

        random.seed(a = 6306819796159687115)

        valid_class     = copy.deepcopy(self)

        # valid_size      = int(ratio * self.num_instances)
        # train_size      = self.num_instances - valid_size
        train_instances = [[], [], []]
        for i in range(0, self.num_instances):
            if self.obj_cls[i] == 4:
                train_instances[0].append(i)
            if self.obj_cls[i] == 5:
                train_instances[1].append(i)
            if self.obj_cls[i] == 8:
                train_instances[2].append(i)

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
