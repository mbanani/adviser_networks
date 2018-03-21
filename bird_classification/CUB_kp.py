import torch.utils.data as data

from PIL import Image
from torchvision import transforms

import torch
import os
import numpy as np
import time
import pandas
import copy
import random

from IPython import embed

class CUB_kp(data.Dataset):

    def __init__(self, csv_path, dataset_dir, preprocessed_root = None, image_size = 227, map_size=46, transform = None, unique = True):
        self.num_classes    = 200
        self.loader         = self.pil_loader
        self.img_size       = image_size

        if not (os.path.isfile(csv_path)):
            print "CSV files do not exist - Creating CSV files for CUB dataset."
            self.create_csv(dataset_dir)

        if preprocessed_root:
            self.preprocessed = True
            self.image_root = preprocessed_root
        else:
            self.preprocessed = False
            self.image_root = os.path.join(dataset_dir, "images")

        self.keypoint_list = [  "back",
                                "beak",
                                "belly",
                                "breast",
                                "crown",
                                "forehead",
                                "left eye",
                                "left leg",
                                "left wing",
                                "nape",
                                "right eye",
                                "right leg",
                                "right wing",
                                "tail",
                                "throat"]

        i = 0
        curr_time = time.time()

        im_paths, img_class_id, bbs, kp_cls, s_kp_locs = self.pandas_csv_to_info(csv_path)

        indices_to_remove = []
        kp_locs = []

        for i in range(0, len(s_kp_locs)):
            # VERY IMPORTANT ... Correct bounding box calculation .. (CSV is x1,y1, w,h NOT  x1,y1,x2,y2)
            bbs[i][2] += bbs[i][0]
            bbs[i][3] += bbs[i][1]

            if (bbs[i][2] == bbs[i][0]) or (bbs[i][3] == bbs[i][1]):
                indices_to_remove.append(i)
            else:
                kp_x = float(s_kp_locs[i][0] - bbs[i][0]) / float(bbs[i][2] - bbs[i][0])
                kp_y = float(s_kp_locs[i][1] - bbs[i][1]) / float(bbs[i][3] - bbs[i][1])

                if not (kp_x >= 0.0 and kp_x <= 1.0 and kp_y >= 0.0 and kp_y <= 1.0) :
                    indices_to_remove.append(i)
                else:
                    assert kp_x >= 0.0 and kp_x <= 1.0,  "Incrrect KP " + str([kp_x, kp_y]) + " at " + str(i) + " -- " + "BBS: " + str(bbs[i]) + " - kps : " + str(s_kp_locs[i])
                    assert kp_y >= 0.0 and kp_y <= 1.0,  "Incrrect KP " + str([kp_x, kp_y]) + " at " + str(i) + " -- " + "BBS: " + str(bbs[i]) + " - kps : " + str(s_kp_locs[i])
                    kp_locs.append(tuple([kp_x, kp_y]))



        print "Removing ", len(indices_to_remove), " of ", len(im_paths), "for having incorrect bounding boxes"
        indices_to_remove = indices_to_remove[::-1]
        for i in indices_to_remove:
            del im_paths[i]
            del img_class_id[i]
            del bbs[i]
            del kp_cls[i]

        # TODO .. generate unique set of images
        if unique:
            unique_set = []
            seen_images = []
            for i in range(0, len(im_paths)):
                if not (im_paths[i] in seen_images):
                    unique_set.append(i)
                    seen_images.append(im_paths[i])

            im_paths        = [im_paths[i] for i in unique_set]
            img_class_id    = [img_class_id[i] for i in unique_set]
            bbs             = [bbs[i] for i in unique_set]
            kp_cls          = [kp_cls[i] for i in unique_set]
            kp_locs         = [kp_locs[i] for i in unique_set]




        # Generate uids and augment im_id
        uids = []
        for i in range(0, len(im_paths)):
            im_paths[i]     = os.path.join(self.image_root, im_paths[i])
            uids.append(im_paths[i] + '_objc' + str(img_class_id[i]) + '_kpc' + str(kp_cls[i]) )


        size_dataset = len(im_paths)
        print "csv file length: ", size_dataset

        bboxes  = [tuple(bbox) for bbox in bbs]
        # kp_locs = [tuple(kpLoc) for kpLoc in kp_locs]

        print "Dataset loaded in ", time.time() - curr_time, " secs."
        print "Dataset size: ", len(im_paths)



        self.image_paths        = im_paths
        self.bboxes             = bboxes
        self.obj_class          = img_class_id
        self.flips              = [False] * len(im_paths)
        self.keypoint_cls       = [kpc - 1 for kpc in kp_cls]
        self.keypoint_loc       = kp_locs
        self.uids               = uids

        self.map_size           = map_size
        self.num_instances      = len(self.image_paths)
        self.num_kps            = len(self.keypoint_list)

        # Normalization as instructed from pyTorch documentation
        self.transform = transform or transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225))])




    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # Load and transform image
        # Load and transform image
        img = self.loader(self.image_paths[index], self.bboxes[index], self.flips[index])
        if self.transform is not None:  img = self.transform(img)

        # Generate keypoint map image, and kp class vector
        kp_loc              = self.keypoint_loc[index]
        kp_cls              = self.keypoint_cls[index]
        kp_class            = np.zeros( (15) )
        kp_class[kp_cls]    = 1
        kp_map              = self.generate_kp_map_chebyshev(kp_loc)

        kp_class   = torch.from_numpy(kp_class).float()
        kp_map     = torch.from_numpy(kp_map).float()

        return img, self.obj_class[index], kp_map, kp_class , self.uids[index]

    def __len__(self):
        return self.num_instances

    def pil_loader(self, path, bbox, flip):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:

                img = img.convert('RGB')

                if not self.preprocessed:
                    img = img.crop(box=bbox)
                    img = img.resize( (self.img_size, self.img_size), Image.LANCZOS)
                else:
                    if not (img.size[0] == img.size[1] == self.img_size):
                        print path

                if flip:
                    img.transpose(Image.FLIP_LEFT_RIGHT)
                return img

    def pandas_csv_to_info(self, csv_path):
        df = pandas.read_csv(csv_path, sep=',')
        data = df.values

        data_split = np.split(data, [0, 1, 5, 6, 7, 9], axis=1)
        del(data_split[0])

        image_paths  = np.squeeze(data_split[0]).tolist()
        bbox         = data_split[1].tolist()
        img_class_id = np.squeeze(data_split[2]).tolist()
        kp_class     = np.squeeze(data_split[3]).tolist()
        kp_locs      = np.squeeze(data_split[4]).tolist()

        return image_paths, img_class_id, bbox, kp_class, kp_locs

    def create_csv(self, dataset_dir):

        file_h    = open(os.path.join(dataset_dir , 'bounding_boxes.txt'))
        old_lines = file_h.readlines()
        new_lines = [ tuple(line.replace('\n', '').split(' ')) for line in old_lines ]
        bounding_boxes  = new_lines
        file_h.close()

        file_h    = open(os.path.join(dataset_dir , 'image_class_labels.txt'))
        old_lines = file_h.readlines()
        new_lines = [ tuple(line.replace('\n', '').split(' ')) for line in old_lines ]
        image_class_labels  = new_lines
        file_h.close()

        file_h    = open(os.path.join(dataset_dir , 'train_test_split.txt'))
        old_lines = file_h.readlines()
        new_lines = [ tuple(line.replace('\n', '').split(' ')) for line in old_lines ]
        train_test_split  = new_lines
        file_h.close()

        file_h    = open(os.path.join(dataset_dir , 'images.txt'))
        old_lines = file_h.readlines()
        new_lines = [ tuple(line.replace('\n', '').split(' ')) for line in old_lines ]
        images  = new_lines
        file_h.close()

        file_h    = open(os.path.join(dataset_dir , 'parts/part_locs.txt'))
        old_lines = file_h.readlines()
        new_lines = [ tuple(line.replace('\n', '').split(' ')) for line in old_lines ]
        part_locs  = new_lines
        file_h.close()
        # <image_id> <part_id> <x> <y> <visible> -- <visible> is 0 if the part is not visible in the image and 1 otherwise.

        assert len(images) == len(train_test_split) == len(image_class_labels) == len(images)
        assert all([ images[i][0] == train_test_split[i][0] == image_class_labels[i][0] == bounding_boxes[i][0] == str(i+1) for i in range(0, len(images))])

        train_file = open(os.path.join(dataset_dir, 'CUB_train_csv.txt'), 'w')
        test_file  = open(os.path.join(dataset_dir, 'CUB_test_csv.txt'), 'w')

        for i in range(0, len(images)):
            im_path  = images[i][1]
            in_train = train_test_split[i][1]
            bbox = tuple( [int( float( bounding_boxes[i][j] ) ) for j in range(1,5)] )
            b_cls = int(image_class_labels[i][1]) - 1

            for j in range(i*15, (i+1)*15):
                assert int(part_locs[j][0]) == int(images[i][0])
                if int(part_locs[j][4]) == 1:
                    kp_cls  = int(part_locs[j][1])
                    kp_x    = float(part_locs[j][2])
                    kp_y    = float(part_locs[j][3])

                    inst_string = ('%s,%d,%d,%d,%d,%d,%d, %f, %f\n' % (im_path,
                                                                        bbox[0],bbox[1],bbox[2],bbox[3],
                                                                        b_cls,
                                                                        kp_cls,
                                                                        kp_x, kp_y) )
                    if in_train == '1':
                    	train_file.write(inst_string)
                    else:
                    	test_file.write(inst_string)

        train_file.close()
        test_file.close()

    """
        Generate Chbyshev-based map given a keypoint location
    """
    def generate_kp_map_chebyshev(self, kp):

        assert kp[0] >= 0. and kp[0] <= 1., kp
        assert kp[1] >= 0. and kp[1] <= 1., kp
        kp_map = np.ndarray( (self.map_size, self.map_size) )

        kp = list(kp)

        kp[0] = kp[0] * self.map_size
        kp[1] = kp[1] * self.map_size

        for i in range(0, self.map_size):
            for j in range(0, self.map_size):
                kp_map[i,j] = max( np.abs(i - kp[0]), np.abs(j - kp[1]))

        # Normalize by dividing by the maximum possible value, which is self.IMG_SIZE -1
        kp_map = kp_map / (1. * self.map_size)

        # Revese map so that keypoint location is weighted more heavily
        kp_map = 1.0 - kp_map

        return kp_map


    def generate_validation(self, ratio = 0.1):
        assert ratio > (2.*self.num_classes/float(self.num_instances)) and ratio < 0.5
        random.seed(a = 2741998)
        valid_class     = copy.deepcopy(self)

        all_images      = list(set(self.image_paths))
        valid_size      = int(ratio * len(all_images))
        valid_image_i   = random.sample( range(0, len(all_images)), valid_size)
        set_valid_im_i  = set([all_images[i] for i in valid_image_i])


        train_instances = range(0, self.num_instances)
        valid_instances = [x for x in train_instances if self.image_paths[x] in set_valid_im_i]
        set_valid = set(valid_instances)
        train_instances = [x for x in train_instances if x not in set_valid]
        set_train = set(train_instances)

        train_size = len(train_instances)
        valid_size = len(valid_instances)
        # assert train_size == len(train_instances) and valid_size == len(valid_instances)

        print "Generating validation (size %d)" % valid_size
        valid_class.image_paths     = [ self.image_paths[i]     for i in sorted(set_valid) ]
        valid_class.bboxes          = [ self.bboxes[i]          for i in sorted(set_valid) ]
        valid_class.obj_class  = [ self.obj_class[i]  for i in sorted(set_valid) ]
        valid_class.flips           = [ self.flips[i]           for i in sorted(set_valid) ]
        valid_class.keypoint_cls    = [ self.keypoint_cls[i]    for i in sorted(set_valid) ]
        valid_class.keypoint_loc    = [ self.keypoint_loc[i]    for i in sorted(set_valid) ]
        valid_class.uids            = [ self.uids[i]            for i in sorted(set_valid) ]
        valid_class.num_instances   = valid_size


        print "Augmenting training (size %d)" % train_size
        self.image_paths     = [ self.image_paths[i]    for i in sorted(set_train) ]
        self.bboxes          = [ self.bboxes[i]         for i in sorted(set_train) ]
        self.obj_class  = [ self.obj_class[i] for i in sorted(set_train) ]
        self.flips           = [ self.flips[i]          for i in sorted(set_train) ]
        self.keypoint_cls    = [ self.keypoint_cls[i]   for i in sorted(set_train) ]
        self.keypoint_loc    = [ self.keypoint_loc[i]   for i in sorted(set_train) ]
        self.uids            = [ self.uids[i]           for i in sorted(set_train) ]
        self.num_instances   = train_size

        assert len(self.image_paths) == train_size

        return valid_class

    """
        Augment dataset -- currently just duplicate it with a flipped version of each instance
    """
    def augment(self):
        self.image_paths     = self.image_paths     + self.image_paths
        self.bboxes          = self.bboxes          + self.bboxes
        self.obj_class  = self.obj_class  + self.obj_class
        self.keypoint_cls    = self.keypoint_cls    + self.keypoint_cls
        self.keypoint_loc    = self.keypoint_loc    + self.keypoint_loc
        self.uids            = self.uids            + self.uids
        self.flip            = self.flip            + [True] * self.num_instances

        assert len(self.flip) == len(self.image_paths)
        self.num_instances = len(self.image_paths)
