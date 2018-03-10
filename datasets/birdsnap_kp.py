
from PIL import Image
from torchvision import transforms

import torch
import os
import numpy as np
import time
import pandas
import copy, random

from IPython import embed


class birdsnap_kp(torch.utils.data.Dataset):

    def __init__(self, csv_path, dataset_dir, preprocessed_root = None, image_size = 227, map_size=46, transform = None):


        self.keypoint_list = ['left_cheek', 'right_cheek',
                             'left_leg', 'right_leg',
                             'left_eye', 'right_eye',
                             'left_wing', 'right_wing',
                             'tail',
                             'throat',
                             'breast',
                             'nape',
                             'beak',
                             'back',
                             'crown']


        if not os.path.isfile(os.path.join(dataset_dir, 'birdsnap_train.txt')):
            print "CSV File does not exist! -- creating CSV file for training and testing"
            self.create_csv(dataset_dir)


        i = 0
        curr_time = time.time()

        s_im_paths, s_img_class, s_img_class_id, s_bbs, s_kps = self.pandas_csv_to_info(csv_path)

        size_dataset = len(s_im_paths)
        print "csv file length: ", size_dataset

        assert len(s_im_paths) == len(s_img_class) == len(s_img_class_id) == len(s_bbs) == len(s_kps), "Error: pandas_csv_to_info outputting elements of different size."

        ommited = 0

        for i in range(0, len(s_kps)):
            for j in range(0, 15):
                # I think 2*j because it's stored as kpx kpy in order ? TODO verify this
                if s_kps[i][2*j] != -1:
                    kpx = float(s_kps[i][2*j]) / float(s_bbs[i][2] - s_bbs[i][0])
                    kpy = float(s_kps[i][2*j + 1]) / float(s_bbs[i][3] - s_bbs[i][1])
                    if kpx > 1. or kpx < 0. or kpy > 1. or kpy < 0.:
                        # print "Incorrect kp values (", kpx, ", ", kpy, "). ommited."
                        ommited += 1
                    else:
                        im_paths.append(s_im_paths[i])
                        bbs.append(s_bbs[i])
                        im_cls.append(s_img_class[i])
                        im_cls_id.append(s_img_class_id[i])
                        # NOTE: current CSV generations gives KP relative to bbox!!!
                        kp_loc.append([kpx, kpy])
                        kp_cls.append(j)


        bboxes = [tuple(bbox) for bbox in bboxes]

        print "Dataset loaded in ", time.time() - curr_time, " secs."
        print "Dataset size: ", len(images)
        print ommited, " kp-image instances ommited due to keypoint lying outside of bounding box"

        if len(images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))


        if preprocessed_root:
            self.preprocessed = True
            self.image_root = preprocessed_root
        else:
            self.preprocessed = False
            self.image_root = os.path.join(dataset_dir, "images")

        self.image_paths    = im_paths
        self.bboxes         = bbs
        self.image_class    = im_cls
        self.image_class_id = im_cls_id
        self.keypoint_loc   = kp_loc
        self.keypoint_cls   = kp_cls
        self.flips          = [False] * len(im_paths)

        self.loader         = self.pil_loader
        self.img_size       = image_size
        self.num_kps        = len(self.keypoint_list)
        self.map_size       = map_size
        self.num_classes    = 500
        self.num_instances  = len(self.image_paths)

        # Normalization as instructed from pyTorch documentation
        self.transform = transform or transforms.Compose([  transforms.ToTensor(),
                                                            transforms.Normalize(   mean=(0.485, 0.456, 0.406),
                                                                                    std=(0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # Load and transform image
        path = os.path.join(self.image_root, self.image_paths[index])

        im_class_id = self.image_class_id[index]
        bbox        = self.bboxes[index]
        kp_loc      = self.keypoint_loc[index]
        kp_cls      = self.keypoint_cls[index]
        flip        = self.flips[index]


        # ASSUMING keypoints are in range [0,1] x [0,1]
        # kept because it's useful for inception transform
        inc_bbox    = [0.0, 0.0, 1.0, 1.0]

        img, kp_loc = self.loader(im_path, bbox, flip, inc_bbox, kp_loc)

        if self.transform is not None:  img = self.transform(img)

        # Generate keypoint map image, and kp class vector
        kpc_vec  = np.zeros( (15) )
        kpc_vec[kp_cls] = 1
        kpm_map  = self.generate_kp_map_chebyshev(kp_loc)

        kp_class   = torch.from_numpy(kpc_vec).float()
        kp_map     = torch.from_numpy(kpm_map).float()

        # # Generate a unique ID
        uid = self.image_paths[index] + '_objc' + str(im_class_id) + '_kpc' + str(kp_cls)

        return img, im_class_id, kp_map, kp_class , uid

    def __len__(self):
        return self.num_instances

    def pandas_csv_to_info(self, csv_path):
        df = pandas.read_csv(csv_path, sep=',')
        data = df.values

        data_split = np.split(data, [0, 1, 2, 3, 7, 37], axis=1)
        del(data_split[0])

        image_paths  = np.squeeze(data_split[0]).tolist()
        img_class    = data_split[1].tolist()
        img_class_id = np.squeeze(data_split[2]).tolist()
        bbox         = data_split[3].tolist()
        kp_locs      = np.squeeze(data_split[4]).tolist()

        return image_paths, img_class, img_class_id, bbox, kp_locs

    def pil_loader(self, path, bbox, flip, inc_bbox):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

                if not self.preprocessed:
                    n_bbox = [0] * 4

                    # bbox ordering left, upper, right , lower
                    o_width     = bbox[2] - bbox[0]
                    o_height    = bbox[3] - bbox[1]
                    n_bbox[0] = bbox[0] + o_width * inc_bbox[0]
                    n_bbox[2] = bbox[0] + o_width * inc_bbox[2]

                    n_bbox[1] = bbox[1] + o_height * inc_bbox[1]
                    n_bbox[3] = bbox[1] + o_height * inc_bbox[3]

                    img = img.crop(box=n_bbox)
                    img = img.resize( (self.img_size, self.img_size), Image.LANCZOS)
                else:
                    if not (img.size[0] == img.size[1] == self.img_size):
                        print "Error: File (%s) was not preprocessed correctly. Exiting." % (path)
                        exit()
                    imSize  = img.size[0]
                    bbox    = tuple([self.img_size * inc_bbox[i] for i in range(0, 4)] )
                    img     = img.crop(box=bbox)
                    img     = img.resize( (self.img_size, self.img_size), Image.LANCZOS)

                if flip:
                    img.transpose(Image.FLIP_LEFT_RIGHT)
                return img

    def create_csv(self, dataset_path):

        train_file = open(os.path.join(dataset_path, 'birdsnap_train.txt'), 'w')
        valid_file = open(os.path.join(dataset_path, 'birdsnap_test.txt'), 'w')

        image_list = self.read_list_of_dicts(os.path.join(dataset_path, "images.txt"))
        species_list = self.read_list_of_dicts(os.path.join(dataset_path, "species.txt"))
        image_dir = os.path.join(dataset_path, "images")
        types_dir = list(os.listdir(image_dir))
        classes_dict = {x['dir']: int(x['id']) for x in species_list}

        test_rows = self.read_list_of_dicts(os.path.join(dataset_path, "test_images.txt"))
        test_imgs = [x['path'] for x in test_rows]

        num_instances = 0
        num_skipped = 0
        num_train = 0
        num_test  = 0

        print "Number of instances: ", len(image_list)
        for instance in image_list:
            img_path   = instance['path']
            full_img_path = os.path.join(image_dir, img_path)
            if os.path.isfile(full_img_path):
                num_instances += 1

                img_class  = img_path.split('/', 1)[0].lower()
                img_cls_id = classes_dict[img_class]

                bbox = [ int(instance['bb_x1']),
                         int(instance['bb_y1']),
                         int(instance['bb_x2']),
                         int(instance['bb_y2'])
                        ]

                kp_locs = [-1] * 30
                i = 0

                for kp in self.keypoint_list:
                    if instance[kp + '_x'] != 'null':
                        kp_locs[i]      = int(instance[kp + '_x']) - bbox[0]
                        kp_locs[i + 1]  = int(instance[kp + '_y']) - bbox[1]
                    i = i + 2

                kp_string = self.keypoint_instance_info_to_str(img_path, img_class, img_cls_id, bbox, kp_locs)

                if img_path in test_imgs:
                    num_test += 1
                    train_file.write(kp_string)
                else:
                    num_train += 1
                    valid_file.write(kp_string)

            else:
                num_skipped += 1

            if num_instances % 1000 == 0:
                print "Processed %d (train: %d, test: %d) images and skipped %d images out of %d images." % (num_instances,
                                                                                                             num_train,
                                                                                                             num_test,
                                                                                                             num_skipped, len(image_list))

        train_file.close()
        valid_file.close()

    def keypoint_instance_info_to_str(self, rel_image_path, bird_class, bird_class_id, bbox, kp_locs):
        return '%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n' % (
                            rel_image_path,
                            bird_class,
                            bird_class_id,
                            bbox[0], bbox[1], bbox[2], bbox[3],
                            kp_locs[0], kp_locs[1], kp_locs[2], kp_locs[3],kp_locs[4],
                            kp_locs[5], kp_locs[6], kp_locs[7], kp_locs[8], kp_locs[9],
                            kp_locs[10], kp_locs[11],kp_locs[12], kp_locs[13], kp_locs[14],
                            kp_locs[15], kp_locs[16], kp_locs[17], kp_locs[18], kp_locs[19],
                            kp_locs[20], kp_locs[21], kp_locs[22], kp_locs[23], kp_locs[24],
                            kp_locs[25], kp_locs[26], kp_locs[27], kp_locs[28], kp_locs[29])

    def read_list_of_dicts(self, path):
        rows = []
        with open(path, 'r') as fin:
            fieldnames = fin.readline().strip().split('\t')
            for line in fin:
                vals = line.strip().split('\t')
                assert len(vals) == len(fieldnames)
                rows.append(dict(zip(fieldnames, vals)))
            return rows

    # TODO figure out how to generate 13x13 from the begining, while accounting for maxpool
    # effects
    """
        Generate Chbyshev-based map given a keypoint location
    """
    def generate_kp_map_chebyshev(self, kp):

        assert kp[0] >= 0. and kp[0] <= 1., kp
        assert kp[1] >= 0. and kp[1] <= 1., kp
        kp_map = np.ndarray( (self.map_size, self.map_size) )


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
        valid_class.image_class_id  = [ self.image_class_id[i]  for i in sorted(set_valid) ]
        valid_class.flips           = [ self.flips[i]           for i in sorted(set_valid) ]
        valid_class.keypoint_cls    = [ self.keypoint_cls[i]    for i in sorted(set_valid) ]
        valid_class.keypoint_loc    = [ self.keypoint_loc[i]    for i in sorted(set_valid) ]
        valid_class.num_instances   = valid_size


        print "Augmenting training (size %d)" % train_size
        self.image_paths     = [ self.image_paths[i]    for i in sorted(set_train) ]
        self.bboxes          = [ self.bboxes[i]         for i in sorted(set_train) ]
        self.image_class_id  = [ self.image_class_id[i] for i in sorted(set_train) ]
        self.flips           = [ self.flips[i]          for i in sorted(set_train) ]
        self.keypoint_cls    = [ self.keypoint_cls[i]   for i in sorted(set_train) ]
        self.keypoint_loc    = [ self.keypoint_loc[i]   for i in sorted(set_train) ]
        self.num_instances   = train_size

        assert len(self.image_paths) == train_size

        return valid_class

    """
        Augment dataset -- currently just duplicate it with a flipped version of each instance
    """
    def augment(self):
        self.image_paths     = self.image_paths     + self.image_paths
        self.bboxes          = self.bboxes          + self.bboxes
        self.image_class_id  = self.image_class_id  + self.image_class_id
        self.keypoint_cls    = self.keypoint_cls    + self.keypoint_cls
        self.keypoint_loc    = self.keypoint_loc    + self.keypoint_loc
        self.flip            = self.flip            + [True] * self.num_instances

        assert len(self.flip) == len(self.image_paths)
        self.num_instances = len(self.image_paths)
