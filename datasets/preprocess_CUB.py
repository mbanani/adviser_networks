from PIL import Image

import os
import numpy as np
import time
import pandas
import copy
import random

from IPython import embed


def pandas_csv_to_info(csv_path):
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

CUB_dir  = '/z/home/mbanani/datasets/CUB_200_2011'
img_dir  = '/z/home/mbanani/datasets/CUB_200_2011/images'
new_dir  = '/z/home/mbanani/datasets/CUB_preprocessed_227'

img_size = 227

# csv_path = os.path.join(CUB_dir, 'CUB_train_csv.txt')
csv_path = os.path.join(CUB_dir, 'CUB_test_csv.txt')


im_paths, _, bbs, _, _ = pandas_csv_to_info(csv_path)

processed_images = []

for i in range(0, len(im_paths)):
    im      = im_paths[i]
    bbox    = bbs[i]
    new_path = os.path.join(new_dir, im)

    if i % 100 == 0:
        print i, " of ", len(im_paths)

    if not(os.path.exists(new_path)):
        processed_images.append(im)
        path = os.path.join(img_dir, im)
        with open(path, 'rb') as f:
            with Image.open(f) as img:

                img = img.convert('RGB')
                img = img.crop(box=bbox)
                img = img.resize( (img_size, img_size), Image.LANCZOS)

                new_path = os.path.join(new_dir, im)

                if not os.path.exists(os.path.dirname(new_path)):
                    os.makedirs(os.path.dirname(new_path))

                img.save(new_path)
