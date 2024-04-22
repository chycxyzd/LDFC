import numpy as np
from scipy.ndimage.interpolation import rotate
import os
import imageio
import pandas as pd
import argparse


def augment(sample, do_flip = False, do_rotate=False, do_swap = True, angle=0):
    #  sample = imgs, target = target, bboxes = labels
    if do_rotate:
        validrot = False
        counter = 0
        while not validrot:
            # newtarget = np.copy(target)
            # angle1 = np.random.rand() * 180
            angle1 = angle
            size = np.array(sample.shape[2:4]).astype('float')
            # newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            # if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
            validrot = True
            # target = newtarget
            sample = rotate(sample, angle1, axes=(0, 1), reshape=False)
    if do_flip:
        # flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, 0]) * 2 - 1  # [0,0]水平垂直翻转，[1, 0]水平翻转，[0, 1]垂直翻转
        sample = np.ascontiguousarray(sample[::flipid[0], ::flipid[1]])
        # for ax in range(3):
        #     if flipid[ax]==-1:
        #         target[ax] = np.array(sample.shape[ax+1])-target[ax]
        #         bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]

    return sample


def main(args):
    # read BMP
    data_path = args.data_path
    output_path = args.output_path
    a = os.listdir(data_path)
    for filename in os.listdir(data_path):
        slices = imageio.imread(os.path.join(data_path, filename))
        imgs = np.array(slices)
        for i in range(2):
            print('start--{}'.format(filename))
            sample = augment(sample=imgs, do_rotate=True, do_flip=False, angle=i * 180 + 90)
            imageio.imsave(os.path.join(output_path, str(i) + 'rot_' + filename), sample)
            print('over--{}'.format(filename))

if __name__ == '__main__':
    "data-path is the data storage path" \
    "output-path is the output path of the processed data" \
    "csv-dir is the path to the csv file" \
    "csv-save is the save path of the csv file"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        default='/home/Chen_hongyu/lung_nodule_dataset/BMP_ALL/imgs')
    parser.add_argument('--output-path', type=str,
                        default='/home/Chen_hongyu/lung_nodule_dataset/BMP_expand/augment_img_rotate')
    opt = parser.parse_args()

    main(opt)
