import numpy as np
from scipy.ndimage.interpolation import rotate
import os
import imageio
import pandas as pd



def augment(sample, do_flip = False, do_rotate=False, do_swap = True):
    #  sample = imgs, target = target, bboxes = labels
    if do_rotate:
        validrot = False
        counter = 0
        while not validrot:
            # newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[2:4]).astype('float')
            # newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            # if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
            validrot = True
            # target = newtarget
            sample = rotate(sample, angle1, axes=(0, 1), reshape=False)





    if do_flip:
        # flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([0, 1]) * 2 - 1  # [0,0]水平垂直翻转，[1, 0]水平翻转，[0, 1]垂直翻转
        sample = np.ascontiguousarray(sample[::flipid[0], ::flipid[1]])
        # for ax in range(3):
        #     if flipid[ax]==-1:
        #         target[ax] = np.array(sample.shape[ax+1])-target[ax]
        #         bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]

    return sample

if __name__ == '__main__':
    # 读取BMP文件
    data_path = '/home/Chen_hongyu/cancer_classification_dataset/augment_BMP/test/0'
    output_path = '/home/Chen_hongyu/cancer_classification_dataset/augment_BMP/train/22'
    a = os.listdir(data_path)
    for filename in os.listdir(data_path):

        slices = imageio.imread(os.path.join(data_path, filename))
        imgs = np.array(slices)




        # 创建target(执行扩充函数)
        print('start--{}'.format(filename))
        sample = augment(sample=imgs, do_rotate=True)
        # a = sample[0,0,:,:]

        imageio.imsave(os.path.join(output_path, 'rot_' + filename ), sample)
        print('over--{}'.format(filename))



