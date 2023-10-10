import numpy as np
from scipy.ndimage.interpolation import rotate
import os
import imageio
import pandas as pd
import csv



def augment(sample, bboxes, do_flip = False, do_rotate=False, do_swap = True):
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
            sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
            for box in bboxes[0]:
                # a = box[2:5:2]
                # b = box[3:6:2]
                # box[3:5] = np.dot(rotmat, box[3:5] - size / 2) + size / 2
                # box[2:6:3] = np.dot(rotmat, box[2:6:3] - size / 2) + size / 2
                """换个思路，把坐标中心点记录下来，计算旋转后的坐标中心点，由于
                直径是不变的，计算出旋转后的坐标中心点后，再将直径/2分别加减
                在中心点，以生成min和max"""
                diameter1 = box[3] - box[2]
                center_y = size[1] - ((box[3] - box[2]) // 2 + box[2])  # (ymax-ymin // 2) + ymin
                center_x = (box[5] - box[4]) // 2 + box[4]
                rotate_x = ((center_x - size[0] // 2) * (np.cos(angle1 / 180 * np.pi))
                            - (center_y - size[0] // 2) * (np.sin(angle1 / 180 * np.pi))) \
                           + size[0] /2
                rotate_y = size[1] - \
                           (((center_x - size[1] // 2) * (np.sin(angle1 / 180 * np.pi))
                             + (center_y - size[1] // 2) * (np.cos(angle1 / 180 * np.pi)))
                            + size[1] /2)
                xmin = rotate_x - diameter1 // 2
                xmax = rotate_x + diameter1 // 2
                ymin = rotate_y - diameter1 // 2
                ymax = rotate_y + diameter1 // 2
                zmin = box[0]
                zmax = box[1]
                box[0] = xmin
                box[1] = xmax
                box[2] = ymin
                box[3] = ymax
                box[4] = zmin
                box[5] = zmax




    if do_flip:
        # flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, 0, 0]) * 2 - 1  # [1,0,0]水平垂直翻转，[1,0,1]水平翻转，[1,1,0]垂直翻转
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        # for ax in range(3):
        #     if flipid[ax]==-1:
        #         target[ax] = np.array(sample.shape[ax+1])-target[ax]
        #         bboxes[:,ax]= np.array(sample.shape[ax+1])-bboxes[:,ax]
        for i in range(3):
            if flipid[i] == -1:
                tem = [0, 2, 4]
                ax = tem[i]
                # target[ax] = np.array(sample.shape[i + 1]) - target[ax]
                bboxes[0, :, ax] = np.array(sample.shape[i + 1]) - bboxes[0, :, ax]
                # target[ax + 1] = np.array(sample.shape[i + 1]) - target[ax + 1]
                bboxes[0, :, ax + 1] = np.array(sample.shape[i + 1]) - bboxes[0, :, ax + 1]   # x和y的坐标转换后，min和max互换
                if ax == 2:  # y在变，即上下翻转
                    bboxes[0, :, [0, 1, 2, 3, 4, 5]] = bboxes[0, :, [0, 1, 3, 2, 4, 5]]
                elif ax == 4:  # x在变，即左右翻转
                    bboxes[0, :, [0, 1, 2, 3, 4, 5]] = bboxes[0, :, [0, 1, 2, 3, 5, 4]]
        bboxes[0, :, [0, 1, 2, 3, 4, 5]] = bboxes[0, :, [4, 5, 2, 3, 0, 1]]

    return sample, bboxes

if __name__ == '__main__':
    # 读取BMP文件
    data_path = '/home/Chen_hongyu/lung_nodule_dataset/BMP_ALL/imgs'
    output_path = '/home/Chen_hongyu/lung_nodule_dataset/BMP_expand/augment_img_rotate'
    a = os.listdir(data_path)
    for filename in os.listdir(data_path):
        # if filename == '55':
        #     continue
        lines = os.listdir(os.path.join(data_path, filename))
        lines = sorted(lines)
        slice_files = [os.path.join(data_path, filename, s) for s in lines]
        slices = [imageio.imread(s) for s in slice_files]
        imgs = np.array(slices)
        imgs = imgs[np.newaxis, ...]

        # 读取csv
        csv_dir = '/home/Chen_hongyu/lung_nodule_dataset/BMP_expand/all_anno_3D.csv'
        annos_all = pd.read_csv(csv_dir)
        annos = annos_all[annos_all['index'] == int(filename)]
        temp_annos = []
        labels = []

        image_name = annos['image']
        index_name = annos['index']
        image_name = np.array(image_name)
        for i in range(image_name.shape[0]):
            image_name[i] = 'rot_' + image_name[i]  # 改名字！！！！
        index_name = np.array(index_name)
        image_name = image_name.reshape(-1, 1)
        index_name = index_name.reshape(-1, 1)

        if len(annos) > 0:
            for index in range(len(annos)):
                anno = annos.iloc[index]
                temp_annos.append(
                    [anno['z_min'], anno['z_max'], anno['y_min'], anno['y_max'], anno['x_min'], anno['x_max']])
        l = np.array(temp_annos)
        if np.all(l == 0):
            l = np.array([])
        labels.append(l)
        labels = np.array(labels)

        # 创建target(执行扩充函数)
        print('start--{}'.format(filename))
        sample, bboxes = augment(sample=imgs, bboxes=labels, do_flip=False, do_rotate=True)
        # a = sample[0,0,:,:]
        idx = 0
        for i in lines:
            if not os.path.exists(os.path.join(output_path, filename)):
                os.mkdir(os.path.join(output_path, filename))
            imageio.imsave(os.path.join(output_path, filename, i), sample[0,idx,:,:])
            idx = idx + 1

        # 计算center坐标
        center_list = []
        for i in range(bboxes.shape[1]):
            a = bboxes[0,i]
            center = []
            x_center = (bboxes[0, i, 0] + bboxes[0, i, 1]) // 2
            y_center = (bboxes[0, i, 2] + bboxes[0, i, 3]) // 2
            z_center = (bboxes[0, i, 4] + bboxes[0, i, 5]) // 2
            diameter = bboxes[0, i, 1] - bboxes[0, i, 0]
            center.append(x_center)
            center.append(y_center)
            center.append(z_center)
            center.append(diameter)
            center_list.append(center)
        center_list = np.array(center_list)
        bboxes_new = bboxes[0]
        bboxes_new = np.concatenate((bboxes_new, center_list), axis=1)
        bboxes_new = np.concatenate((image_name, bboxes_new), axis=1)
        bboxes_new = np.concatenate((index_name, bboxes_new), axis=1)



        # 保存csv
        for j in range(bboxes_new.shape[0]):
            with open('/home/Chen_hongyu/lung_nodule_dataset/BMP_expand/augment_rotate.csv',
                    'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(bboxes_new[j])
        # pd.DataFrame(bboxes_new).to_csv(os.path.join(output_path, filename, 'augment.csv'))
        print('over--{}'.format(filename))
