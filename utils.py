import torch
from torch import tensor
import numpy as np
from PIL import Image
import os


class MetricsTools:
    @staticmethod
    def get_predict_label(output):
        return torch.max(output, 1)[1].data.numpy().squeeze()

    @staticmethod
    def prob2pred(x):
        pred = []
        for p in x:
            if p > 0.5:
                pred += [1]
            else:
                pred += [0]
        return pred

    @staticmethod
    def scoring_acc(target, output):
        s = 0
        for i in range(len(target)):
            if int(target[i]) == output[i]:
                s += 1
        return s / len(target)

    @staticmethod
    def confusion_matrix(target, output):
        cm = np.zeros((2, 2), dtype=np.int)
        for i in range(len(target)):
            if target[i] == 1 and output[i] == 1:  # TP
                cm[0, 0] += 1
            elif target[i] == 0 and output[i] == 1:  # FP
                cm[1, 0] += 1
            elif target[i] == 1 and output[i] == 0:  # FN
                cm[0, 1] += 1
            else:  # TN
                cm[1, 1] += 1
        return cm

    @staticmethod
    def sensitivity(cm_arr):
        return cm_arr[0, 0] / (cm_arr[0, 0] + cm_arr[0, 1])

    @staticmethod
    def specificity(cm_arr):
        return cm_arr[1, 1] / (cm_arr[1, 1] + cm_arr[1, 0])


class PathTools:
    @staticmethod
    def get_writer_dir(now_time):
        dir_name = '-'.join(now_time.split(':'))
        dir_name = '_'.join(dir_name.split(' ')[1:])
        return dir_name

    @staticmethod
    def generate_datapath(path_li, path_label):
        filepath = []
        img_label = []
        for i, root in enumerate(path_li):
            label = path_label[i]
            length = 0
            for filename in os.listdir(root):
                filepath.append(os.path.join(root, filename))
                length += 1
            img_label += [label] * length
        return filepath, img_label

    @staticmethod
    def get_img_array(path, _range=None):
        if _range is None:
            length = len(path)
        else:
            length = _range

        img_set = []
        for i in range(length):
            filepath = path[i]
            img = np.load(filepath)
            imObj = Image.fromarray(img).resize((224, 224))
            img = np.array(imObj).swapaxes(0, 2) / 255.
            img_set.append(img.astype('float32'))

        return tensor(img_set)
