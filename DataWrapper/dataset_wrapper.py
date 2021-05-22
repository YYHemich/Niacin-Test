import torch.utils.data as data
import numpy as np
from PIL import Image
import os


class CNNDataSet(data.Dataset):
    def __init__(self, path, label):
        super(CNNDataSet, self).__init__()
        self.x, self.y = self.generate_datapath(path, label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        filepath, label = self.x[index].strip(), self.y[index]
        img_array = np.load(filepath)
        imObj = Image.fromarray(img_array).resize((224, 224))
        img_array = np.array(imObj).swapaxes(0, 2) / 255.
        return img_array.astype('float32'), label

    def generate_datapath(self, path_li, path_label):
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


class LSTMDataSet(data.Dataset):
    def __init__(self, path, label):
        super(LSTMDataSet, self).__init__()
        self.x, self.y = path, label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        filepath, label = self.x[index].strip(), np.array(self.y[index], dtype='float32')
        person_featuures = np.load(filepath)
        return person_featuures.astype('float32'), label
