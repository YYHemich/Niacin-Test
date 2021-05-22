import torchvision
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import os, sys
import torch
import pickle
from torch import tensor
import time
import argparse


class FeaturesExtractor:
    def __init__(self, encoder):
        self.encoder = encoder
        self.encoder.eval()
        self.in_dict = None
        self.sort_dict = None

    def set_reference(self, in_src, sort_src):
        with open(in_src, 'rb') as f:
            self.in_dict = pickle.load(f)
        with open(sort_src, 'rb') as f:
            self.sort_dict = pickle.load(f)

    def sort(self, files, dir_name):
        if self.in_dict[dir_name]:
            tmp_files = eval(self.sort_dict[dir_name])
            files.sort()
        else:
            try:
                tmp_files = [int(s[:-4]) for s in files]
            except Exception:
                tmp_files = [s[:-4] for s in files]
        sort_li = list(zip(tmp_files, files))
        sort_li.sort(key=lambda x: x[0])
        files = [s[1] for s in sort_li]
        return files

    def load(self, src):
        img = np.load(src)
        imObj = Image.fromarray(img).resize((224, 224))
        img = np.array(imObj).swapaxes(0, 2) / 255.
        return img.astype('float32')

    def extract(self, src):
        dir_name = src.strip().split('/')[-1]
        files = None
        root = None
        for root_name, dir_li, file_li in os.walk(src):
            if not dir_li:
                files = file_li
                root = root_name
        if self.sort_dict is not None and self.in_dict is not None:
            files = self.sort(files, dir_name)

        if len(files) < 21:
            files += ['None'] * (21 - len(files))
        else:
            files = files[:21]

        img_batch = []
        for s in files:
            if s == 'None':
                img = np.zeros((3, 224, 224), dtype='float32')
            else:
                img = self.load('%s/%s' % (root, s))
            img_batch.append(img)
        img_batch = tensor(img_batch)

        with torch.no_grad():
            features = self.encoder(img_batch)
        return features.data.numpy()

    @staticmethod
    def save_feature(save_name, root, file):
        if not os.path.exists(root):
            os.makedirs(root)
        save_pth = '%s/%s.npy' % (root, save_name)
        np.save(save_pth, file)


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="path of model")
ap.add_argument("--savedir", required=True, help="Name of the directory to save")
ap.add_argument("--src", default="datasets", help="path of the data.")
ap.add_argument("--classes", type=int, default=2, help="The number of the output layer of the pre-trained cnn.")
ap.add_argument("--comment", default="None", help="Execute comment")
args = vars(ap.parse_args())


if __name__ == '__main__':
    print('----------------- info ---------------')
    print('Date: %s' % time.asctime(time.localtime(time.time())))
    print('Execute: %s' % (sys.argv[0]))
    print('Extractor is from: %s' % args['model'])
    print('Execute comment: %s' % args['comment'])

    root = os.path.join("ImgFeatures", args['savedir'])

    print('Features saved in %s' % root)

    neg_root = os.path.join(args['src'], 'negative')
    neg_path = [os.path.join(neg_root, directory) for directory in sorted(os.listdir(neg_root))]
    neg_label = [0] * len(neg_path)

    pos_root = os.path.join(args['src'], 'positive')
    pos_path = [os.path.join(pos_root, directory) for directory in sorted(os.listdir(pos_root))]
    pos_label = [1] * len(pos_path)

    total_path = pos_path + neg_path
    total_label = pos_label + neg_label

    path_train_va, path_test, label_train_va, label_test = train_test_split(total_path, total_label,
                                                                            test_size=0.3,
                                                                            random_state=42,
                                                                            stratify=total_label,
                                                                            shuffle=True)

    path_train, path_dev, label_train, label_dev = train_test_split(path_train_va, label_train_va,
                                                                    test_size=0.1,
                                                                    random_state=38,
                                                                    stratify=label_train_va,
                                                                    shuffle=True)

    cnn_model = torchvision.models.resnet18(pretrained=False, num_classes=args['classes'])
    state_dict = torch.load(args['model'])
    cnn_model.load_state_dict(state_dict['model'])
    del cnn_model.fc
    cnn_model.fc = lambda x: x
    cnn_model.eval()
    extractor = FeaturesExtractor(cnn_model)
    extractor.set_reference(in_src='in_dict.pkl', sort_src='sort_dict.pkl')
    print('Model is ready.')

    train_root = os.path.join(root, 'train')
    test_root = os.path.join(root, 'test')
    dev_root = os.path.join(root, 'validate')

    for i in [train_root, test_root, dev_root]:
        if not os.path.exists(i):
            os.makedirs(i)

    with open(os.path.join(train_root, 'info.txt'), 'w', encoding='utf8') as f:
        for i, s in enumerate(path_train):
            features_out = extractor.extract(s)
            label = label_train[i]
            name = s.split('/')[-1]
            FeaturesExtractor.save_feature(name, train_root, features_out)
            f.write('%s/%s.npy\t%s\n' % (train_root, name, label))

    with open(os.path.join(test_root, 'info.txt'), 'w', encoding='utf8') as f:
        for i, s in enumerate(path_test):
            features_out = extractor.extract(s)
            label = label_test[i]
            name = s.split('/')[-1]
            FeaturesExtractor.save_feature(s.split('/')[-1], test_root, features_out)
            f.write('%s/%s.npy\t%s\n' % (test_root, name, label))

    with open(os.path.join(dev_root, 'info.txt'), 'w', encoding='utf8') as f:
        for i, s in enumerate(path_dev):
            features_out = extractor.extract(s)
            label = label_dev[i]
            name = s.split('/')[-1]
            FeaturesExtractor.save_feature(s.split('/')[-1], dev_root, features_out)
            f.write('%s/%s.npy\t%s\n' % (dev_root, name, label))
    print('Extraction finish.')
