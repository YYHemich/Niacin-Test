import torchvision
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import os, sys
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import argparse

from DataWrapper import CNNDataSet
from utils import MetricsTools, PathTools


ap = argparse.ArgumentParser()
ap.add_argument("--src", default='datasets')
ap.add_argument("--epoch", default=25, type=int, help="Number of the training epochs")
ap.add_argument("--warmup", default=5, type=int, help="Number of the warmup epochs")
ap.add_argument("--lr", default=0.001, type=float, help="Number of the training epochs")
ap.add_argument("--batch", default=512, type=int, help="Number of the training epochs")
ap.add_argument("--pretrained", type=bool, default=False, help="Use ImageNet pre-train image.")
ap.add_argument("--comment", default="None", help="Execute comment")
ap.add_argument("--verbose", type=bool, default=False, help="Show detailed output.")
ap.add_argument("--device", default="cuda")
args = vars(ap.parse_args())

EPOCH = args['epoch']
WARMUP = args['warmup']
LR = args['lr']
BATCH_SIZE = args['batch']
DEVICE = args['device']

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)


if __name__ == '__main__':
    now_time = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())  # Sat Mar 28 22:24:24 2016
    print('---------- Start at %s ---------' % now_time)
    print('Execute: %s' % (sys.argv[0]))
    root = 'ResNetRuns'
    save_dir = os.path.join(root, PathTools.get_writer_dir(now_time))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(save_dir)
    print('Result save in %s' % save_dir)

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

    train_data = CNNDataSet(path_train, label_train)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_data = CNNDataSet(path_dev, label_dev)
    dev_loader = data.DataLoader(dataset=dev_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    if args['verbose']:
        print('Data loaded.')

    cnn_model = torchvision.models.resnet18(num_classes=2)
    cnn_model.to(DEVICE)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    loss_func = nn.CrossEntropyLoss()

    Eva = 0
    n_iter = 0
    for epoch in range(EPOCH + WARMUP):
        # train
        cnn_model.train()
        for i, (x, y) in enumerate(train_loader):
            y = y.long().to(DEVICE)
            x = x.to(DEVICE)
            output = cnn_model(x)
            loss = loss_func(output, y)
            if args['verbose']:
                print('Epoch %d Iteration %d Loss: %s' % (epoch + 1, i + 1, loss))
            writer.add_scalar('train loss', float(loss.data), global_step=n_iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter += 1

        # evaluate
        cnn_model.eval()
        pred_va = np.array([])
        ground_truth = np.array([])
        val_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(dev_loader):
                x = x.to(DEVICE)
                va_out = cnn_model(x).to('cpu')
                pred_va = np.hstack([pred_va, MetricsTools.get_predict_label(va_out)])
                ground_truth = np.hstack([ground_truth, y.data.numpy()])
                val_loss += loss_func(va_out, y.long())
            val_loss /= len(dev_loader)
            writer.add_scalar('validate loss', float(val_loss.data), global_step=n_iter)
            Eva = (Eva * epoch + float(val_loss)) / (epoch + 1)
            writer.add_scalar('avg val loss', float(Eva), global_step=n_iter)
            acc_va = MetricsTools.scoring_acc(ground_truth, pred_va)
            writer.add_scalar('val acc', float(acc_va), global_step=n_iter)
        if args['verbose']:
            print('Epoch %s now loss %s' % (epoch, val_loss))

        if epoch >= WARMUP:
            scheduler.step()
    print('Training complete.')

    print('Saving model ... at %s' % save_dir)
    state_dict = {'model': cnn_model.state_dict(), 'optimizer': optimizer.state_dict(), 'Epoch': EPOCH}
    model_save_pth = os.path.join(save_dir, 'checkpoint.dic')
    torch.save(state_dict, model_save_pth)
    print('Model is saved.')

    test_data = CNNDataSet(path_test, label_test)
    test_loader = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    cnn_model.eval()
    pred_test = np.array([])
    test_ground_truth = np.array([])
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(DEVICE)
            test_out = cnn_model(x).to('cpu')
            pred_test = np.hstack([pred_test, MetricsTools.get_predict_label(test_out)])
            test_ground_truth = np.hstack([test_ground_truth, y.data.numpy()])
        acc = MetricsTools.scoring_acc(test_ground_truth, pred_test)

    print('Acc on all test:', acc)
    conf_matrix = MetricsTools.confusion_matrix(test_ground_truth, pred_test)
    sen = MetricsTools.sensitivity(conf_matrix)
    spe = MetricsTools.specificity(conf_matrix)
    print('sensitivity', sen)
    print('specificity', spe)
    print('confusion matrix', conf_matrix)
