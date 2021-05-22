import torch.utils.data as data
import numpy as np
import os, sys
import torch
import torch.nn as nn
from torch import tensor
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

from DataWrapper import LSTMDataSet
from RNNModel import att_LSTM
from utils import MetricsTools, PathTools


def get_features_set(path, label):
    length = len(path)

    features_set = []
    for i in range(length):
        filepath = path[i]
        img = np.load(filepath)
        features_set.append(img.astype('float32'))

    return tensor(features_set), np.array(label, dtype='float32')


torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", required=True, help="Name of the directory to read features")
ap.add_argument("-e", "--epoch", default=15, type=int, help="Number of the training epochs")
ap.add_argument("-w", "--warmup", default=5, type=int, help="Number of the warmup epochs")
ap.add_argument("-l", "--lr", default=0.001, type=float, help="Number of the training epochs")
ap.add_argument("-bs", "--batch", default=256, type=int, help="Number of the training epochs")
ap.add_argument("-c", "--comment", default="None", help="Execute comment")
ap.add_argument("-hid", "--hidden", type=int, default=32, help="Dimension of the hidden state.")
ap.add_argument("-v", "--verbose", type=bool, default=False, help="Show detailed output.")
args = vars(ap.parse_args())

EPOCH = args['epoch']
WARNUP = args['warmup']
LR = args['lr']
BATCH_SIZE = args['batch']

if __name__ == '__main__':
    now_time = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())  # Sat Mar 28 22:24:24 2016
    print('---------- Start at %s ---------' % now_time)
    print('Execute: %s' % (sys.argv[0]))

    root = args['src']
    print('Features read from %s' % root)

    print('Execute comment: %s' % args['comment'])

    save_root = os.path.join('AttLSTMRuns')
    save_dir = os.path.join(save_root, PathTools.get_writer_dir(now_time))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    writer = SummaryWriter(save_dir)
    print('Result save in %s' % save_dir)

    with open(os.path.join(root, 'train', 'info.txt'), 'r', encoding='utf8') as f:
        train = f.read().strip().split('\n')
        train = [tuple(s.split('\t')) for s in train]
    train_set = LSTMDataSet([elem[0].strip() for elem in train], [int(elem[1].strip()) for elem in train])
    train_DataLoader = data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    print('Train data loaded.')

    with open(os.path.join(root, 'validate', 'info.txt'), 'r', encoding='utf8') as f:
        dev = f.read().strip().split('\n')
        dev = [tuple(s.split('\t')) for s in dev]
    dev_x, dev_y = get_features_set([elem[0].strip() for elem in dev], [int(elem[1].strip()) for elem in dev])

    with open(os.path.join(root, 'test', 'info.txt'), 'r', encoding='utf8') as f:
        test = f.read().strip().split('\n')
        test = [tuple(s.split('\t')) for s in test]
    test_x, test_y = get_features_set([elem[0].strip() for elem in test], [int(elem[1].strip()) for elem in test])

    lstm = att_LSTM(input_size=512, hidden_size=args['hidden'], time_step=21)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH, eta_min=0, last_epoch=-1)

    loss_func = nn.BCELoss()

    Eva = 0
    Ebestva = 20000
    epoch = 0
    decrease_cnt = 0
    threshold_alpha = 0.07

    validate_loss = []
    average_loss = []

    n_iter = 0

    for epoch in range(EPOCH + WARNUP):
        # train
        lstm.train()
        lstm.test_off()
        for i, (x, y) in enumerate(train_DataLoader):
            x = x.permute(1, 0, 2)  # reshape the input as (time_step, batch_size, input_dim) format
            output = lstm(x)
            loss = loss_func(output, y)
            if args['verbose']:
                print('Epoch %d Iteration %d Loss: %s' % (epoch + 1, i + 1, loss))
            writer.add_scalar('train loss', float(loss.data.numpy()), global_step=n_iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter += 1

        # evaluate
        lstm.eval()
        with torch.no_grad():
            va_out = lstm(dev_x.permute(1, 0, 2))
            pred_va = MetricsTools.prob2pred(va_out)
            acc_va = MetricsTools.scoring_acc(dev_y, pred_va)
            writer.add_scalar('val acc', float(acc_va), global_step=n_iter)
            val_loss = loss_func(va_out, tensor(dev_y))
            writer.add_scalar('validate loss', float(val_loss.data.numpy()), global_step=n_iter)
            Eva = (Eva * epoch + float(val_loss.data.numpy())) / (epoch + 1)
            writer.add_scalar('avg val loss', float(Eva), global_step=n_iter)

        if args['verbose']:
            print('Epoch %s now loss %s' % (epoch+1, val_loss))

        if epoch >= WARNUP:
            scheduler.step()
    print('Training complete.')

    print('Saving model ... at %s' % save_dir)
    state_dict = {'model': lstm.state_dict(), 'optimizer': optimizer.state_dict(), 'Epoch': EPOCH}
    model_save_pth = os.path.join(save_dir, 'checkpoint.dic')
    torch.save(state_dict, model_save_pth)
    print('Model is saved.')

    lstm.eval()
    lstm.test_on()
    with torch.no_grad():
        test_out = lstm(test_x.permute(1, 0, 2))
        test_possible = test_out.data.numpy().squeeze()
        pred_y = MetricsTools.prob2pred(test_out)
        acc = MetricsTools.scoring_acc(test_y, pred_y)

    lstm.saveAttentionList(os.path.join(save_dir, 'attention.npy'))

    print('Acc on test:', acc)
    conf_matrix = MetricsTools.confusion_matrix(test_y, pred_y)
    sen = MetricsTools.sensitivity(conf_matrix)
    spe = MetricsTools.specificity(conf_matrix)
    print('sensitivity', sen)
    print('specificity', spe)
    print('confusion matrix', conf_matrix)
