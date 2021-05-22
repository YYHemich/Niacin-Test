import torch.nn as nn
import numpy as np


class att_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, time_step):
        super(att_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.f_att = nn.Linear(time_step, time_step)
        self.att_softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(time_step * hidden_size, 1)
        self.output_activation = nn.Sigmoid()

        self.test_mode = False
        self.attention_list = []

    def forward(self, x):
        x, (_hn, _cn) = self.lstm(x)
        x = x.permute(1, 2, 0)
        _, h_num, time_step = x.size()
        att = self.f_att(x)
        att = self.att_softmax(att)
        if self.test_mode:
            att_tmp = np.mean(att.data.numpy(), axis=1)
            self.attention_list.append(att_tmp)
        x_att = att * x
        x_att = x_att.view(-1, time_step * h_num)
        output = self.output_activation(self.fc(x_att))
        return output

    def test_off(self):
        self.test_mode = False
        self.attention_list = []

    def test_on(self):
        self.test_mode = True

    def saveAttentionList(self, saveDir):
        if not self.attention_list:
            print("No attention information recorded.")
            return

        res = self.attention_list[0].copy()
        for i in range(1, len(self.attention_list)):
            res = np.vstack((res, self.attention_list[i]))

        np.save(saveDir, res)
        print("Attention information saved in %s" % saveDir)
