import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Test the process of data by using DL model
author: 112020333002191
date: 20/8/19
'''
#Simple RNN
class RNN_v1(nn.Module):
    def __init__(self):
        super(RNN_v1, self).__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=64,
                          num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x, _ = self.rnn(x)
        # 10, 8, 4
        x = self.fc(x[:, -1, :])
        # 10, 8 ,1
        return x
# Simple CNN

class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(p=0.5),
            nn.Conv1d(64, 192, kernel_size=1),
            nn.BatchNorm1d(192, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(192, 512, kernel_size=1),
            nn.BatchNorm1d(512, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.BatchNorm1d(1024, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1024, 2048, kernel_size=1),
            nn.BatchNorm1d(2048, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(2048, 4096, kernel_size=1),
            nn.BatchNorm1d(4096, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.BatchNorm1d(256 * 16),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(256 * 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256 * 16, num_classes),
            #w 8, 22, 22
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

#CNN + LSTM
class CRNN(nn.Module):

    def __init__(self, num_classes=2):
        super(CRNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.5),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 192, kernel_size=1),
            nn.BatchNorm1d(192, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(192, 384, kernel_size=1),
            nn.BatchNorm1d(384, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
        )
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, dropout=0.5, num_layers=hidden_num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.BatchNorm1d(8064),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(8064),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8064, num_classes),
            # w [84, 21]
        )

    def forward(self, x, hidden=None):
        x = self.features(x)
        x, hidden = self.rnn(x, hidden)
        #print(hidden[0].shape)
        #hidden[3,200,21]
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, hidden
