import os
from model import CRNN
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from Param_counter import get_model_complexity_info

'''
Test the process of data by using Attention-CNN method
author: 112020333002191
date: 20/8/19
reference: https://github.com/AndrewCuHo/
'''
#
RESUME = True
eval = False

# Hyper parameters
num_epochs = 2000
num_classes = 2
batch_size = 50
learning_rate = 3e-4
weight_decay = 1e-4
torch.cuda.set_device(0)
torch.manual_seed(999)
#################
#  DataLoad   ###
#################
x, y, _ = pickle.load(open('20min_data.npy', 'rb'))
x = x[:, :8]  # Take EEG
df = pd.DataFrame(x)
df['y'] = y
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=999)
X_train_valid = torch.from_numpy(X_train).float().unsqueeze(1)  # dimension expand
# X_train_valid = X_train_valid.view(-1, 4, 4)
# [9156, 1 , 8]
y_train_valid = torch.from_numpy(y_train).long()
train_data = torch.utils.data.TensorDataset(X_train_valid, y_train_valid)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

X_test = torch.from_numpy(X_test).float().unsqueeze(1)
y_test = torch.from_numpy(y_test).long()
test_data = torch.utils.data.TensorDataset(X_test, y_test)
testloader = DataLoader(test_data, batch_size=X_test.shape[0], shuffle=False)

def train_model():

    model = CRNN().cuda()
    writer = SummaryWriter('./tblogs/%f/' % learning_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.1)
    t0 = time.time()
    cnt = 0
    total_step = len(trainloader)
    hidden = None
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    classes = ('True', 'False')
    start_epoch = 0
    ########################
    ####断点重训########
    if RESUME:
        path_checkpoint = 'F:/EEG_data/MNE_test/code/reweights/checkpoint/%f/ckpt_best.pth' % (learning_rate)  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])

    for epoch in range(start_epoch + 1, num_epochs + 1):

        # keep track of training
        train_loss = 0.0
        train_counter = 0
        train_losses = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in trainloader:
            data, target = data.cuda(), target.cuda()
            target = target.long()
            optimizer.zero_grad()
            output, hidden = model(data, hidden)
            a, b = hidden
            hidden = (a.data, b.data)
            loss = criterion(output, target)
            # print(target.data)
            loss.backward()
            optimizer.step()
            train_loss += (loss.item() * data.size(0))
            train_counter += data.size(0)
            train_losses = (train_loss / train_counter)
            writer.add_scalar('Train/Loss', train_losses, epoch)
            cnt += 1
            if cnt % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, cnt + 1, total_step, loss.item()))
        cnt = 0
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': lr_schedule.state_dict()
        }
        if not os.path.isdir("F:/EEG_data/MNE_test/code/reweights/checkpoint/%f" % (learning_rate)):
            os.makedirs("F:/EEG_data/MNE_test/code/reweights/checkpoint/%f" % (learning_rate))
        torch.save(checkpoint, 'F:/EEG_data/MNE_test/code/reweights/checkpoint/%f/ckpt_best.pth' % (learning_rate))
        if epoch % 200 == 0:
            torch.save(model.state_dict(), './model_EEG.pt')
        #############
        # eval
        #############
        #TODO add eval part
    # torch.save(model.state_dict(), './model_EEG.pt')
    time_total = time.time() - t0
    print('Total time: {:4.3f}, average time per epoch: {:4.3f}'.format(time_total, time_total / num_epochs))


def eval_model():
    model = CRNN()
    model.load_state_dict(torch.load('./model_EEG.pt'))
    # specify the target classes
    classes = ('True', 'False')

    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    model.eval()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data, target
            target = target.long()
            output, _ = model(data)
            #print(output.data)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # print(pred)
            # compare predictions to true label
            correct = (pred == target).squeeze()
            for i, label in enumerate(target):
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        for i in range(len(classes)):
            print('Accuracy of %s : %2d%% out of %d cases' %
                  (classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))

        data = next(iter(testloader))
        inputs, targets = data
        inputs = inputs
        targets = targets
        targets = targets.long()
        outputs, _ = model(inputs)
        probability, predicted = torch.max(outputs.data, 1)
        c = (predicted == targets).squeeze()

        eval_metrics = pd.DataFrame(np.empty([2, 4]))
        eval_metrics.index = ["baseline"] + ['RNN']
        eval_metrics.columns = ["Accuracy", "ROC AUC", "PR AUC", "Log Loss"]
        pred = np.repeat(0, len(y_test.cpu()))
        pred_proba = np.repeat(0.5, len(y_test.cpu()))
        eval_metrics.iloc[0, 0] = accuracy_score(y_test.cpu(), pred)
        eval_metrics.iloc[0, 1] = roc_auc_score(y_test.cpu(), pred_proba)
        eval_metrics.iloc[0, 2] = average_precision_score(y_test.cpu(), pred_proba)
        eval_metrics.iloc[0, 3] = log_loss(y_test.cpu(), pred_proba)
        eval_metrics.iloc[1, 0] = accuracy_score(y_test.cpu(), predicted.cpu())
        eval_metrics.iloc[1, 1] = roc_auc_score(y_test.cpu(), probability.cpu())
        eval_metrics.iloc[1, 2] = average_precision_score(y_test.cpu(), probability.cpu())
        eval_metrics.iloc[1, 3] = 0  # log_loss(y_test.cpu(), pred_proba[:, 1])

        print(eval_metrics)


if __name__ == "__main__":
    if not eval :
        train_model()
    eval_model()
    # flops, params = get_model_complexity_info(model, (224, 224), as_strings=False, print_per_layer_stat=False)
    # print('Flops:  %.3f' % (flops / 1e9))
    # print('Params: %.2fM' % (params / 1e6))
