import __init__paths
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import json
from tensorboardX import SummaryWriter

from utils.loss import dice_loss
from utils.util import segment2n_segment
from model.convshape import ConvShape
from model.densenet2SAT import Features2SAT, PositionalEncoding
from model.densenet3d import densenet3d
from dataloader.dataset_quantified import ClfDataset

parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--train_batch', type=int, default=32, help='batch that get backward')
parser.add_argument('--batch_size', type=int, default=16, help='dataset and test batch size')
parser.add_argument('--n_sat', type=int, default=1024, help='control the number of sat')
parser.add_argument('--file_name', type=str, default='', help='file name')
parser.add_argument('--write_log', type=bool, default=True, help='write log control')
parser.add_argument('--load_model', type=bool, default=True, help='whether load model')
parser.add_argument('--load_model_file', type=str, default='', help='dictionary of load model')
opt = parser.parse_args()
print(opt)

divide_batch = opt.train_batch // opt.batch_size
device_ids = [0, 1]


def train(epoch, model, criterion, optimizer, writer, opt, data_loader):
    model['densesharp'].train()
    model['sat'].train()
    total_cross_loss = 0
    total_dice_loss = 0
    eval_loss = 0
    correct = 0
    positive_target = np.zeros(len(data_loader.dataset))
    positive_score = np.zeros(len(data_loader.dataset))
    optimizer.zero_grad()
    for batch_idx, (data, target, segment_target) in enumerate(tqdm(data_loader)):
        if torch.cuda.is_available():
            data, target, segment_target = data.cuda(), target.cuda(), segment_target.cuda()

        # forward the models
        output, features, segment_output = model['densesharp'](data)
        n_segment = segment2n_segment(segment_target, opt.n_sat)
        batch_features = model['feature2SAT'](features, n_segment)
        batch_features = batch_features + model['p_encoder'](n_segment)
        output_SAT = model['sat'](batch_features)

        # get the loss
        indiv_cross_loss = criterion(output_SAT, target)
        indiv_dice_loss = dice_loss(segment_output, segment_target)
        loss = indiv_cross_loss + 0.2 * indiv_dice_loss
        loss.backward()
        total_cross_loss += indiv_cross_loss.item()
        total_dice_loss += indiv_dice_loss.item()
        eval_loss += loss.item()

        pred = output_SAT.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        possi = F.softmax(output_SAT, dim=1)
        for i, t in enumerate(target):
            pos = opt.batch_size * batch_idx + i
            positive_target[pos] = target.data[i]
            positive_score[pos] = possi.cpu().data[i][0]

        if (batch_idx+1) % divide_batch == 0:
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % 10 == 0:
            log_tmp = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tCrossEntropyLoss: {:.6f}\tDiceLoss: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), indiv_cross_loss.item() / opt.batch_size,
                indiv_dice_loss.item(), loss.item())
            print(log_tmp)
            if opt.write_log:
                with open("./log/{}.txt".format(opt.file_name), "a") as log:
                    log.write('{}\n'.format(log_tmp))

    eval_loss /= len(data_loader.dataset)
    total_cross_loss /= len(data_loader.dataset)
    total_dice_loss = total_dice_loss / (len(data_loader.dataset) / opt.batch_size)
    log_tmp = 'Eval Epoch:{} CrossEntropyLoss:{:.6f} DiceLoss:{:.6f} Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, total_cross_loss, total_dice_loss, eval_loss, correct, len(data_loader.dataset),
        100. * float(correct) / len(data_loader.dataset))
    print(log_tmp)

    # draw the ROC curve
    fpr, tpr, thresholds = roc_curve(positive_target, positive_score, pos_label=0)
    roc_auc = auc(fpr, tpr)
    print('Train_AUC = %.8f' % roc_auc)

    if opt.write_log:
        with open("./data/{}/epoch{}_train_fpr.json".format(opt.file_name, epoch), "w") as f:
            json.dump(fpr.tolist(), f)
        with open("./data/{}/epoch{}_train_tpr.json".format(opt.file_name, epoch), "w") as f:
            json.dump(tpr.tolist(), f)
        with open("./log/{}.txt".format(opt.file_name), "a") as log:
            log.write('{}\n'.format(log_tmp))
            log.write('Train_AUC = %.8f\n' % roc_auc)
        writer.add_scalar('Train_AUC', roc_auc, epoch)
        writer.add_scalar('Train_CrossEntropyLoss', total_cross_loss, epoch)
        writer.add_scalar('Train_DiceLoss', total_dice_loss, epoch)
        writer.add_scalar('Eval_loss', eval_loss, epoch)
        writer.add_scalar('Eval_accuracy', 100. * float(correct) / len(data_loader.dataset), epoch)
        torch.save(model['densesharp'].state_dict(), './model_saved/{}/densesharp_{}_epoch_{}_dict.pkl'
                   .format(opt.file_name, opt.file_name, epoch))
        torch.save(model['sat'].state_dict(), './model_saved/{}/sat_{}_epoch_{}_dict.pkl'
                   .format(opt.file_name, opt.file_name, epoch))


def test(epoch, model, criterion, writer, opt, data_loader, dichotomy=False):
    model['densesharp'].eval()
    model['sat'].eval()
    total_cross_loss = 0
    total_dice_loss = 0
    test_loss = 0
    correct = 0
    positive_target = np.zeros(len(data_loader.dataset))
    positive_score = np.zeros(len(data_loader.dataset))
    for index, (data, target, segment_target) in enumerate(tqdm(data_loader)):
        if torch.cuda.is_available():
            data, target, segment_target = data.cuda(), target.cuda(), segment_target.cuda()

        # forward the models
        output, features, segment_output = model['densesharp'](data)
        n_segment = segment2n_segment(segment_target, opt.n_sat)
        batch_features = model['feature2SAT'](features, n_segment)
        batch_features = batch_features + model['p_encoder'](n_segment)
        output_SAT = model['sat'](batch_features)

        # sum up batch loss
        indiv_cross_loss = criterion(output_SAT, target)
        indiv_dice_loss = dice_loss(segment_output, segment_target)
        loss = indiv_cross_loss + 0.2 * indiv_dice_loss
        total_cross_loss += indiv_cross_loss.item()
        total_dice_loss += indiv_dice_loss.item()
        test_loss += loss.item()

        # get the index of the max log-probability
        pred = output_SAT.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if dichotomy:
            possi = F.softmax(output_SAT, dim=1)
            for i, t in enumerate(target):
                pos = opt.batch_size * index + i
                positive_target[pos] = target.data[i]
                positive_score[pos] = possi.cpu().data[i][0]

    total_cross_loss /= len(data_loader.dataset)
    total_dice_loss /= (len(data_loader.dataset) / opt.batch_size)
    test_loss /= len(data_loader.dataset)
    log_tmp = 'Test epoch:{} CrossEntropyLoss:{:.6f} DiceLoss:{:.6f} Average loss:{:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, total_cross_loss, total_dice_loss, test_loss, correct, len(data_loader.dataset),
        100. * float(correct) / len(data_loader.dataset))
    print(log_tmp)

    # draw the ROC curve
    fpr, tpr, thresholds = roc_curve(positive_target, positive_score, pos_label=0)
    roc_auc = auc(fpr, tpr)
    print('AUC = %.8f' % roc_auc)

    if opt.write_log:
        with open("./log/{}.txt".format(opt.file_name), "a") as log:
            log.write('{}\n'.format(log_tmp))
        writer.add_scalar('Test_AUC', roc_auc, epoch)
        writer.add_scalar('Test_loss', test_loss, epoch)
        writer.add_scalar('Test_CrossEntropyLoss', total_cross_loss, epoch)
        writer.add_scalar('Test_DiceLoss', total_dice_loss, epoch)
        writer.add_scalar('Test_accuracy', 100. * float(correct) / len(data_loader.dataset), epoch)

    if dichotomy and opt.write_log:

        with open("./log/{}.txt".format(opt.file_name), "a") as log:
            log.write('Test_AUC = %.8f\n' % roc_auc)
        with open("./data/{}/epoch{}_test_fpr.json".format(opt.file_name, epoch), "w") as f:
            json.dump(fpr.tolist(), f)
        with open("./data/{}/epoch{}_test_tpr.json".format(opt.file_name, epoch), "w") as f:
            json.dump(tpr.tolist(), f)


def main():
    for eval_set in range(1, 6):
        opt.file_name = 'both_train_no_segment_SAT_3_8_4_crossval_{}_batch_{}'.format(eval_set, opt.train_batch)

        writer = SummaryWriter('./runs/{}_epoch_{}'.format(opt.file_name, opt.nepoch))
        if opt.write_log:
            if not os.path.exists('./data/{}'.format(opt.file_name)):
                os.makedirs('./data/{}'.format(opt.file_name))

            if not os.path.exists('./model_saved/{}'.format(opt.file_name)):
                os.makedirs('./model_saved/{}'.format(opt.file_name))

        train_subset = [1, 2, 3, 4, 5]
        train_subset.remove(eval_set)
        train_dataset = ClfDataset(train=True, crop_size=32, move=5, subset=train_subset, output_segment=True)
        train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=None)

        test_dataset = ClfDataset(train=False, crop_size=32,  subset=[eval_set], output_segment=True)
        test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, sampler=None)

        # define models
        pkl_step_list = [159, 124, 166, 132, 28]
        pkl_name = 'denseSharp_4_4_4_dichotomy_layer_crossval_{}_dataset_60_epoch_{}_dict.pkl'.format(
            eval_set, pkl_step_list[eval_set - 1])
        opt.load_model_file = './model_saved/denseSharp_4_4_4_dichotomy_layer_crossval_{}_dataset_60/{}'.format(
            eval_set, pkl_name)

        # define models
        criterion = nn.CrossEntropyLoss()
        densesharp = densenet3d(with_segment=True)
        feature2SAT = Features2SAT(batch_size=opt.batch_size, n_channel=120, n_sat=opt.n_sat, size=32)
        positional_encoding = PositionalEncoding(n_channel=120, batch_size=opt.batch_size, n_sat=opt.n_sat)
        sat = ConvShape(input_size=120, output_size=2, hidden_size=256, L=3, agg='avg')
        if torch.cuda.is_available():
            densesharp, feature2SAT, sat = densesharp.cuda(), feature2SAT.cuda(), sat.cuda()
            densesharp = nn.DataParallel(densesharp, device_ids=device_ids)
            sat = nn.DataParallel(sat, device_ids=device_ids)
            criterion = criterion.cuda()

        print('----- Start loading DenseSharp! -----')
        if opt.load_model:
            densesharp.load_state_dict(torch.load(opt.load_model_file))
            print('----- Load finished! -----')
        else:
            print('----- Load failed! -----')

        param = list(densesharp.parameters()) + list(sat.parameters())
        optimizer = optim.Adam(param)
        models = {'densesharp': densesharp, 'feature2SAT': feature2SAT, 'p_encoder': positional_encoding, 'sat': sat}
        for epoch in range(opt.nepoch):
            train(epoch, models, criterion, optimizer, writer, opt, train_data_loader)
            test(epoch, models, criterion, writer, opt,  test_data_loader, dichotomy=True)
        writer.close()


if __name__ == '__main__':
    main()
