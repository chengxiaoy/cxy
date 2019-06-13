import resnet as ResNet
import utils
from torchvision.transforms import transforms
import torchvision
import numpy as np
from PIL import Image
import PIL
import os
import torch
from glob import glob
from collections import defaultdict
import pandas as pd
from fiw_dataset import *
from torch import nn
import torch
import torch.nn.functional as F
import time
import copy
from torch.optim import Adam
import math
from torch.optim.lr_scheduler import StepLR
from fiw_dataset import *
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Config():
    train_batch_size = 64
    val_batch_size = 64


def get_pretrained_model(include_top=False, pretrain_kind='vggface2'):
    if pretrain_kind == 'vggface2':
        N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained

        model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top).eval()

        weight_file = 'resnet50_ft_weight.pkl'

        utils.load_state_dict(model, weight_file)
        return model
    elif pretrain_kind == 'imagenet':
        return nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
    return None


class SiameseNetwork(nn.Module):
    def __init__(self, include_top=False):
        super(SiameseNetwork, self).__init__()
        self.pretrained_model = get_pretrained_model(include_top, pretrain_kind='imagenet')

        self.ll1 = nn.Linear(4096, 100)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        self.dropout = nn.Dropout(0.01)
        self.ll2 = nn.Linear(100, 1)

    def forward_once(self, x):
        x = self.pretrained_model(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        globalmax = nn.AdaptiveMaxPool2d(1)
        globalavg = nn.AdaptiveAvgPool2d(1)
        output1 = torch.cat([globalmax(output1), globalavg(output1)], 1)
        output2 = torch.cat([globalmax(output2), globalavg(output2)], 1)

        sub = torch.sub(output1, output2)
        mul1 = torch.mul(sub, sub)
        # mul2 = torch.mul(output1, output2)
        #
        # x = torch.cat([mul1, mul2], 1)

        x = mul1.view(mul1.size(0), -1)

        x = self.ll1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ll2(x)
        x = self.sigmod(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # model.eval()
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
                # model.apply(set_batchnorm_eval)
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            for i, (img1, img2, target) in enumerate(dataloaders[phase]):
                img1 = img1.to(device)
                img2 = img2.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(img1, img2)
                    loss = criterion(output, target)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss = running_loss + loss.item()
                    output = output.data.cpu().numpy()
                    label = output > 0.5
                    for i, j in zip(label, target.data.cpu().numpy()):
                        if i[0] == j[0]:
                            running_corrects += 1

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('min loss : {:4f}'.format(min_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model
    torch.save(model.state_dict(), str(model) + ".pth")

    return model


if __name__ == '__main__':
    img1 = loader('face.jpg', 'extract').unsqueeze(0)
    img2 = loader('face.jpg', 'extract').unsqueeze(0)
    model = SiameseNetwork(False).to(device)
    res = model(img1.to(device), img2.to(device)).data.cpu().numpy()
    print(res)

    train, train_map, val, val_map = get_data()

    datasets = {'train': FaceDataSet(train, train_map, 'train'), 'val': FaceDataSet(val, val_map, 'val')}

    train_dataloader = DataLoader(dataset=datasets['train'], shuffle=True, num_workers=4,
                                  batch_size=Config.train_batch_size)

    print(len(train_dataloader))

    val_dataloader = DataLoader(dataset=datasets['val'], shuffle=True, num_workers=4,
                                batch_size=Config.val_batch_size)
    data_loaders = {'train': train_dataloader, 'val': val_dataloader}

    criterion = F.binary_cross_entropy

    optimizer = Adam(model.parameters(), lr=0.00001)

    exp_decay = math.exp(-0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    train_model(model, criterion, optimizer, scheduler, data_loaders)
