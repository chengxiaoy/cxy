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
from fiw_dataset import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def get_pretrained_model(include_top=False):
    N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained

    model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top).eval()

    weight_file = 'resnet50_ft_weight.pkl'

    utils.load_state_dict(model, weight_file)
    return model


class SiameseNetwork(nn.Module):
    def __init__(self, include_top=False):
        super(SiameseNetwork, self).__init__()
        self.pretrained_model = get_pretrained_model(include_top)

        self.ll1 = nn.Linear(8196, 100)
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
        mul2 = torch.mul(output1, output2)

        x = torch.cat([mul1, mul2], 1)

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
            # Iterate over version5_gray_data_2W_top3-0.7.
            for i, (img1, img2, target) in enumerate(dataloaders[phase]):
                optimizer.zero_grad()
                output = model(img1, img2)
                loss = criterion(output, target)
                loss.backward()
                running_loss = running_loss + loss.data.cpu().numpy()
                optimizer.step()

            epoch_loss = running_loss / len(dataloaders[phase])

            print('{} Loss: {:.4f} '.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('min loss : {:4f}'.format(min_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model
    torch.save(model.state_dict(), str(model) + ".pth")

    return model


img = loader('face.jpg', 'extract')
model = SiameseNetwork(False)
features = model.forward_once(img.to(device)).data.cpu().numpy()

train, train_map, val, val_map = get_data()

datasets = {'train': FaceDataSet(train, train_map, 'train'), 'val': FaceDataSet(val, val_map, 'val')}


train_dataloader = DataLoader(dataset=datasets['train'], shuffle=True, num_workers=4,
                                      batch_size=Config.train_batch_size, collate_fn=collate_triples)

test_dataloader = DataLoader(dataset=test_data, shuffle=True, num_workers=4, batch_size=Config.test_batch_size,
                                     collate_fn=collate_triples)



print(features.shape)
