import resnet as ResNet
import senet as SeNet
import utils
from torchvision.transforms import transforms
import torchvision
import numpy as np
from PIL import Image
import PIL
import os
import torch
from fiw_dataset import *
from torch import nn
import torch
import torch.nn.functional as F
import time
import copy
from torch.optim import Adam
from fiw_dataset import *
from torch.utils.data import RandomSampler, Sampler

from torchvision import models
import joblib
from tensorboardX import SummaryWriter
from datetime import datetime
import math
from submit import *
from tricks.tricks import *

# from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

writer = SummaryWriter(logdir=os.path.join("../tb_log", datetime.now().strftime('%b%d_%H-%M-%S')))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Config():
    train_batch_size = 16
    val_batch_size = 16


def get_pretrained_model(include_top=False, pretrain_kind='imagenet', model_name='resnet50'):
    if pretrain_kind == 'vggface2':
        N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained
        resnet50_weight_file = 'resnet50_ft_weight.pkl'
        senet_weight_file = 'senet50_ft_weight.pkl'

        if model_name == 'resnet50':
            model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=include_top).eval()
            utils.load_state_dict(model, resnet50_weight_file)
            return model

        elif model_name == 'senet50':
            model = SeNet.senet50(num_classes=N_IDENTITY, include_top=include_top).eval()
            utils.load_state_dict(model, senet_weight_file)
            return model
    elif pretrain_kind == 'imagenet':
        return nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
    return None


class SiameseNetwork(nn.Module):
    def __init__(self, include_top=False):
        super(SiameseNetwork, self).__init__()
        self.pretrained_model = get_pretrained_model(include_top, pretrain_kind='vggface2')

        # self.pretrained_model2 = get_pretrained_model(include_top, pretrain_kind='vggface2', model_name='senet50')
        self.ll1 = nn.Linear(4096, 100)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        self.ll2 = nn.Linear(50, 1)

        self.bilinear = nn.Bilinear(512, 512, 1024)
        self.lll = nn.Linear(1024, 50)
        self.ll = nn.Linear(1024, 100)

        self.ll3 = nn.Linear(100, 1)

        self.conv = nn.Conv2d(2048, 512, 1)

        # self.stn = STNLayer()

        # self.selayer = SELayer(512)
        self.globalavg = nn.AdaptiveAvgPool2d(1)
        self.globalmax = nn.AdaptiveMaxPool2d(1)

        self.dropout2 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv_sw1 = nn.Conv2d(512, 50, 1)
        self.sw1_bn = nn.BatchNorm2d(50)
        self.sw1_activation = nn.ReLU()

        self.conv_sw2 = nn.Conv2d(50, 1, 1)
        self.sw2_activation = nn.Softplus()

    def forward_once(self, input):
        # input = self.stn(input)
        x = self.pretrained_model(input)
        output1 = self.conv(x)
        output1 = self.bn1(output1)
        output1 = self.relu(output1)

        output1 = self.globalavg(output1)
        output1 = output1.view(output1.size(0), -1)

        return output1

    def last_part(self, x):
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.lll(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ll2(x)
        return x

    def forward_spatial_weight(self, input):
        """
        input is the feature map need spatial weight
        :param input:
        :return:
        """
        input_sw1 = self.conv_sw1(input)
        input_sw1 = self.sw1_bn(input_sw1)
        input_sw1 = self.sw1_activation(input_sw1)

        input_sw2 = self.conv_sw2(input_sw1)
        input_sw2 = self.sw2_activation(input_sw2)

        return torch.mul(input, input_sw2)

    def forward(self, input1, input2, input3):
        return self.forward_compact_bilinear(input1, input2, input3)

    def forward_compact_bilinear(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        x1 = self.bilinear(output1, output2)
        x2 = self.bilinear(output2, output3)
        x3 = self.bilinear(output1, output3)
        return self.last_part(x1), self.last_part(x2), self.last_part(x3)

    def __repr__(self):
        return self.__class__.__name__


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')
    max_acc = 0.0
    epoch_nums = {'train': 200, 'val': 70}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_loss = {}
        epoch_acc = {}
        epoch_true_negative = {}
        epoch_false_positive = {}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # model.train()
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
                # model.apply(set_batchnorm_eval)
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            true_negative = 0
            false_positive = 0

            for i, (img1, img2, img3, target) in enumerate(dataloaders[phase]):
                if i == epoch_nums[phase]:
                    break
                img1 = img1.to(device)
                img2 = img2.to(device)
                img3 = img3.to(device)

                target = []
                for num in range(Config.train_batch_size):
                    target.append([1.0])
                for num in range(2 * Config.train_batch_size):
                    target.append([0.0])
                target = torch.Tensor(target).float()
                target = target.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    vision_info = [False, epoch]
                    if phase == 'val' and i == 10:
                        vision_info[0] = True
                        vision_info[1] = i * epoch
                    output1, output2, output3 = model(img1, img2, img3)

                    output = torch.cat([output1, output2, output3], 0)

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
                        elif i[0]:
                            true_negative += 1
                        else:
                            false_positive += 1

            epoch_true_negative[phase] = true_negative / (epoch_nums[phase] * Config.train_batch_size)
            epoch_false_positive[phase] = false_positive / (epoch_nums[phase] * Config.train_batch_size)
            epoch_loss[phase] = running_loss / epoch_nums[phase]
            epoch_acc[phase] = running_corrects / (epoch_nums[phase] * Config.train_batch_size)

            writer.add_text('Text', '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase], epoch_acc[phase]),
                            epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase], epoch_acc[phase]))
            print('{} true_negative:{:.4f} false_positive:{:.4f}'.format(phase, epoch_true_negative[phase],
                                                                         epoch_false_positive[phase]))

            # deep copy the model
            if phase == 'val' and epoch_loss[phase] < min_loss:
                min_loss = epoch_loss[phase]
                torch.save(model.state_dict(), str(model) + "loss.pth")

            if phase == 'val' and epoch_acc[phase] > max_acc:
                max_acc = epoch_acc[phase]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), str(model) + ".pth")
        writer.add_scalars('data/true_negative',
                           {'train': epoch_true_negative['train'], 'val': epoch_true_negative['val']}, epoch)
        writer.add_scalars('data/false_positive',
                           {'train': epoch_false_positive['train'], 'val': epoch_false_positive['val']}, epoch)

        writer.add_scalars('data/loss', {'train': epoch_loss['train'], 'val': epoch_loss['val']}, epoch)
        writer.add_scalars('data/acc', {'train': epoch_acc['train'], 'val': epoch_acc['val']}, epoch)
        # writer.add_scalar('data/loss', scheduler.get_lr())
        scheduler.step(epoch_acc['val'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('min loss : {:4f}'.format(min_loss))
    print('max acc : {:4f}'.format(max_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    # save model
    torch.save(model.state_dict(), str(model) + ".pth")

    return model


class CusRandomSampler(Sampler):

    def __init__(self, batch_size, iter_num, relation_sizes):
        super(CusRandomSampler, self).__init__(data_source=None)
        self.batch_size = batch_size
        self.iter_num = iter_num
        self.relation_sizes = relation_sizes

    def __iter__(self):
        even_list = [x for x in range(2 * self.relation_sizes) if x % 2 == 0]
        res = []
        for i in range(self.iter_num):
            res.extend(sample(even_list, self.batch_size))

        return iter(res)

    def __len__(self):
        return self.batch_size * self.iter_num


if __name__ == '__main__':
    train, train_map, val, val_map = get_data()

    datasets = {'train': FaceDataSet_V2(train, train_map, 'train', False),
                'val': FaceDataSet_V2(val, val_map, 'val', False)}

    train_dataloader = DataLoader(dataset=datasets['train'], num_workers=4,
                                  batch_size=Config.train_batch_size,
                                  sampler=CusRandomSampler(Config.train_batch_size, 200, len(train))
                                  )

    val_dataloader = DataLoader(dataset=datasets['val'], num_workers=4,
                                batch_size=Config.val_batch_size,
                                sampler=CusRandomSampler(Config.train_batch_size, 100, len(val)),
                                # shuffle=True
                                )
    data_loaders = {'train': train_dataloader, 'val': val_dataloader}

    model = SiameseNetwork(False).to(device)

    # weights = []
    # for i in range(Config.train_batch_size // 2):
    #     weights.append([2.0])
    # for i in range(Config.train_batch_size // 2):
    #     weights.append([1.0])
    # weights = torch.Tensor(weights).to(device)
    # criterion = nn.BCELoss(weights)
    criterion = nn.BCEWithLogitsLoss()

    optim_params = []

    frozen_layers = ['pretrained_model.conv1.weight', 'pretrained_model.bn1.weight', 'pretrained_model.bn1.bias']
    prefix_layers = ['pretrained_model.layer1', 'pretrained_model.layer2']

    # for name, params in model.named_parameters():
    #     frozen = False
    #     for frozen_layer in frozen_layers:
    #         if name.startswith(frozen_layer):
    #             frozen = True
    #     for prefix_layer in prefix_layers:
    #         if name.startswith(prefix_layer):
    #             frozen = True
    #     if not frozen:
    #         optim_params.append(params)

    optimizer = Adam(model.parameters(), lr=0.00001, amsgrad=True)
    # optimizer = Adam(model.parameters(), lr=0.00001)
    # optimizer = Adam(optim_params, lr=0.00001)

    # exp_decay = math.exp(-0.01)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 100], 0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.1, verbose=True)

    train_model(model, criterion, optimizer, scheduler, data_loaders)
    get_submit()