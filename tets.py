import resnet as ResNet
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

# from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

writer = SummaryWriter(logdir=os.path.join("../tb_log", datetime.now().strftime('%b%d_%H-%M-%S')))

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class Config():
    train_batch_size = 16
    val_batch_size = 16


def get_pretrained_model(include_top=False, pretrain_kind='imagenet'):
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
        self.pretrained_model = get_pretrained_model(include_top, pretrain_kind='vggface2')
        self.ll1 = nn.Linear(4096, 100)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        self.dropout = nn.Dropout(0.01)
        self.ll2 = nn.Linear(100, 1)

        self.bilinear = nn.Bilinear(512, 512, 512)
        self.lll = nn.Linear(512, 100)

        self.conv = nn.Conv2d(2048, 512, 1)
        self.globalavg = nn.AdaptiveAvgPool2d(1)
        self.dropout2 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward_once(self, x):
        x = self.pretrained_model(x)
        return x

    def forward(self, input1, input2, visual_info):
        # return self.forward_baseline(input1, input2, visual_info)
        return self.forward_compact_bilinear(input1, input2)

    def forward_baseline(self, input1, input2, visual_info):
        """
        baseline op for compare two input
        :param input1:
        :param input2:
        :return:
        """
        output1 = self.forward_once(input1)
        # if visual_info[0]:
        #     x = vutils.make_grid(output1[:, :3, :, :], normalize=True, scale_each=True)
        #     writer.add_image('Image', x, visual_info[1])

        output2 = self.forward_once(input2)
        globalmax = nn.AdaptiveMaxPool2d(1)
        globalavg = nn.AdaptiveAvgPool2d(1)

        output1 = globalavg(output1)
        output2 = globalavg(output2)

        # output1 = torch.cat([globalavg(output1), globalavg(output1)], 1)
        # output2 = torch.cat([globalavg(output2), globalavg(output2)], 1)

        # (x1-x2)**2
        sub = torch.sub(output1, output2)
        mul1 = torch.mul(sub, sub)

        # x = mul1.view(mul1.size(0),-1)

        # (x1**2-x2**2)
        mul2 = torch.sub(torch.mul(output1, output1), torch.mul(output2, output2))
        # x1*x2
        # mul2 = torch.mul(output1, output2)

        x = torch.cat([mul1, mul2], 1)
        x = x.view(x.size(0), -1)

        x = self.ll1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ll2(x)
        x = self.sigmod(x)
        return x

    def forward_compact_bilinear(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output1 = self.conv(output1)
        output2 = self.conv(output2)
        output1 = self.bn1(output1)
        output2 = self.bn1(output2)
        output1 = self.relu(output1)
        output2 = self.relu(output2)

        output1 = self.globalavg(output1)
        output2 = self.globalavg(output2)

        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)

        output = self.bilinear(output1, output2)

        x = self.bn2(output)
        x = self.lll(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ll2(x)
        x = self.sigmod(x)
        return x

    def forward_bilinear(self, input1, input2):
        """
        the output size of bilinear is related to channel_size**2
        the linear params after bilinear op maybe very huge
        :param input1:
        :param input2:
        :return:
        """
        output1 = self.forward_once(input1)
        h, w = output1.shape[2], output1.shape[3]
        c = output1.shape[1]
        output1 = output1.view(-1, c, h * w)

        output2 = self.forward_once(input2)
        output2 = output2.view(-1, c, h * w)

        output = torch.matmul(output1, output2.permute(0, 2, 1)).view(-1, c * c) / (h * w)
        output_sqrt = torch.sign(output) * (torch.sqrt(torch.abs(output)) + 1e-10)
        output = F.normalize(output_sqrt, dim=1)
        x = output
        x = self.lll(x)
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
    max_acc = 0.0
    epoch_nums = {'train': 200, 'val': 70}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_loss = {}
        epoch_acc = {}
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            model.train()
            # if phase == 'train':
            #     # scheduler.step()
            #     model.train()  # Set model to training mode
            #     # model.apply(set_batchnorm_eval)
            # else:
            #     model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for i, (img1, img2, target) in enumerate(dataloaders[phase]):
                if i == epoch_nums[phase]:
                    break
                img1 = img1.to(device)
                img2 = img2.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    vision_info = [False, epoch]
                    if phase == 'val' and i == 10:
                        vision_info[0] = True
                        vision_info[1] = i * epoch
                    output = model(img1, img2, vision_info)
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

            # epoch_loss[phase] = running_loss / len(dataloaders[phase])
            # epoch_acc[phase] = running_corrects / len(dataloaders[phase].dataset)
            epoch_loss[phase] = running_loss / epoch_nums[phase]
            epoch_acc[phase] = running_corrects / (epoch_nums[phase] * Config.train_batch_size)

            writer.add_text('Text', '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase], epoch_acc[phase]),
                            epoch)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase], epoch_acc[phase]))

            # deep copy the model
            if phase == 'val' and epoch_loss[phase] < min_loss:
                min_loss = epoch_loss[phase]

            if phase == 'val' and epoch_acc[phase] > max_acc:
                max_acc = epoch_acc[phase]
                best_model_wts = copy.deepcopy(model.state_dict())
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
        self.batch_size = batch_size
        self.iter_num = iter_num
        self.relation_sizes = relation_sizes

    def __iter__(self):
        even_list = [x for x in range(2 * self.relation_sizes) if x % 2 == 0]
        res = []
        for i in range(self.iter_num):
            res.extend(sample(even_list, self.batch_size // 2))
            res.extend([1] * (self.batch_size // 2))

        return iter(res)

    def __len__(self):
        return self.batch_size * self.iter_num


if __name__ == '__main__':
    # img1 = loader('face.jpg', 'extract').unsqueeze(0)
    # img2 = loader('face.jpg', 'extract').unsqueeze(0)
    model = SiameseNetwork(False).to(device)
    #
    # # print(model.forward_bilinear(img1,img2).data.cpu().numpy())
    #
    # res = model(img1.to(device), img2.to(device), [False, 0]).data.cpu().numpy()
    # print(res)

    train, train_map, val, val_map = get_data()

    datasets = {'train': FaceDataSet(train, train_map, 'train'), 'val': FaceDataSet(val, val_map, 'val')}

    train_dataloader = DataLoader(dataset=datasets['train'], num_workers=4,
                                  batch_size=Config.train_batch_size,
                                  sampler=CusRandomSampler(Config.train_batch_size, 200, len(train))
                                  )

    print(len(train_dataloader))

    val_dataloader = DataLoader(dataset=datasets['val'], num_workers=4,
                                batch_size=Config.val_batch_size,
                                sampler=CusRandomSampler(Config.train_batch_size, 100, len(val)),
                                # shuffle=True
                                )
    data_loaders = {'train': train_dataloader, 'val': val_dataloader}

    criterion = nn.BCELoss()

    # optim_params = []
    # for name, params in model.named_parameters():
    #     if name.startswith('pretrained_model.7') or name.startswith('ll'):
    #         optim_params.append(params)

    optimizer = Adam(model.parameters(), lr=0.00001)

    # exp_decay = math.exp(-0.01)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 60], 0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.1, verbose=True)

    train_model(model, criterion, optimizer, scheduler, data_loaders)
