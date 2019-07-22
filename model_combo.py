from models import resnet as ResNet
from models import senet as SeNet
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
from torch.optim import Adam, RMSprop
from fiw_dataset import *
from torch.utils.data import RandomSampler, Sampler

from torchvision import models
import joblib
from tensorboardX import SummaryWriter
from datetime import datetime
import math
from submit import *
from tricks.tricks import *
from tricks.advance_pool import *

# from compact_bilinear_pooling import CountSketch, CompactBilinearPooling


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Config():
    train_batch_size = 16
    val_batch_size = 16
    # concate or bilinear
    use_bilinear = False

    # conv trick
    use_spatial_attention = False
    use_se = False
    use_stack = False
    use_model_ensemble = False
    use_drop_out = True
    drop_out_rate = 0.3

    use_random_erasing = True
    replacement_sampling = True

    use_resnet = True

    use_data_extension = False
    use_kinfacew = False

    pooling_method = 'avg'

    optimizer = 'adam'
    adam_amsgrad = True

    lr_scheduler = 'default'
    weight_decay = 0.0

    val_families = 'F09'

    name = 'default'


def get_pretrained_model(include_top=False, pretrain_kind='imagenet', model_name='resnet50'):
    if pretrain_kind == 'vggface2':
        N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained
        resnet50_weight_file = 'weights/resnet50_ft_weight.pkl'
        senet_weight_file = 'weights/senet50_ft_weight.pkl'

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
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.config = config

        if self.config.use_resnet:
            self.pretrained_model = get_pretrained_model(False, pretrain_kind='vggface2', model_name='resnet50')
        else:
            self.pretrained_model = get_pretrained_model(False, pretrain_kind='vggface2', model_name='senet50')

        if self.config.use_bilinear:
            self.bi_conv = nn.Conv2d(2048, 512, 1)
            self.bi_bn = nn.BatchNorm2d(512)
            self.bi_rule = nn.ReLU(0.1)
            self.bilinear = nn.Bilinear(512, 512, 1024)
            if self.config.use_spatial_attention:
                self.conv_sw1 = nn.Conv2d(512, 50, 1)
                self.sw1_bn = nn.BatchNorm2d(50)
                self.sw1_activation = nn.ReLU()

                self.conv_sw2 = nn.Conv2d(50, 1, 1)
                self.sw2_activation = nn.Softplus()
            if self.config.use_se:
                self.selayer = SELayer(512)
            self.ll1 = nn.Linear(1024, 50)
            self.relu = nn.ReLU()
            self.sigmod = nn.Sigmoid()
            self.dropout1 = nn.Dropout(self.config.drop_out_rate)
            self.dropout2 = nn.Dropout(self.config.drop_out_rate)
            self.ll2 = nn.Linear(50, 1)

        else:
            if self.config.use_spatial_attention:
                self.conv_sw1 = nn.Conv2d(2048, 50, 1)
                self.sw1_bn = nn.BatchNorm2d(50)
                self.sw1_activation = nn.ReLU()

                self.conv_sw2 = nn.Conv2d(50, 1, 1)
                self.sw2_activation = nn.Softplus()
            if self.config.use_se:
                self.selayer = SELayer(2048)
            self.ll1 = nn.Linear(4096, 100)
            # self.ll3 = nn.Linear(512, 64)
            self.relu = nn.ReLU()
            self.sigmod = nn.Sigmoid()
            self.dropout = nn.Dropout(self.config.drop_out_rate)
            self.ll2 = nn.Linear(100, 1)

        if self.config.pooling_method == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif self.config.pooling_method == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif self.config.pooling_method == 'rmac':
            self.pool = rmac
        elif self.config.pooling_method == 'gem':
            self.pool = gem

        self.maxpool = nn.AdaptiveMaxPool2d(1)

        if self.config.use_stack:
            self.reduce_conv1 = nn.Conv2d(4096, 4096, 3)
            self.reduce_conv2 = nn.Conv2d(4096, 4096, 3)
            self.reduce_conv3 = nn.Conv2d(4096, 4096, 3)

    def forward_once(self, x):
        # input = self.stn(input)
        x = self.pretrained_model(x)
        if self.config.use_bilinear:
            x = self.bi_conv(x)
            x = self.bi_bn(x)
            x = self.bi_rule(x)
        if self.config.use_se:
            x = self.selayer(x)

        if self.config.use_spatial_attention:
            x = self.forward_spatial_weight(x)

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

    def forward(self, input1, input2):
        if self.config.use_bilinear:
            return self.forward_compact_bilinear(input1, input2)
        else:
            return self.forward_baseline(input1, input2)

    def forward_baseline(self, input1, input2):
        """
        baseline op for compare two input
        :param input1:
        :param input2:
        :return:
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # max1 = self.maxpool(output1)
        # max2 = self.maxpool(output2)

        output1 = self.pool(output1)
        output2 = self.pool(output2)

        # (x1-x2)**2
        sub = torch.sub(output1, output2)
        mul1 = torch.mul(sub, sub)

        # (x1**2-x2**2)
        mul2 = torch.sub(torch.mul(output1, output1), torch.mul(output2, output2))

        # mul1 = torch.cat([max1, output1], 1)
        # mul2 = torch.cat([max2, output2], 1)

        x = torch.cat([mul1, mul2], 1)
        x = x.view(x.size(0), -1)

        x = self.ll1(x)
        x_ = self.relu(x)
        if self.config.use_drop_out:
            x_ = self.dropout(x_)
        x = self.ll2(x_)
        x = self.sigmod(x)
        return x, x_

    def forward_compact_bilinear(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output1 = self.pool(output1)
        output2 = self.pool(output2)

        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)

        x = self.bilinear(output1, output2)
        x = self.relu(x)
        if self.config.use_drop_out:
            x = self.dropout1(x)

        x = self.ll1(x)
        x_ = self.relu(x)
        if self.config.use_drop_out:
            x_ = self.dropout2(x_)

        x = self.ll2(x_)
        x = self.sigmod(x)
        return x, x_

    def forward_use_stack(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = torch.cat([output1, output2], dim=1)
        output = self.reduce_conv1(output)
        output = self.reduce_conv2(output)
        output = self.reduce_conv3(output)

        output = self.pool(output)
        output = output.view(output.size(0), -1)

        x = self.ll1(output)
        x_ = self.relu(x)
        if self.config.use_drop_out:
            x_ = self.dropout(x_)
        x = self.ll2(x_)
        x = self.sigmod(x)
        return x, x_

    def __repr__(self):
        return self.__class__.__name__


def train_model(model, criterion, optimizer, scheduler, dataloaders, writer, num_epochs=200, center_loss=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')
    max_acc = 0.0
    epoch_nums = {'train': 200, 'val': 100}
    for epoch in range(num_epochs):
        # if epoch < 15:
        #     optimizer = optimizers[0]
        # else:
        #     optimizer = optimizers[1]
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
                    output, output_ = model(img1, img2)

                    if center_loss:
                        bce_loss = criterion(output, target)
                        target_ = target.squeeze()
                        centerloss = center_loss(target_, output_)
                        loss = bce_loss + 0.05 * centerloss
                    else:
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

    return max_acc


class CusRandomSampler(Sampler):

    def __init__(self, batch_size, iter_num, relation_sizes, replacement=False):
        super(CusRandomSampler, self).__init__(data_source=None)
        self.batch_size = batch_size
        self.iter_num = iter_num
        self.relation_sizes = relation_sizes
        self.replacement = replacement
        self.use_srs = False

    def __iter__(self):
        if not self.replacement:
            even_list = [x for x in range(2 * self.relation_sizes) if x % 2 == 0]
            random.shuffle(even_list)
            even_list = even_list * 6
            res = []
            for i in range(self.iter_num):
                same_size = self.batch_size // 2
                res.extend(even_list[i * same_size:(i + 1) * same_size])
                res.extend([1] * (self.batch_size - same_size))
            return iter(res)
        else:

            if self.use_srs:
                even_list = [x for x in range(2 * self.relation_sizes) if x % 2 == 0]
                queue = [x for x in range(2 * self.relation_sizes) if x % 2 == 0] * 3
                res = []
                for i in range(self.iter_num):
                    same_size = self.batch_size // 2
                    sm = sample(even_list, same_size)
                    res.extend(sm)
                    for ele in sm:
                        even_list.remove(ele)
                    even_list.extend(queue[i * same_size:(i + 1) * same_size])

                    res.extend([1] * (self.batch_size - same_size))
                return iter(res)


            else:
                even_list = [x for x in range(2 * self.relation_sizes) if x % 2 == 0]
                res = []
                for i in range(self.iter_num):
                    same_size = self.batch_size // 2
                    res.extend(sample(even_list, same_size))
                    res.extend([1] * (self.batch_size - same_size))
                return iter(res)

    def __len__(self):
        return self.batch_size * self.iter_num


def run(config):
    writer = SummaryWriter(logdir=os.path.join("../tb_log", datetime.now().strftime('%b%d_%H-%M-%S') + config.name))
    train, train_map, val, val_map = get_data(config.val_families, config.use_data_extension, config.use_kinfacew)

    datasets = {'train': FaceDataSet(train, train_map, 'train', config.use_random_erasing),
                'val': FaceDataSet(val, val_map, 'val', config.use_random_erasing)}

    train_dataloader = DataLoader(dataset=datasets['train'], num_workers=4,
                                  batch_size=Config.train_batch_size,
                                  sampler=CusRandomSampler(Config.train_batch_size, 200, len(train),
                                                           config.replacement_sampling)
                                  )

    val_dataloader = DataLoader(dataset=datasets['val'], num_workers=4,
                                batch_size=Config.val_batch_size,
                                sampler=CusRandomSampler(Config.train_batch_size, 100, len(val),
                                                         config.replacement_sampling),
                                # shuffle=True
                                )
    data_loaders = {'train': train_dataloader, 'val': val_dataloader}

    model = SiameseNetwork(config=config).to(device)

    # weights = []
    # for i in range(Config.train_batch_size // 2):
    #     weights.append([2.0])
    # for i in range(Config.train_batch_size // 2):
    #     weights.append([1.0])
    # weights = torch.Tensor(weights).to(device)
    # criterion = nn.BCELoss(weights)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

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
    optimizer = None
    if config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=0.00001, amsgrad=config.adam_amsgrad, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=0.0001)
    # optimizer2 = Adam(model.parameters(), lr=0.000001, amsgrad=True, weight_decay=0.1)

    # optimizer = Adam(model.parameters(), lr=0.00001)
    # optimizer = Adam(optim_params, lr=0.00001)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 60, 100], 0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scheduler = None
    if config.lr_scheduler == 'default':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.1,
                                                               verbose=True)
    elif config.lr_scheduler == 'cosineAnneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    elif config.lr_scheduler == 'exp':
        exp_decay = math.exp(-0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    # train_model(model, criterion, optimizer, scheduler, data_loaders, num_epochs=200,center_loss=CenterLoss(2, 50).to(device))
    max_acc = train_model(model, criterion, optimizer, scheduler, data_loaders, writer, num_epochs=100)
    try:
        get_submit(model, config)
    except Exception as e:
        print(e)
    del model
    return max_acc


if __name__ == '__main__':

    configs = []
    for i in range(10):
        val_families = 'F0' + str(i)
        config = Config()
        config.val_families = val_families
        config.name = val_families
        configs.append(config)

    max_accs = []
    for config in configs:
        img = loader('face.jpg', 'train', config.use_random_erasing)
        img = img.unsqueeze(dim=0).to(device)
        model = SiameseNetwork(config=config).to(device)
        print(len(model(img, img)))
        del model
    for config in configs:
        max_accs.append(run(config))

    print(max_accs)
    #
