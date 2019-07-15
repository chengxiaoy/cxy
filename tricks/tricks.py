import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import *
import random
import math
from PIL import Image
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    "Implement label smoothing.  size表示类别总数  "

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing  # if i=y的公式
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """

        assert x.size(1) == self.size
        true_dist = x.data.clone()  # 先深复制过来
        true_dist.fill_(self.smoothing / (self.size - 1))  # otherwise的公式
        # print true_dist
        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=None):
        if mean is None:
            mean = [0.4914, 0.4822, 0.4465]
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            width, height = img.size
            area = width * height

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = random.randint(0, height - h)
                y1 = random.randint(0, width - w)

                img = np.array(img)

                img[y1:y1 + w, x1:x1 + h, 0] = self.mean[0]
                img[y1:y1 + w, x1:x1 + h, 1] = self.mean[1]
                img[y1:y1 + w, x1:x1 + h, 2] = self.mean[2]

                img = Image.fromarray(img.astype('uint8')).convert('RGB')
                return img

        return img


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class STNLayer(nn.Module):

    def __init__(self):
        """
        input is 3*197*197
        """
        super(STNLayer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(5, 8, kernel_size=6),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=4),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 9 * 9, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 9 * 9)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x


import torch
import torch.nn as nn
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


def main(test_cuda=False):
    print('-' * 80)
    device = torch.device("cuda" if test_cuda else "cpu")
    ct = CenterLoss(10, 2, size_average=True).to(device)
    y = torch.Tensor([0, 0, 2, 1]).to(device)
    feat = torch.zeros(4, 2).to(device).requires_grad_()
    print(list(ct.parameters()))
    print(ct.centers.grad)
    out = ct(y, feat)
    print(out.item())
    out.backward()
    print(ct.centers.grad)
    print(feat.grad)


if __name__ == '__main__':
    torch.manual_seed(999)
    main(test_cuda=False)
    if torch.cuda.is_available():
        main(test_cuda=True)
