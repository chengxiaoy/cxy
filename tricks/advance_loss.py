import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
           x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


class CusAngleLinearLoss(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(CusAngleLinearLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.m = m
        self.phiflag = phiflag
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        self.iter = 0

        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.gamma = 0

    def forward(self, x, labels):
        self.iter += 1
        eps = 1e-12
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x_norm = F.normalize(x, dim=1)
        x_len = x.norm(2, 1, True).clamp_min(eps)
        cos_theta = self.fc(x_norm)
        cos_m_theta = self.mlambda[self.m](cos_theta)

        theta = Variable(cos_theta.data.acos())
        k = (self.m * theta / math.pi).floor()
        n_one = k * 0.0 - 1
        phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        cos_theta = cos_theta * x_len
        phi_theta = phi_theta * x_len

        target = labels.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.iter))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        #
        # output[index] -= cos_theta[index] * (1.0 + 0)
        # output[index] += phi_theta[index] * (1.0 + 0)

        loss = F.cross_entropy(output, target.squeeze())

        # softmax loss

        # logit = F.log_softmax(output)
        #
        # logit = logit.gather(1, target).view(-1)
        # pt = logit.data.exp()
        #
        # loss = -1 * (1 - pt) ** self.gamma * logit
        # loss = loss.mean()
        return cos_theta, loss





class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta)
        return output  # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        return F.cross_entropy(output, target.squeeze())

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss


class SphereMarginProduct(nn.Module):
    def __init__(self, in_feature, out_feature, m=4, base=1000.0, gamma=0.0001, power=2, lambda_min=5.0, iter=0):
        assert m in [1, 2, 3, 4], 'margin should be 1, 2, 3 or 4'
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.m = m
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min
        self.iter = 0
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.margin_formula = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        self.iter += 1
        self.cur_lambda = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta(-1, 1)

        cos_m_theta = self.margin_formula(self.m)(cos_theta)
        theta = cos_theta.data.acos()
        k = ((self.m * theta) / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        phi_theta_ = (self.cur_lambda * cos_theta + phi_theta) / (1 + self.cur_lambda)
        norm_of_feature = torch.norm(input, 2, 1)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = one_hot * phi_theta_ + (1 - one_hot) * cos_theta
        output *= norm_of_feature.view(-1, 1)

        return output


if __name__ == "__main__":
    input = torch.randn(3, 50, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    angle_loss = AngleLoss()
    angle_linear = CusAngleLinearLoss(50, 5)
    theta = angle_linear(input, target)
    # loss = F.cross_entropy(input, target)
    print(theta)
