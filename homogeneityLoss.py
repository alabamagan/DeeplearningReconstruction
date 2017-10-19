import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pyinn.im2col import Im2Col, Col2Im

class HomogeneityLoss(nn.Module):
    def __init__(self, kernsize):
        super(HomogeneityLoss, self).__init__()

        self.im2col = Im2Col(kernsize, kernsize)

    def forward(self, input, target):
        assert input.data.size() == target.data.size(), "Input and target has different dimensions"

        colinput = self.im2col(input)
        coltarget = self.im2col(target)
        s = colinput.data.size()

        colinput = colinput.view(s[0], s[1], s[2], s[3]*s[4]).contiguous()
        coltarget = coltarget.view(s[0], s[1], s[2], s[3]*s[4]).contiguous()

        colinput = F.avg_pool2d(colinput, kernel_size=[s[1], s[2]])
        coltarget = F.avg_pool2d(coltarget, kernel_size=[s[1], s[2]])

        var_input = colinput.data.var()
        var_target = coltarget.data.var()

        loss = (var_input - var_target) / var_target
        loss = loss.sum()
        return loss