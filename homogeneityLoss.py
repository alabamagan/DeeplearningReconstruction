import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class HomogeneityLoss(nn.Module):