import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from thop import profile
#from torchstat import stat
from torch.autograd import Variable
from model import NetworkImageNet as Network


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data/tiny-imagenet', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--model_path', type=str, default='-20-30eval-EXP-20201026-092302/checkpoint.pth.tar', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='NASP', help='which architecture to use')
args = parser.parse_args()


CLASSES =8





genotype = eval("genotypes.%s" % args.arch)
model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
model = model.cuda()
model.load_state_dict(torch.load(args.model_path)['state_dict'])
input = torch.cuda.FloatTensor(1, 3, 128, 128)
flops, params = profile(model, inputs=(input,))
print(flops)



