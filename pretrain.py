# £¡Users\GG\anaconda3\envs\zxx1\python
# -*- coding:utf-8 -*-
# author£ºZXX time:5/15/23
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import NetworkCIFAR2 as Network
import torch.utils.data as dataloader
from torchvision import transforms
from torch.utils.data import Dataset
import torch, gzip, os
import pandas as pd
from numpy import array

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data/zhangxx/package/TFDDatabase-20-30(xx)', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=400, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=15, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='NASP40lei15', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
# parser.add_argument('--weight-cent', type=float, default=0.01, help="weight for center loss")
# parser.add_argument('--weight-cent2', type=float, default=0.00, help="weight for triplet loss")

args = parser.parse_args()

args.save = 'NASP 40lei15(4) eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))

CIFAR_CLASSES = 40

class DealDataset(Dataset):

  def __init__(self, folder, data_name, label_name, transform=None):
    (train_set, train_labels) = load_data(folder, data_name, label_name)
    self.train_set = train_set
    self.train_labels = train_labels
    self.transform = transform

  def __getitem__(self, index):
    img, target = self.train_set[index], int(self.train_labels[index])
    if self.transform is not None:
      img = self.transform(img)
    return img, target

  def __len__(self):
    return len(self.train_set)

def load_data(data_folder, data_name, label_name):
  with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
    x_train = np.frombuffer(
      imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
  return (x_train, y_train)


trainDataset = DealDataset(
  '/data/zhangxx/SEI/MTC/FSL-MTC/TIFS_FS-MTC/Dataset40lei/5_Mnist',
  "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
testDataset = DealDataset(
  '/data/zhangxx/SEI/MTC/FSL-MTC/TIFS_FS-MTC/Dataset40lei/5_Mnist',
  "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
# xtrain, ytrain = load_data('/data/zhangxx/SEI/MTC/FSL-MTC/Dataset40lei/5_Mnist',"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
# print('ytrain', ytrain.shape)
# xtest, ytest = load_data('/data/zhangxx/SEI/MTC/FSL-MTC/Dataset40lei/5_Mnist',
#   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
# print('ytest', ytest.shape)


train_queue = dataloader.DataLoader(
  dataset=trainDataset,
  batch_size=args.batch_size,
  shuffle=False,
)
test_queue = dataloader.DataLoader(
  dataset=testDataset,
  batch_size=args.batch_size,
  shuffle=False,
)
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  # model = CNN()
  # model = VGG()
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
      momentum=args.momentum, weight_decay=args.weight_decay)


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc_top1 = 0
  train_acc1 = np.zeros([args.epochs])
  valid_acc1 = np.zeros([args.epochs])
  train_obj1 = np.zeros([args.epochs])
  valid_obj1 = np.zeros([args.epochs])
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    # logging.info('train_acc %f', train_acc, 'train_loss %f', train_obj)

    valid_acc, valid_obj = infer(test_queue, model, criterion)
    # logging.info('valid_acc %f', valid_acc, 'valid_loss %f', valid_obj)
    train_acc1[epoch] = train_acc
    valid_acc1[epoch] = valid_acc
    train_obj1[epoch] = train_obj
    valid_obj1[epoch] = valid_obj
    result_acc_loss = train_acc1, valid_acc1, train_obj1, valid_obj1
    result_acc_loss = array(result_acc_loss)
    result_acc_loss = result_acc_loss.reshape(4, -1).T
    df = pd.DataFrame(result_acc_loss, columns=['train_acc', 'valid_acc', 'train_loss', 'valid_loss'])
    # ±£´æµ½±¾µØexcel
    df.to_excel(f"result/FS-NASP40lei15(4)acc_loss.xlsx", index=False)

    utils.save(model, 'model/NASP40lei15(4).pt')


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top2 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    # input, _, target, _ = train_test_split(input, target, test_size=0.3, random_state=30)
    input = Variable(input).cuda()
    target = Variable(target).cuda()
    optimizer.zero_grad()
    logits, features = model(input)
    # print('feature', features.shape)

    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top2.update(prec2.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)

  return top1.avg, objs.avg


def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top2 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(test_queue):

    # _, input, _, target = train_test_split(input, target, test_size=0.3, random_state=30)

    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()
    # logits, features, _ = model(input)
    logits, features = model(input)
    loss = criterion(logits, target)
    prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top2.update(prec2.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
