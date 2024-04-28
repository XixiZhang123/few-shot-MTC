# -*- coding: utf-8 -*-
__author__ = "XU"
__date__ = '11/2/21 3:54 PM'

import os
import torch
import random
import numpy as np

from torchvision import transforms


class Hyperparameter:
    # #####################################################################
    #                            Data
    # #####################################################################
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    data_train_path = './data'
    data_eval_path = './data'
    data_test_path = './data'

    in_channels = 3
    out_dim = 10
    seed = 1234
    fc_drop_prob = 0.3
    share_parameters_frozen = False
    old_task_parameters_frozen = False
    new_task_parameters_frozen = False
    conv_frozen = False
    fc_frozen = False
    classes_num = 20
    add_num = 0

    # #####################################################################
    #                            Data
    # #####################################################################
    batch_size = 16
    init_lr = 0.001
    epochs = 40
    verbose_step = 250
    save_step = 500
    T = 2


HP = Hyperparameter()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Seed = seed_everything(HP.seed)


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # transforms.RandomRotation(degrees=45),
    # transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

