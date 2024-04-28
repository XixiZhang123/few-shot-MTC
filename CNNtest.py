# £¡Users\GG\anaconda3\envs\zxx1\python
# -*- coding:utf-8 -*-
# author£ºZXX time:9/13/22

from sklearn.linear_model import LogisticRegression
import torch.utils
from model import NetworkCIFAR as Network
import sys
import numpy as np
import utils
import random
import logging
import argparse
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# from model import NetworkCIFAR as Network
from model import CNN, VGG
import torch.utils.data as dataloader
import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from warnings import filterwarnings
from torch.utils.data import Dataset
import torch, gzip, os

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data/zhangxx/package/TFDDatabase-20-30', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--model_path', type=str, default='model/NASP40lei15(2).pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='NASP40lei15', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CLASSES = 40

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


    def load_data(data_folder, data_name, label_name):
        with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb¡À¨ª¨º?¦Ì?¨º??¨¢¨¨??t????¨ºy?Y
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        return (x_train, y_train)

    def TrainDatasetKshotRround(num, k):
        X, Y = load_data('/data/zhangxx/SEI/MTC/FSL-MTC/TIFS_FS-MTC/USTC5Malware',
                         "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
        # X, Y = load_data('/data/zhangxx/SEI/MTC/FSL-MTC/TIFS_FS-MTC/CIC10lei',
        #                  "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")

        Y = Y.astype(np.uint8)
        sorted_inx = np.argsort(Y)
        X = X.reshape(-1, 1, 28, 28)
        x = X[sorted_inx, :, :, :]
        y = Y[sorted_inx]
        List_train = y.tolist()

        X_train_K_Shot = np.zeros([int(k * num), 1, 28, 28])
        Y_train_K_Shot = np.zeros([int(k * num)])

        for i in range(num):
            index_train_start = List_train.index(i)
            if i == num - 1:
                index_train_end = y.shape[0]
            else:
                index_train_end = List_train.index(i + 1) - 1
            index_shot = range(index_train_start, index_train_end)
            random_shot = random.sample(index_shot, k)

            X_train_K_Shot[i * k:i * k + k, :, :, :] = x[random_shot, :, :, :]
            Y_train_K_Shot[i * k:i * k + k] = y[random_shot]
        return X_train_K_Shot, Y_train_K_Shot

    data, label = load_data(
        '/data/zhangxx/SEI/MTC/FSL-MTC/TIFS_FS-MTC/USTC5Malware',
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    # data, label = load_data(
    #     '/data/zhangxx/SEI/MTC/FSL-MTC/TIFS_FS-MTC/CIC10lei',
    #     "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    test_label = label.reshape(-1)
    data_train = torch.from_numpy(data)
    data_train = data_train.type(torch.FloatTensor)
    label_test = torch.from_numpy(label)
    label_test = label_test.type(torch.LongTensor)
    label_test = label_test.squeeze()
    data_train = data_train.reshape(-1, 1, 28, 28)
    data_train = data_train.cuda()
    label_test = label_test.cuda()
    test_data = torch.utils.data.TensorDataset(data_train, label_test)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    # model = CNN()
    model = model.cuda()

    # utils.load(model, args.model_path)
    model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'), False)
    # model.load_state_dict(torch.load(args.model_path)['state_dict'], False)
    model.drop_path_prob = args.drop_path_prob
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model.drop_path_prob = args.drop_path_prob
    model.eval()

    for step, (input, target) in enumerate(test_queue):
        input = Variable(input, volatile=True).cuda()
        # target = Variable(target_1, volatile=True).cuda()
        feature = model(input)
        feature = feature.detach().cpu().numpy()
        if(step == 0):
            X_test_feature = feature
        else:
            X_test_feature = np.concatenate((X_test_feature, feature), axis=0)
    X_test_feature = np.array(X_test_feature)
    print('feature', X_test_feature.shape)


    filterwarnings('ignore')
    Ks = [1, 3, 5, 10]
    num_Ks = 4
    Ns = [5]
    num_Ns = 1
    Rs = 100
    acc = np.zeros([int(num_Ks * num_Ns), Rs])

    import time
    TT = 0.0
    t1 = time.time()
    for r in range(Rs):

        print(f"--------r={r}---------")
        for n in range(num_Ns):
            X_test_feature = X_test_feature
            for k in range(num_Ks):
                x, y = TrainDatasetKshotRround(Ns[n], Ks[k])
                x = torch.from_numpy(x)
                x = x.type(torch.FloatTensor)
                x = x.cuda()
                label_train = torch.from_numpy(y)
                label_train = label_train.type(torch.LongTensor)
                label_train = label_train.squeeze()
                label_train = label_train.cuda()
                train_data = torch.utils.data.TensorDataset(x, label_train)
                train_queue = torch.utils.data.DataLoader(
                    train_data, batch_size=2, shuffle=False, num_workers=0)
                for step, (input, target) in enumerate(train_queue):
                    input = Variable(input, volatile=True).cuda()
                    # target = Variable(target_1, volatile=True).cuda()
                    feature = model(input)
                    feature = feature.detach().cpu().numpy()
                    if(step == 0):
                        X_train_feature = feature
                    else:
                        X_train_feature = np.concatenate((X_train_feature, feature), axis = 0)
                X_train_feature = np.array(X_train_feature)
                x_feature = X_train_feature
                # clf = RandomForestClassifier()
                clf = LogisticRegression()  # KNeighborsClassifier()
                # clf = ExtraTreesClassifier()
                clf.fit(x_feature, y)
                # print('X_test_feature', X_test_feature.shape)
                # print('test_label', test_label.shape)
                acc[n * num_Ks + k, r] = clf.score(X_test_feature, test_label)
    t2 = time.time()
    TT = t2 - t1
    print('USTC time', TT)
    T = (t2 - t1)/Rs
    print('USTC Avg time', T)

    df = pd.DataFrame(acc)
    df.to_excel(f"result/FS-NASP_USTC_time.xlsx", index=False)
    # df.to_excel(f"result/FS-NASP_CIC_8channel.xlsx", index=False)

    acc_3m = np.zeros([int(num_Ks * num_Ns), 3])
    acc_3m[:, 0] = np.mean(acc, axis=1)
    acc_3m[:, 1] = np.max(acc, axis=1)
    acc_3m[:, 2] = np.min(acc, axis=1)
    print(acc_3m)

    df = pd.DataFrame(acc_3m, columns=['mean', 'max', 'min'])
    df.to_excel(f"result/FS-NASP_USTC_time_3m.xlsx", index=False)
    # df.to_excel(f"result/FS-NASP_CIC_3m.xlsx", index=False)

if __name__ == '__main__':
    main()

