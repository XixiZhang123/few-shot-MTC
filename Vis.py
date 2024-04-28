
# £¡Users\GG\anaconda3\envs\zxx1\python
# -*- coding:utf-8 -*-
# author£ºZXX time:9/13/22

from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
import seaborn as sns
import torch.utils
from model import NetworkCIFAR as Network
import sys
import numpy as np
import utils
import logging
import argparse
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.autograd import Variable
# from model import NetworkCIFAR as Network
import torch.utils.data as dataloader
import torch
# from Dataset import MyData
import torch, gzip, os

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data/zhangxx/package/TFDDatabase-20-30', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
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
# classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']

# classes = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'SMB', 'Weibo', 'WorldOfWarcraft', 'Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
# classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# [0:'BitTorrent',1:'Facetime',2:'FTP',3:'Gmail',4:'MySQL',5:'Outlook',6:'Skype',7:'SMB',8:'Weibo',9:'WorldOfWarcraft',10:'Cridex',11:'Geodo',12:'Htbot',13:'Miuref',14:'Neris',15:'Nsis-ay',16:'Shifu',17:'Tinba',18:'Virut',19:'Zeus']


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

    # data, label = load_data('/data/zhangxx/SEI/MTC/FSL-MTC/TIFS_FS-MTC/USTC5Malware',
    #                          "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    data, label = load_data(
        '/data/zhangxx/SEI/MTC/FSL-MTC/TIFS_FS-MTC/CIC10lei',
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
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
    model = model.cuda()

    # utils.load(model, args.model_path)
    model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'), False)
    model.drop_path_prob = args.drop_path_prob
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    # model.drop_path_prob = args.drop_path_prob

    n_classes = 10
    model.eval()
    # X_test_feature = []
    for step, (input, target) in enumerate(test_queue):
        input = Variable(input, volatile=True).cuda()
        # target = Variable(target_1, volatile=True).cuda()
        feature = model(input)
        feature = feature.detach().cpu().numpy()
        if(step == 0):
            X_test_feature = feature
        else:
            X_test_feature = np.concatenate((X_test_feature, feature), axis = 0)
    X_test_feature = np.array(X_test_feature)
    print(X_test_feature.shape)
    tsne = TSNE(n_components=2)
    eval_tsne_embeds = tsne.fit_transform(X_test_feature)
    scatter(eval_tsne_embeds, label, "CIC", n_classes)
    # scatter(eval_tsne_embeds, label, "USTC2016", n_classes)


def scatter(features, labels, subtitle=None, n_classes = 5):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_classes))#"hls",
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=150, c = palette[labels, :])#
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.legend()

    txts = []
    for i in range(n_classes):
        xtext, ytext = np.median(features[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig(f"result/{subtitle}.png", dpi = 600)
    plt.show()
if __name__ == '__main__':
    main()
