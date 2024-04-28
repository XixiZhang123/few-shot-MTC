import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from tensorboard_logger import configure, log_value
import torch.utils.data as dataloader
from torchvision import transforms
from torch.utils.data import Dataset
import torch,gzip,os
from sklearn.model_selection import train_test_split

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
  with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb±íÊ¾µÄÊÇ¶ÁÈ¡¶þ½øÖÆÊý¾Ý
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
    x_train = np.frombuffer(
      imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
  return (x_train, y_train)


x, y = DealDataset(
  '/data/zhangxx/SEI/MAT/1.malware_traffic_classification/3.PreprocessedResults/20class/FlowAllLayers/',
  "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.3, random_state=30)
train_Queue = torch.utils.data.TensorDataset(X_train, Y_train)
valid_Queue = torch.utils.data.TensorDataset(X_val, Y_val)
train_queue = torch.utils.data.DataLoader(train_Queue, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                            num_workers=2)

valid_queue = torch.utils.data.DataLoader(valid_Queue, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                            num_workers=2)


testDataset = DealDataset(
  '/data/zhangxx/SEI/MAT/1.malware_traffic_classification/3.PreprocessedResults/20class/FlowAllLayers/',
  "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

train_queue = dataloader.DataLoader(
  dataset=trainDataset,
  batch_size=100,
  shuffle=False,
)


test_queue = dataloader.DataLoader(
  dataset=testDataset,
  batch_size=100,
  shuffle=False,
)