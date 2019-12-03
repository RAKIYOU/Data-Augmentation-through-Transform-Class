#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torchvision.datasets.vision as vision
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import codecs
from torchvision.datasets.utils import  download_and_extract_archive,  makedir_exist_ok
import torchvision.models as  models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils as utils
import torch


class MyMNIST(vision.VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, limit_data=None, resize=False):
        super(MyMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        # ------- We can ingore this block -------
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        # -----------------------------------------------

        # We change the lines below; these specify how the data are loaded.
        # self.data contain images (type 'torch.Tensor' of [num_images, H, W])
        # self.targets contain labels (class ids) (type 'torch.Tensor' of [num_images])
        # images and labels are stored in the same
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        # We use only the images and lables whose indeces are in the range of 0..limit_data-1.
        if not limit_data is None:
          self.data    = self.data[ :limit_data, :,:]
          self.targets = self.targets[:limit_data]
          if self.train:
            print("[WRN]: Trainig Data is limited, only the first "+str(self.data.size(0))+" samples will be used.")
          else:
            print("[WRN]: Test Data is limited, only the first "   +str(self.data.size(0))+" samples will be used.")


    def __getitem__(self, index):
        # We extract the image and label of the specified 'index'.
        img, target = self.data[index], int(self.targets[index])

        # Prepare for self.transform below.
        img = Image.fromarray(img.numpy(), mode='L')

        # Transform img.
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()

device = torch.device("cuda:0")
net = models.MobileNetV2()
net.features[0][0]= nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
net.classifier[1]=nn.Linear(in_features=1280, out_features=10, bias=True)

if torch.cuda.device_count() >1:
    net = nn.DataParallel(net)
net.to(device)
transformnone=transforms.Compose([ transforms.Resize((224,224)),  transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform1 = transforms.Compose([ transforms.Resize((224,224)),  transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

transform2 = transforms.Compose([  transforms.Resize((300,300)), transforms.CenterCrop(224),      transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform3 = transforms.Compose([  transforms.Resize((300,300)), transforms.RandomCrop(224),        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

transform4 = transforms.Compose([  transforms.Resize((224,224)), transforms.RandomVerticalFlip(p=0.5),  transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

transform5 = transforms.Compose([  transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=0.5),  transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


#transform6 = transforms.Compose([  transforms.Resize(300), transforms.FiveCrop(224),  transforms.Lambda( lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.ToTensor(),   transforms.Normalize((0.1307,), (0.3081,))])

transform6 = transforms.Compose([  transforms.Resize((224,224)), transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),  transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


transformlist = [   transforms.RandomVerticalFlip(p=0.5),   transforms.RandomHorizontalFlip(p=0.5),  transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)        ]

transform7 = transforms.Compose([  transforms.Resize(224),  transforms.RandomApply(transformlist, p=0.5),   transforms.ToTensor(),   transforms.Normalize((0.1307,), (0.3081,))])
#transform = transform7


mnist_train = MyMNIST('./data', train=True,  download=True, transform=transform7, limit_data=1000)

mnist_test  = MyMNIST('./data', train=False, download=True, transform=transformnone)
trainloader = utils.data.DataLoader(mnist_train, batch_size=50, shuffle=True,  num_workers=4)
testloader  = utils.data.DataLoader(mnist_test, batch_size=120, shuffle=False, num_workers=4)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def evaluate_model():
  print("Testing the network...")
  net.eval()
  total_num   = 0
  correct_num = 0
  for test_iter, test_data in enumerate(testloader):
    inputs, labels = test_data    
    bch = inputs.size(0)
    inputs = inputs.to(device)
    labels = torch.LongTensor(list(labels)).to(device)
    outputs = net(inputs)
    _, pred_cls = torch.max(outputs, 1)
    correct_num+= (pred_cls == labels).float().sum().item()
    total_num+= bch
  net.train() 
  print("Accuracy: "+"%.5f"%(correct_num/float(total_num)))


'''

def evaluate_model1():
  print("Testing the network...")
  net.eval()
  total_num   = 0
  correct_num = 0
  for test_iter, test_data in enumerate(testloader):
    inputs, labels = test_data
    bs, ncrops, c, h, w = inputs.size()
    inputs = inputs.to(device)
    labels = torch.LongTensor(list(labels)).to(device)
    outputs = net(inputs.view(-1,c,h,w))
    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
    _, pred_cls = torch.max(outputs_avg, 1)
    correct_num+= (pred_cls == labels).float().sum().item()
    total_num+= bs
  net.train()
  print("Accuracy: "+"%.5f"%(correct_num/float(total_num)))

'''

epoch_size = 40
epoch = 0



for epoch_idx in range(epoch_size):
  running_loss = 0.0
  ct_num       = 0
  for iteration, data in enumerate(trainloader):
    inputs, labels = data
    bch = inputs.size(0)
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss    = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    ct_num+= 1

    if iteration%2 == 1:
        print("[Epoch: "+str(epoch+1)+"]"" --- Iteration: "+str(iteration+1)+", Loss: "+str(running_loss/ct_num)+'.')
  epoch += 1

evaluate_model()
