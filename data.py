from time import time
import numpy
import numpy as np
import csv
import scipy.io
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle
import os
import scipy.misc
from PIL import Image


def get_next_batch(X, Y, batch_size, shuffle=False):
    n_batches = len(X) // batch_size
    if shuffle:
        x_idx = np.random.permutation(len(X))[:n_batches * batch_size]
    else:
        x_idx = np.arange(n_batches * batch_size)
    for batch_idx in x_idx.reshape([n_batches, batch_size]):
        batch_x, batch_y = X[batch_idx], Y[batch_idx]
        yield batch_x, batch_y


def normalize(x_train, x_test):
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train_pixel_mean = x_train.mean(axis=0)  # per-pixel mean
    x_train_pixel_std = x_train.std(axis=0)  # per-pixel std
    x_train = (x_train - x_train_pixel_mean) / x_train_pixel_std
    x_test = (x_test - x_train_pixel_mean) / x_train_pixel_std
    return x_train, x_test

class Dataset:
    def __init__(self, batch_size, augm_flag):
        self.batch_size = batch_size
        self.augm_flag = augm_flag
        self.train_dataset = None
        self.test_dataset = None
        # Num workers is really important. For small datasets it should be 1 (0 slows down x2),
        # for large datasets 4*n_gpus maybe
        self.n_workers_train = 1
        self.n_workers_test = 1
        self.base_path = '../datasets/'

    @staticmethod
    def yield_data(iterator, n_batches):
        for i, (x, y) in enumerate(iterator):
            if type(x) != np.ndarray:
                x, y = x.numpy(), y.numpy()
            yield (x, y)
            if i + 1 == n_batches:
                break

    def get_train_batches(self, n_batches, shuffle):
        # Creation of a DataLoader object is instant, the queue starts to fill up on enumerate(train_loader)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle,
                                                   num_workers=self.n_workers_train, drop_last=True)
        return self.yield_data(train_loader, n_batches)

    def get_test_batches(self, n_batches, shuffle):
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=shuffle,
                                                  num_workers=self.n_workers_test, drop_last=True)
        return self.yield_data(test_loader, n_batches)


class GrayscaleDataset(Dataset):
    @staticmethod
    def yield_data(iterator, n_batches):
        """
        We need to redefine yield_data() to fix the fact that mnist by default is bs x 28 x 28 and not bs x 28 x 28 x 1
        """
        for i, (x, y) in enumerate(iterator):
            x = x[:, :, :, np.newaxis]  # bs x 28 x 28   ->   bs x 28 x 28 x 1
            x, y = x.numpy(), y.numpy()
            yield (x, y)
            if i + 1 == n_batches:
                break


class MNIST(GrayscaleDataset):
    def __init__(self, batch_size, augm_flag):
        super().__init__(batch_size, augm_flag)
        self.n_train, self.n_test = 60000, 10000
        self.n_classes = 10
        self.height, self.width, self.n_colors = 28, 28, 1
        self.data_dir = self.base_path + 'mnist/'

        transform_base = [transforms.Lambda(lambda x: np.array(x) / 255.0)]
        transform_train = transforms.Compose([
            transforms.RandomCrop(self.height, padding=4),
        ] + transform_base)
        transform_test = transforms.Compose(transform_base)
        transform_train = transform_train if self.augm_flag else transform_test
        self.train_dataset = datasets.MNIST(self.data_dir, train=True, transform=transform_train, download=True)
        self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=transform_test, download=True)


class FMNIST(GrayscaleDataset):
    def __init__(self, batch_size, augm_flag):
        super().__init__(batch_size, augm_flag)
        self.n_train, self.n_test = 60000, 10000
        self.n_classes = 10
        self.height, self.width, self.n_colors = 28, 28, 1
        self.data_dir = self.base_path + 'fmnist/'

        transform_base = [transforms.Lambda(lambda x: np.array(x) / 255.0)]
        transform_train = transforms.Compose([
            transforms.RandomCrop(self.height, padding=4),
        ] + transform_base)
        transform_test = transforms.Compose(transform_base)
        transform_train = transform_train if self.augm_flag else transform_test
        self.train_dataset = datasets.FashionMNIST(self.data_dir, train=True, transform=transform_train, download=True)
        self.test_dataset = datasets.FashionMNIST(self.data_dir, train=False, transform=transform_test, download=True)


class EMNIST(GrayscaleDataset):
    def __init__(self, batch_size, augm_flag):
        super().__init__(batch_size, augm_flag)
        self.n_train, self.n_test = 60000, 10000
        # TODO: actually, these numbers are smaller than the real ones.
        self.n_classes = 10
        self.height, self.width, self.n_colors = 28, 28, 1
        self.data_dir = self.base_path + 'emnist/'

        transform_base = [transforms.Lambda(lambda x: np.array(x).T / 255.0)]
        transform_train = transforms.Compose([
            transforms.RandomCrop(self.height, padding=4),
        ] + transform_base)
        transform_test = transforms.Compose(transform_base)
        transform_train = transform_train if self.augm_flag else transform_test
        self.train_dataset = datasets.EMNIST(self.data_dir, split='letters', train=True, transform=transform_train, download=True)
        self.test_dataset = datasets.EMNIST(self.data_dir, split='letters', train=False, transform=transform_test, download=True)


class CIFAR10(Dataset):
    def __init__(self, batch_size, augm_flag):
        super().__init__(batch_size, augm_flag)
        self.n_train, self.n_test = 50000, 10000
        self.n_classes = 10
        self.height, self.width, self.n_colors = 32, 32, 3
        self.data_dir = self.base_path + 'cifar10/'

        # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        # transformations = transforms.Compose([transforms.ToTensor(), normalize])
        transform_base = [transforms.Lambda(lambda x: np.array(x) / 255.0)]
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.height, padding=4),
        ] + transform_base)
        transform_test = transforms.Compose(transform_base)
        transform_train = transform_train if self.augm_flag else transform_test
        self.train_dataset = datasets.CIFAR10(self.data_dir, train=True, transform=transform_train, download=True)
        self.test_dataset = datasets.CIFAR10(self.data_dir, train=False, transform=transform_test, download=True)


class CIFAR10Grayscale(GrayscaleDataset):
    def __init__(self, batch_size, augm_flag):
        super().__init__(batch_size, augm_flag)
        self.n_train, self.n_test = 50000, 10000
        self.n_classes = 10
        self.height, self.width, self.n_colors = 28, 28, 3
        self.data_dir = self.base_path + 'cifar10/'

        # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        # transformations = transforms.Compose([transforms.ToTensor(), normalize])
        transform_base = [
            transforms.Resize(size=(self.height, self.width)),  # resize from 32x32 to 28x28
            transforms.Lambda(lambda x: np.array(x).mean(axis=2) / 255.0)  # make them black-and-white
        ]
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.height, padding=4),
        ] + transform_base)
        transform_test = transforms.Compose(transform_base)
        transform_train = transform_train if self.augm_flag else transform_test
        self.train_dataset = datasets.CIFAR10(self.data_dir, train=True, transform=transform_train, download=True)
        self.test_dataset = datasets.CIFAR10(self.data_dir, train=False, transform=transform_test, download=True)

class CIFAR100(Dataset):
    def __init__(self, batch_size, augm_flag):
        super().__init__(batch_size, augm_flag)
        self.n_train, self.n_test = 50000, 10000
        self.n_classes = 100
        self.height, self.width, self.n_colors = 32, 32, 3
        self.data_dir = self.base_path + 'cifar100/'

        # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        # transformations = transforms.Compose([transforms.ToTensor(), normalize])
        transform_base = [transforms.Lambda(lambda x: np.array(x) / 255.0)]
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.height, padding=4),
        ] + transform_base)
        transform_test = transforms.Compose(transform_base)
        transform_train = transform_train if self.augm_flag else transform_test
        self.train_dataset = datasets.CIFAR100(self.data_dir, train=True, transform=transform_train, download=True)
        self.test_dataset = datasets.CIFAR100(self.data_dir, train=False, transform=transform_test, download=True)


class SVHN(Dataset):
    def __init__(self, batch_size, augm_flag):
        super().__init__(batch_size, augm_flag)
        self.n_train, self.n_test = 73257, 26032
        self.n_classes = 10
        self.height, self.width, self.n_colors = 32, 32, 3
        self.data_dir = self.base_path + 'svhn/'

        # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        # transformations = transforms.Compose([transforms.ToTensor(), normalize])
        transform_base = [transforms.Lambda(lambda x: np.array(x) / 255.0)]
        transform_train = transforms.Compose([
            transforms.RandomCrop(self.height, padding=4),
        ] + transform_base)
        transform_test = transforms.Compose(transform_base)
        transform_train = transform_train if self.augm_flag else transform_test
        self.train_dataset = datasets.SVHN(self.data_dir, split='train', transform=transform_train, download=True)
        self.test_dataset = datasets.SVHN(self.data_dir, split='test', transform=transform_test, download=True)
        

class LSUNClassroom(Dataset):
    def __init__(self, batch_size, augm_flag, test_only=True):
        super().__init__(batch_size, augm_flag)
        self.n_train, self.n_test = 168103, 300
        self.n_classes = 1  # i.e. only the classroom
        self.height, self.width, self.n_colors = 32, 32, 3
        self.data_dir = '/scratch/maksym'
        #self.data_dir = self.base_path + 'lsun/'

        # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        # transformations = transforms.Compose([transforms.ToTensor(), normalize])
        transform_base = [transforms.Lambda(lambda x: np.array(x) / 255.0)]
        transform_train = transforms.Compose([
            transforms.Resize(size=(self.height, self.width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.height, padding=4),
        ] + transform_base)
        transform_test = transforms.Compose([
            transforms.Resize(size=(self.height, self.width))
        ] + transform_base)
        transform_train = transform_train if self.augm_flag else transform_test
        #self.train_dataset = torch.utils.data.Subset(datasets.LSUN(self.data_dir, classes=['classroom_train'], transform=transform_train), range(0, 10000))
        if test_only:
            self.test_dataset = datasets.LSUN(self.data_dir, classes=['classroom_val'], transform=transform_test)
        else:
            self.test_dataset = torch.utils.data.Subset(datasets.LSUN(self.data_dir, classes=['classroom_train'], transform=transform_test), range(0, 10000))
        print('LSUNClassroom __init__ done')


class ImageNetMinusCifar10(Dataset):
    def __init__(self, batch_size, augm_flag):
        super().__init__(batch_size, augm_flag)
        self.n_train, self.n_test = 500000, 30000
        self.n_classes = 853
        self.height, self.width, self.n_colors = 32, 32, 3
        #self.train_dir = self.base_path + 'imagenet_minus_cifar10/imagenet/train'
        self.test_dir = self.base_path + 'imagenet_minus_cifar10/imagenet/val_orig'
        
        # normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        # transformations = transforms.Compose([transforms.ToTensor(), normalize])
        transform_base = [transforms.Lambda(lambda x: np.array(x) / 255.0)]
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.height, padding=4),
        ] + transform_base)
        transform_test = transforms.Compose(transform_base)
        transform_train = transform_train if self.augm_flag else transform_test
        #self.train_dataset = torch.utils.data.Subset(datasets.ImageFolder(self.test_dir, transform=transform_test), range(0,10000))
        self.test_dataset = torch.utils.data.Subset(datasets.ImageFolder(self.test_dir, transform=transform_test), np.random.permutation(range(self.n_test))[:10000])
        print('ImageNetMinusCifar10 __init__ done')
        

# All available datasets
datasets_dict = {'mnist':          MNIST,
                 'fmnist':         FMNIST,
                 'cifar10_gray':   CIFAR10Grayscale,
                 'emnist':         EMNIST,
                 'cifar10':        CIFAR10,
                 'cifar100':       CIFAR100,
                 'svhn':           SVHN,
                 'lsun_classroom': LSUNClassroom,
                 'imagenet_minus_cifar10':  ImageNetMinusCifar10,  # no automatic download; you have to have it on disk
                 }