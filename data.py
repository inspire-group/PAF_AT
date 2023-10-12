import os
import scipy
import scipy.io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle

# ImageNet type datasets, i.e., which support loading with ImageFolder
def imagenette(datadir="data_dir", batch_size=128, mode="org", size=224, normalize=False, norm_layer=None, workers=4, distributed=False, **kwargs):
    # mode: base | org
    
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.4648, 0.4543, 0.4247], std=[0.2785, 0.2735, 0.2944])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    transform_train = transforms.Compose([transforms.RandomResizedCrop(size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       norm_layer
                       ])
    transform_test = transforms.Compose([transforms.Resize(int(1.14*size)),
                      transforms.CenterCrop(size),
                      transforms.ToTensor(), 
                      norm_layer])
    
    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
        
    trainset = datasets.ImageFolder(
        os.path.join(datadir, "train"), 
        transform=transform_train)
    testset = datasets.ImageFolder(
        os.path.join(datadir, "val"), 
        transform=transform_test)
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


# cifar10
def cifar10(datadir="data_dir", batch_size=128, mode="org", size=32, normalize=False, norm_layer=None, workers=4, distributed=False, **kwargs):
    # mode: base | org
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    trtrain = [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor(), norm_layer]
    if size != 32:
        trtrain = [transforms.Resize(size)] + trtrain
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor(), norm_layer]
    if size != 32:
        trval = [transforms.Resize(size)] + trval
    transform_test = transforms.Compose(trval)

    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
        
    trainset = datasets.CIFAR10(
            root=os.path.join(datadir, "cifar10"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.CIFAR10(
            root=os.path.join(datadir, "cifar10"),
            train=False,
            download=True,
            transform=transform_test,
        )
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train

# cifar100
def cifar100(datadir="data_dir", batch_size=128, mode="org", size=32, normalize=False, norm_layer=None, workers=4, distributed=False, **kwargs):
    # mode: base | org
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

    trtrain = [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(),
          transforms.ToTensor(), norm_layer]
    if size != 32:
        trtrain = [transforms.Resize(size)] + trtrain
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor(), norm_layer]
    if size != 32:
        trval = [transforms.Resize(size)] + trval
    transform_test = transforms.Compose(trval)

    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")

    trainset = datasets.CIFAR100(
            root=os.path.join(datadir, "cifar100"),
            train=True,
            download=True,
            transform=transform_train
        )
    testset = datasets.CIFAR100(
            root=os.path.join(datadir, "cifar100"),
            train=False,
            download=True,
            transform=transform_test,
        )

    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)

    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


class custom_loader_from_image_list(torch.utils.data.DataLoader):
    def __init__(self, f, label_extractor, classes=None, training_images=None, transform=None):
        self.images = np.genfromtxt(f, delimiter=',', dtype=str) 
        self.labels = np.array([label_extractor(i) for i in self.images])
        self.transform = transform
        print(f"Loaded original data of {len(self.labels)} images from {f}")
        
        total = len(self.labels)
        if training_images:
            print("Assuming that indexes of stored images are pre-sorted")
            print(f"Selecting first {training_images} images from total {total} available images")
            self.images = self.images[:training_images]
            self.labels = self.labels[:training_images]
        
        if classes:
            print(f"Loading only class {classes} images.")
            valid_indices = []
            new_labels = []
            for i, index in enumerate(classes):
                temp = np.where(self.labels==index)[0]
                valid_indices += list(temp)
                new_labels += [i] * len(temp)
            
            self.images = self.images[valid_indices]
            self.labels = np.array(new_labels)
            # lets just shuffle them to ease our conscience, in case we miss shuffling in dataloader
            indices = np.random.permutation(np.arange(len(self.labels)))
            self.images, self.labels = self.images[indices], self.labels[indices]
            print(f"Carved dataset for only {classes} classes comprising {len(self.labels)} images")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = Image.open(self.images[idx]), self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def diffusion_cifar10(datadir="data_dir", batch_size=128, mode="org", size=32, normalize=False, norm_layer=None, workers=4, distributed=False, classes=None, training_images=None, **kwargs):
    # mode: base | org
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[0., 0., 0.])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    trtrain = [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor(), norm_layer]
    if size != 32:
        trtrain = [transforms.Resize(size)] + trtrain
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor(), norm_layer]
    if size != 32:
        trval = [transforms.Resize(size)] + trval
    transform_test = transforms.Compose(trval)

    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
    
    datadir = "data_dir/cifar5m"
    print(f"Discarding args.datadir and loading data from fixed source: {datadir}")
    label_extractor = lambda x: int(x.split("/")[-2])
    
    trainset = custom_loader_from_image_list(os.path.join(datadir, "train_split_seed_0.txt"), label_extractor, classes, training_images, transform_train)
    testset = custom_loader_from_image_list(os.path.join(datadir, "test_split_seed_0.txt"), label_extractor, classes, None, transform_test)
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train


def styleganC_cifar10(datadir="data_dir", batch_size=128, mode="org", size=32, normalize=False, norm_layer=None, workers=4, distributed=False, classes=None, training_images=None, **kwargs):
    # mode: base | org
    if norm_layer is None:
        if normalize:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[0., 0., 0.])
        else:
            norm_layer = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
    
    trtrain = [transforms.RandomCrop(size, padding=4), transforms.RandomHorizontalFlip(), 
          transforms.ToTensor(), norm_layer]
    if size != 32:
        trtrain = [transforms.Resize(size)] + trtrain
    transform_train = transforms.Compose(trtrain)
    trval = [transforms.ToTensor(), norm_layer]
    if size != 32:
        trval = [transforms.Resize(size)] + trval
    transform_test = transforms.Compose(trval)

    if mode == "org":
        None
    elif mode == "base":
        transform_train = transform_test
    else:
        raise ValueError(f"{mode} mode not supported")
    
    datadir = "data_dir"
    print(f"Discarding args.datadir and loading data from fixed source: {datadir}")
    label_extractor = lambda x: int(x.split("/")[-2].split("_")[-1])
    
    trainset = custom_loader_from_image_list(os.path.join(datadir, "conditional_train_split_seed_0.txt"), label_extractor, classes, training_images, transform_train)
    testset = custom_loader_from_image_list(os.path.join(datadir, "conditional_test_split_seed_0.txt"), label_extractor, classes, None, transform_test)
    
    train_sampler, test_sampler = None, None
    if distributed:
        print("Using DistributedSampler")
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=workers, pin_memory=True)
    
    return train_loader, train_sampler, test_loader, test_sampler, None, None, transform_train




