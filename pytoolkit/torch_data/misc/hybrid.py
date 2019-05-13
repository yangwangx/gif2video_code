from PIL import Image
import numpy as np
import random

import torch
import torch.utils.data as DD
import torchvision.transforms as TT

normalize = TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = TT.Compose([
    TT.RandomResizedCrop(224),
    TT.RandomHorizontalFlip(),
    TT.ToTensor(),
    normalize,
])

valid_transform = TT.Compose([
    TT.Resize(256),
    TT.CenterCrop(224),
    TT.ToTensor(),
    normalize,
])

def get_info(splitFile):
    Vds, Lbs = [], []
    with open(splitFile) as f:
        for line in f.readlines():
            tmp = line.rstrip().split(' ')
            Vds.append(tmp[0])
            Lbs.append(int(tmp[1]))
    return Vds, Lbs

class hybrid_train(DD.Dataset):
    def __init__(self, balance=1300, transform=train_transform):
        super(hybrid_train, self).__init__()
        splitFile = '/mnt/disk1/yangwang/hybrid/hybrid_train.txt'
        Vds, Lbs = get_info(splitFile)

        # balanced sampling
        HashT = {}
        for i, label in enumerate(Lbs):
            if label in HashT:
                HashT[label].append(i)
            else:
                HashT[label] = [i,]

        INXs = []
        for k in HashT:
            inxs = HashT[k]
            repeat, remain = divmod(balance, len(inxs))
            inxs = inxs * repeat + random.sample(inxs, remain)
            INXs.extend(inxs)

        self.Vds = []
        self.Lbs = []
        for i in INXs:
            self.Vds.append(Vds[i])
            self.Lbs.append(Lbs[i])
        self.balance = balance
        self.transform = transform

    def __getitem__(self, index):
        img = self.transform(Image.open(self.Vds[index]))
        img = img.expand(3, -1, -1)
        cls = self.Lbs[index] # already 0-based
        return img, cls

    def __len__(self):
        return len(self.Vds)

class hybrid_valid(DD.Dataset):
    def __init__(self, transform=valid_transform):
        super(hybrid_valid, self).__init__()
        splitFile = '/mnt/disk1/yangwang/hybrid/hybrid_valid.txt'
        Vds, Lbs = get_info(splitFile)
        self.Vds = Vds
        self.Lbs = Lbs
        self.transform = transform

    def __getitem__(self, index):
        img = self.transform(Image.open(self.Vds[index]))
        img = img.expand(3, -1, -1)
        cls = self.Lbs[index] # already 0-based
        return img, cls

    def __len__(self):
        return len(self.Vds)

if __name__ == '__main__':
    dataset = hybrid_valid()
    img, cls = random.choice(dataset)
    TT.ToPILImage()(img*0.23 + 0.45).show()
