import torch
import torch.utils.data as DD
import torchvision.transforms as TT

import imageio
from PIL import Image
import numpy as np
import random

import os, sys, glob
try:
    cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
except NameError:
    cwd = ''

im2tensor = lambda x: torch.ByteTensor(np.moveaxis(x, 2, 0))
tensor2im = lambda x: np.moveaxis(x.numpy(), 0, 2)
imshow = lambda x: Image.fromarray(x).show()

class pix2pixDataset(DD.Dataset):
    def __init__(self, name='facades', subset='train', direction ='AtoB', patchSize=None):
        super(pix2pixDataset, self).__init__()
        imgDir = os.path.join(cwd, 'datasets', name, subset)
        imgList = glob.glob(os.path.join(imgDir, '*.png'))
        imgList += glob.glob(os.path.join(imgDir, '*.jpg'))
        imgList.sort()
        self.name = name
        self.imgList = imgList
        self.direction = direction
        self.patchSize = patchSize

    def __getitem__(self, index):
        imgFile = self.imgList[index]
        img = imageio.imread(imgFile)
        H, WW, C = img.shape
        W = WW // 2
        A, B = img[:, :W, :], img[:, W:, :]
        A, B = im2tensor(A), im2tensor(B)

        if self.patchSize is not None:
            y1 = random.randrange(H - self.patchSize + 1)
            x1 = random.randrange(W - self.patchSize + 1)
            crop = lambda x: x[:, y1:y1+self.patchSize, x1:x1+self.patchSize + 1]        
            A, B = crop(A), crop(B)
        
        if self.direction == 'AtoB':
            return A, B
        else:
            return B, A
    
    def __len__(self):
        return len(self.imgList)

if __name__ == '__main__':
    dataset = pix2pixDataset(patchSize=256)
    A, B = random.choice(dataset)
    imshow(tensor2im(A))
    imshow(tensor2im(B))
