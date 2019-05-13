import torch
import torch.utils.data as DD

import imageio
from PIL import Image
import numpy as np
import random

dataRoot = '/nfs/bigbrain/yangwang/Gif2Video/gif2video_data/gif_moments/'
targetRoot = dataRoot + 'frames/size360_div12/'
inputRoot = dataRoot + 'gifs/size360_div12_s1_g32_nodither/'

def get_videoList(splitFile=dataRoot+'split/video_info_train.txt'):
    videoList = []
    with open(splitFile) as f:
        f.readline()
        for line in f.readlines():
            tmp = line.rstrip().split(' ')
            VID, nFrm = tmp[0], int(tmp[3])
            videoList.append([VID, nFrm])
    return videoList

# train-valid-test split ratio: 85%-5%-10% 
videoList_train = get_videoList(dataRoot+'split/video_info_train.txt')
videoList_valid = get_videoList(dataRoot+'split/video_info_valid.txt')
videoList_test  = get_videoList(dataRoot+'split/video_info_test.txt')

def get_subsetList(subset):
    if subset == 'train':
        return videoList_train
    elif subset == 'valid':
        return videoList_valid
    elif subset == 'trval':
        return videoList_train + videoList_valid
    elif subset == 'test':
        return videoList_test
    elif subset == 'all':
        return videoList_train + videoList_valid + videoList_test
    else:
        raise ValueError('subset value {} is not valid!'.format(subset))

def getPaletteInRgb(gifFile):
    gif = Image.open(gifFile)
    assert gif.mode == 'P', "image should be palette mode"
    nColor = len(gif.getcolors())
    pal = gif.getpalette()
    colors = list(list(pal[i:i+3]) for i in range(0, len(pal), 3))
    return nColor, colors

class gif_moments_train(DD.Dataset):
    def __init__(self, inputRoot=inputRoot, targetRoot=targetRoot, subset='train', sCrop=256):
        super(gif_moments_train, self).__init__()
        self.inputRoot = inputRoot
        self.targetRoot = targetRoot
        self.videoList = get_subsetList(subset)
        self.sCrop = sCrop

    def __getitem__(self, index):
        VID, nFrm = self.videoList[index]
        def inputFile(t):
            im = imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, t))
            if im.ndim == 2:
                return np.broadcast_to(np.expand_dims(im, 2), list(im.shape) + [3])
            elif im.ndim == 3:
                return im[:,:,:3]

        def targetFile(t):
            im = imageio.imread('{}/{}/frame_{:06d}.jpg'.format(self.targetRoot, VID, t))
            if im.ndim == 2:
                return np.broadcast_to(np.expand_dims(im, 2), list(im.shape) + [3])
            elif im.ndim == 3:
                return im

        def gifPalette(t):
            im = '{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, t)
            nColor, colors = getPaletteInRgb(im)
            return colors

        # config
        sCrop = self.sCrop
        sFlip = random.random() > 0.5

        # temporal crop
        sFrm, eFrm = 1, nFrm
        t0 = random.randrange(sFrm, eFrm+1)

        # spatial crop and flip
        gif = inputFile(t0)
        H, W, _ = gif.shape
        y, x = random.randrange(H - sCrop + 1), random.randrange(W - sCrop + 1)
        def proc(im):
            crop = im[y:y+sCrop, x:x+sCrop]
            return np.flip(crop, axis=1) if sFlip else crop

        gif = proc(gif)
        target = proc(targetFile(t0))
        colors = gifPalette(t0) 

        # numpy to tensor
        gif = torch.ByteTensor(np.moveaxis(gif, 2, 0).copy()) # C H W
        target = torch.ByteTensor(np.moveaxis(target, 2, 0).copy()) # C H W
        colors = torch.ByteTensor(colors)
        return gif, target, colors

    def __len__(self):
        return len(self.videoList)

class gif_moments_eval(DD.Dataset):
    def __init__(self, inputRoot=inputRoot, targetRoot=targetRoot, subset='valid', tStride=4):
        super(gif_moments_eval, self).__init__()
        self.inputRoot = inputRoot
        self.targetRoot = targetRoot
        self.videoList = get_subsetList(subset)
        self.tStride = tStride

    def __getitem__(self, index):
        VID, nFrm = self.videoList[index]
        def inputFile(t):
            im = imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, t))
            if im.ndim == 2:
                return np.broadcast_to(np.expand_dims(im, 2), list(im.shape) + [3])
            elif im.ndim == 3:
                return im[:,:,:3]

        def targetFile(t):
            im = imageio.imread('{}/{}/frame_{:06d}.jpg'.format(self.targetRoot, VID, t))
            if im.ndim == 2:
                return np.broadcast_to(np.expand_dims(im, 2), list(im.shape) + [3])
            elif im.ndim == 3:
                return im

        def gifPalette(t):
            im = '{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, t)
            nColor, colors = getPaletteInRgb(im)
            return colors

        ts = np.arange(1, nFrm+1, self.tStride)
        gifs = list(inputFile(t) for t in ts)
        targets = list(targetFile(t) for t in ts)
        colors = list(gifPalette(t) for t in ts)

        # numpy to tensor
        gifs = torch.ByteTensor(np.moveaxis(np.asarray(gifs), 3, 1)) # N C H W
        targets = torch.ByteTensor(np.moveaxis(np.asarray(targets), 3, 1)) # N C H W
        colors = torch.ByteTensor(colors)
        return gifs, targets, colors

    def __len__(self):
        return len(self.videoList)

if __name__ == '__main__':
    tensor2im = lambda x: np.moveaxis(x.numpy(), 0, 2)
    if True:
        dataset = gif_moments_train()
        print('training set is of {} videos'.format(len(dataset)))
        # gif, target, colors = random.choice(dataset)
        # Image.fromarray(np.concatenate(list(map(tensor2im, [gif, target])), axis=1)).show()
    if True:
        dataset = gif_moments_eval(subset='valid')
        print('validation set is of {} videos'.format(len(dataset)))
        # gifs, targets, colors = random.choice(dataset)
        # Image.fromarray(np.concatenate(list(map(tensor2im, [gifs[0], targets[0]])), axis=1)).show()
    if True:
        dataset = gif_moments_eval(subset='test')
        print('test set is of {} videos'.format(len(dataset)))
        # gifs, targets, colors = random.choice(dataset)
        # Image.fromarray(np.concatenate(list(map(tensor2im, [gifs[0], targets[0]])), axis=1)).show()
