import torch
import torch.utils.data as DD

import imageio
from PIL import Image
import numpy as np
import random

__all__ = ['gif_moments_DithRec_train', 'gif_moments_DithRec_eval']

dataRoot = '/nfs/bigbrain/yangwang/Gif2Video/gif2video_data/gif_moments/'
dithRoot = dataRoot + 'gifs/size360_div12_s1_g32_dither/'
nodithRoot = dataRoot + 'gifs/size360_div12_s1_g32_nodither/'

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

class gif_moments_DithRec_train(DD.Dataset):
    def __init__(self, dithRoot=dithRoot, nodithRoot=nodithRoot, subset='train', sCrop=256):
        super(gif_moments_DithRec_train, self).__init__()
        self.dithRoot = dithRoot
        self.nodithRoot = nodithRoot
        self.videoList = get_subsetList(subset)
        self.sCrop = sCrop

    def __getitem__(self, index):
        VID, nFrm = self.videoList[index]
        def dithFile(t):
            im = imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.dithRoot, VID, t))
            if im.ndim == 2:
                im = np.broadcast_to(np.expand_dims(im, 2), list(im.shape) + [3])
            elif im.ndim == 3:
                im = im[:,:,:3]
            return im

        def nodithFile(t):
            im = imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.nodithRoot, VID, t))
            if im.ndim == 2:
                return np.broadcast_to(np.expand_dims(im, 2), list(im.shape) + [3])
            elif im.ndim == 3:
                return im[:,:,:3]
        # config
        sCrop = self.sCrop
        sFlip = random.random() > 0.5

        # temporal crop
        sFrm, eFrm = 1, nFrm
        t0 = random.randrange(sFrm, eFrm+1)

        # spatial crop and flip
        gif = dithFile(t0)
        H, W, _ = gif.shape
        y, x = random.randrange(H - sCrop + 1), random.randrange(W - sCrop + 1)
        def proc(im):
            crop = im[y:y+sCrop, x:x+sCrop]
            return np.flip(crop, axis=1) if sFlip else crop

        dith = proc(dithFile(t0))
        nodith = proc(nodithFile(t0))

        # numpy to tensor
        dith = torch.ByteTensor(np.moveaxis(dith, 2, 0).copy()) # C H W
        nodith = torch.ByteTensor(np.moveaxis(nodith, 2, 0).copy()) # C H W
        
        return dith, nodith

    def __len__(self):
        return len(self.videoList)

class gif_moments_DithRec_eval(DD.Dataset):
    def __init__(self, dithRoot=dithRoot, nodithRoot=nodithRoot, subset='test', tStride=10):
        super(gif_moments_DithRec_eval, self).__init__()
        self.dithRoot = dithRoot
        self.nodithRoot = nodithRoot
        self.videoList = get_subsetList(subset)
        self.tStride = tStride

    def __getitem__(self, index):
        VID, nFrm = self.videoList[index]
        def dithFile(t):
            im = imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.dithRoot, VID, t))
            if im.ndim == 2:
                im = np.broadcast_to(np.expand_dims(im, 2), list(im.shape) + [3])
            elif im.ndim == 3:
                im = im[:,:,:3]
            return im

        def nodithFile(t):
            im = imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.nodithRoot, VID, t))
            if im.ndim == 2:
                return np.broadcast_to(np.expand_dims(im, 2), list(im.shape) + [3])
            elif im.ndim == 3:
                return im[:,:,:3]

        ts = np.arange(1, nFrm+1, self.tStride)
        diths = list(dithFile(t) for t in ts)
        nodiths = list(nodithFile(t) for t in ts)
        diths = torch.ByteTensor(np.moveaxis(np.asarray(diths), 3, 1)) # N C H W
        nodiths = torch.ByteTensor(np.moveaxis(np.asarray(nodiths), 3, 1)) # N C H W
        return diths, nodiths

    def __len__(self):
        return len(self.videoList)

if __name__ == '__main__':
    tensor2im = lambda x: np.moveaxis(x.numpy(), 0, 2)
    if False:
        dataset = gif_moments_DithRec_train()
        dith, nodith = random.choice(dataset)
        Image.fromarray(np.concatenate(list(map(tensor2im, [dith, nodith])), axis=1)).show()
    if True:
        dataset = gif_moments_DithRec_eval()
        diths, nodiths = random.choice(dataset)
        dith, nodith = diths[0], nodiths[0]
        Image.fromarray(np.concatenate(list(map(tensor2im, [dith, nodith])), axis=1)).show()
