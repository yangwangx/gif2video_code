import torch
import torch.utils.data as DD

import imageio
from PIL import Image
import numpy as np
import random

__all__ = ['gif_faces_DithRec_train', 'gif_faces_DithRec_eval']

dataRoot = '/nfs/bigbrain/yangwang/Gif2Video/gif2video_data/gif_faces/'
trainSplit = dataRoot + 'split/face_train.txt'
validSplit = dataRoot + 'split/face_valid.txt'
dithRoot = dataRoot + 'face_gif_image/expand1.5_size256_s1_g32_dither/'
nodithRoot = dataRoot + 'face_gif_image/expand1.5_size256_s1_g32_nodither/'

class gif_faces_DithRec_train(DD.Dataset):
    def __init__(self, dithRoot=dithRoot, nodithRoot=nodithRoot, sCrop=256):
        super(gif_faces_DithRec_train, self).__init__()
        self.dithRoot = dithRoot
        self.nodithRoot = nodithRoot
        self.videoList = self.get_videoList(trainSplit)
        self.sCrop = sCrop

    def get_videoList(self, splitFile=trainSplit):
        videoList = []
        with open(splitFile, 'r') as f:
            for line in f.readlines():
                # line = f.readline()
                tmp = line.rstrip().split()
                VID, nFrm = tmp[0], int(tmp[1])
                videoList.append([VID, nFrm])
        return videoList

    def __getitem__(self, index):
        VID, nFrm = self.videoList[index]
        dithFile = lambda t: imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.dithRoot, VID, t))[:,:,:3]
        nodithFile = lambda t: imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.nodithRoot, VID, t))[:,:,:3]
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

class gif_faces_DithRec_eval(DD.Dataset):
    def __init__(self, dithRoot=dithRoot, nodithRoot=nodithRoot, tStride=10):
        super(gif_faces_DithRec_eval, self).__init__()
        self.dithRoot = dithRoot
        self.nodithRoot = nodithRoot
        self.videoList = self.get_videoList(validSplit)
        self.tStride = tStride

    def get_videoList(self, splitFile=validSplit):
        videoList = []
        with open(splitFile, 'r') as f:
            for line in f.readlines():
                # line = f.readline()
                tmp = line.rstrip().split()
                VID, nFrm = tmp[0], int(tmp[1])
                videoList.append([VID, nFrm])
        return videoList

    def __getitem__(self, index):
        VID, nFrm = self.videoList[index]
        dithFile = lambda t: imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.dithRoot, VID, t))[:,:,:3]
        nodithFile = lambda t: imageio.imread('{}/{}/frame_{:06d}.gif'.format(self.nodithRoot, VID, t))[:,:,:3]
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
    if True:
        dataset = gif_faces_DithRec_train()
        dith, nodith = random.choice(dataset)
        Image.fromarray(np.concatenate(list(map(tensor2im, [dith, nodith])), axis=1)).show()
    if True:
        dataset = gif_faces_DithRec_eval()
        diths, nodiths = random.choice(dataset)
        dith, nodith = diths[0], nodiths[0]
        Image.fromarray(np.concatenate(list(map(tensor2im, [dith, nodith])), axis=1)).show()
