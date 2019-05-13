import torch
import torch.utils.data as DD

import imageio
from PIL import Image
import numpy as np
import random

dataRoot = '/nfs/bigbrain/yangwang/Gif2Video/gif2video_data/gif_faces/'
trainSplit = dataRoot + 'split/face_train.txt'
validSplit = dataRoot + 'split/face_valid.txt'
targetRoot = dataRoot + 'face/expand1.5_size256/'
inputRoot = dataRoot + 'face_gif_image/expand1.5_size256_s1_g32_nodither/'
# inputRoot = dataRoot + 'face_gif_image/expand1.5_size256_s1_g32_dither/'

def getPaletteInRgb(gifFile):
    gif = Image.open(gifFile)
    assert gif.mode == 'P', "image should be palette mode"
    nColor = len(gif.getcolors())
    pal = gif.getpalette()
    colors = list(list(pal[i:i+3]) for i in range(0, len(pal), 3))
    return nColor, colors

class gif_faces_train(DD.Dataset):
    def __init__(self, inputRoot=inputRoot, targetRoot=targetRoot, patchSize=256):
        super(gif_faces_train, self).__init__()
        self.inputRoot = inputRoot
        self.targetRoot = targetRoot
        self.imageList = self.get_imageList(trainSplit)
        self.patchSize = patchSize

    def get_imageList(self, splitFile=trainSplit):
        imageList = []
        with open(splitFile, 'r') as f:
            for line in f.readlines():
                # line = f.readline()
                tmp = line.rstrip().split()
                VID, nFrm = tmp[0], int(tmp[1])
                for i in range(1, nFrm+1):
                    imageList.append('{}/frame_{:06d}'.format(VID, i))
        return imageList

    def __getitem__(self, index):
        imID = self.imageList[index]
        inputFile = '{}/{}.gif'.format(self.inputRoot, imID)
        targetFile = '{}/{}.jpg'.format(self.targetRoot, imID)
        # get input gif and target image
        input = imageio.imread(inputFile)[:, :, :3]
        target = imageio.imread(targetFile)
        # get random patch
        H, W, _ = input.shape
        y1 = random.randrange(H - self.patchSize + 1)
        x1 = random.randrange(W - self.patchSize + 1)
        input = input[y1:y1+self.patchSize, x1:x1+self.patchSize, :]
        target = target[y1:y1+self.patchSize, x1:x1+self.patchSize, :]
        # get color palette
        nColor, colors = getPaletteInRgb(inputFile)
        # numpy to tensor
        input = torch.ByteTensor(np.moveaxis(input, 2, 0))
        target = torch.ByteTensor(np.moveaxis(target, 2, 0))
        colors = torch.ByteTensor(colors)
        return input, target, colors, nColor

    def __len__(self):
        return len(self.imageList)

class gif_faces_test(DD.Dataset):
    def __init__(self, inputRoot=inputRoot, targetRoot=targetRoot, tDown=4):
        super(gif_faces_test, self).__init__()
        self.inputRoot = inputRoot
        self.targetRoot = targetRoot
        self.tDown = tDown
        self.videoList, self.frameCount = self.get_videoList(validSplit)

    def get_videoList(self, splitFile=validSplit):
        videoList = []
        frameCount = []
        with open(splitFile, 'r') as f:
            for line in f.readlines():
                # line = f.readline()
                tmp = line.rstrip().split()
                VID, nFrm = tmp[0], int(tmp[1])
                videoList.append(VID)
                frameCount.append(nFrm)
        return videoList, frameCount

    def __getitem__(self, index):
        VID = self.videoList[index]
        nFrm = self.frameCount[index]
        # get video gif frames and target frames
        input, target = [], []
        for i in range(1, nFrm+1, self.tDown):
            inputFile = '{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, i)
            targetFile = '{}/{}/frame_{:06d}.jpg'.format(self.targetRoot, VID, i)
            input.append(imageio.imread(inputFile)[:, :, 0:3])
            target.append(imageio.imread(targetFile))

        # get video gif color palettes
        nColor, colors = [], []
        for i in range(1, nFrm+1, self.tDown):
            inputFile = '{}/{}/frame_{:06d}.gif'.format(self.inputRoot, VID, i)
            _nColor, _colors = getPaletteInRgb(inputFile)
            nColor.append(_nColor)
            colors.append(_colors)

        # numpy to tensor
        im2tensor = lambda x: torch.ByteTensor(np.moveaxis(np.asarray(x), 3, 1))
        input, target = im2tensor(input), im2tensor(target) # T [RGB] H W
        nColor = torch.LongTensor(nColor)
        colors = torch.ByteTensor(colors)
        return input, target, colors, nColor

    def __len__(self):
        return len(self.videoList)

if __name__ == '__main__':
    tensor2im = lambda x: np.moveaxis(x.numpy(), 0, 2)
    if True:
        dataset = gif_faces_train(patchSize=256)
        print('training set is of {} frames'.format(len(dataset)))
        input, target, colors, nColor = random.choice(dataset)
        input, target = tensor2im(input), tensor2im(target)
        Image.fromarray(np.concatenate([input, target], axis=1)).show()
        print('the color palette size is {}'.format(nColor))
    if True:
        dataset = gif_faces_test(tDown=4)
        print('test set is of {} videos'.format(len(dataset)))
        input, target, colors, nColor = random.choice(dataset)
        input, target = tensor2im(input[0]), tensor2im(target[0])
        Image.fromarray(np.concatenate([input, target], axis=1)).show()
        print('the color palette size is {}'.format(nColor[0]))
