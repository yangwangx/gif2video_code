import os
import sys
import scipy.io as sio
from PIL import Image
import numpy as np
import random

import torch
import torch.utils.data as DD
import torchvision.transforms as TT

def get_cropChoice(HW, scale, HW_):
    H, W = HW
    H_, W_ = HW_
    cropChoice=[]
    for h in scale:
        for w in scale:
            cropChoice += [[      0,       h-1,       0,       w-1]]; # topleft
            cropChoice += [[      0,       h-1,     W-w,       W-1]]; # topright
            cropChoice += [[    H-h,       H-1,       0,       w-1]]; # botleft
            cropChoice += [[    H-h,       H-1,     W-w,       W-1]]; # botright
            cropChoice += [[H/2-h/2, H/2+h/2-1, W/2-w/2, W/2+w/2-1]]; # center
    return cropChoice

def get_Hollywood2_info(dataDir):
    L = sio.loadmat('%s/info/info.mat'%(dataDir))
    nTr = L['nTrain'][0,0]
    nTst = L['nTest'][0,0]
    trVds = ['train%05d'%(i) for i in range(1, nTr+1)]
    tstVds = ['test%05d'%(i) for i in range(1, nTst+1)]
    trLbs = L['TrainLabel']
    tstLbs = L['TestLabel']
    return trVds, trLbs, tstVds, tstLbs

class Hollywood2_rgb_train(DD.Dataset):
    def __init__(self, subset='train',
                 HW=[150, 200], scale=[150, 128, 112], HW_=[128, 128], T_=16):
        super(Hollywood2_rgb_train, self).__init__()

        # information about Hollywood2
        dataDir = '/nfs/bigeye/yangwang/DataSets/Hollywood2/'
        trVds, trLbs, tstVds, tstLbs = get_Hollywood2_info(dataDir)

        if subset == 'all':
            Vds = trVds + tstVds
            Lbs = np.concatenate((trLbs, tstLbs), 0)
        elif subset == 'train':
            Vds = trVds
            Lbs = trLbs
        elif subset == 'test':
            Vds = tstVds
            Lbs = tstLbs

        # crop choices for training images
        cropChoice = get_cropChoice(HW, scale, HW_)

        self.dataDir = dataDir
        self.Vds = Vds
        self.Lbs = Lbs
        self.HW = HW
        self.HW_ = HW_
        self.T_ = T_
        self.cropChoice = cropChoice
        self.toTensor = TT.ToTensor()

    def __getitem__(self, index):
        vid = self.Vds[index]

        frameDir = self.dataDir + '/frameflow/%s/'%(vid)
        maxFrm = len(os.listdir(frameDir))/3

        r_crop = random.choice(self.cropChoice)
        r_flip = random.random()>0.5
        H, W = self.HW
        H_, W_ = self.HW_
        T_ = self.T_

        frmInx = random.randrange(1, max(maxFrm-T_+1, 1)+1) + np.arange(T_)
        clip = torch.Tensor(3, T_, H_, W_)
        for i, t in enumerate(frmInx):
            im = Image.open('%s/i_%06d.jpg'%(frameDir, min(t, maxFrm)))
            # scale, crop, flip, rescale
            im = im.resize((W, H), Image.BILINEAR)
            im = im.crop((r_crop[2], r_crop[0], r_crop[3]+1, r_crop[1]+1)) # left-inclusive, [a, b)
            if r_flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.resize((W_, H_), Image.BILINEAR)
            im = self.toTensor(im)
            clip[:,i,:,:].copy_(im)
        clip = clip*255

        label = self.Lbs[index]
        cls = np.array([random.choice(np.nonzero(label>0)[0])]) # 0-based
        label = torch.ShortTensor(label).float() # shape: (12,), entry: 1 or -1
        cls = torch.LongTensor(cls) # shape: (1,)
        return clip, label, cls

    def __len__(self):
        return len(self.Vds)

class Hollywood2_rgb_feature(DD.Dataset):
    def __init__(self, subset, tLen=25, useCorner=False, useFlip=False,
                 HW=[150, 200], HW_=[128, 128], T_=16):
        super(Hollywood2_rgb_feature, self).__init__()

        # information about Hollywood2
        dataDir = '/nfs/bigeye/yangwang/DataSets/Hollywood2/'
        trVds, trLbs, tstVds, tstLbs = get_Hollywood2_info(dataDir)

        if subset == 'all':
            Vds = trVds + tstVds
            Lbs = np.concatenate((trLbs, tstLbs), 0)
        elif subset == 'train':
            Vds = trVds
            Lbs = trLbs
        elif subset == 'test':
            Vds = tstVds
            Lbs = tstLbs

        # crop choices for test images
        cropChoice=[]
        H, W = HW
        H_, W_ = HW_
        h, w = H_, W_
        cropChoice += [[int(H/2-h/2), int(H/2+h/2-1), int(W/2-w/2), int(W/2+w/2-1)]]; # center
        if useCorner:
            cropChoice += [[      0,       h-1,       0,       w-1]]; # topleft
            cropChoice += [[      0,       h-1,     W-w,       W-1]]; # topright
            cropChoice += [[    H-h,       H-1,       0,       w-1]]; # botleft
            cropChoice += [[    H-h,       H-1,     W-w,       W-1]]; # botright

        self.dataDir = dataDir
        self.Vds = Vds
        self.Lbs = Lbs
        self.HW = HW
        self.HW_ = HW_
        self.T_ = T_
        self.cropChoice = cropChoice
        self.tLen = tLen
        self.useFlip = useFlip
        self.toTensor = TT.ToTensor()

    def __getitem__(self, index):
        vid = self.Vds[index]

        frameDir = self.dataDir + '/frameflow/%s/'%(vid)
        maxFrm = int(len(os.listdir(frameDir))/3)

        H, W = self.HW
        H_, W_ = self.HW_
        T_ = self.T_

        # load video
        video = torch.Tensor(maxFrm, 3, H, W)
        for i, t in enumerate(range(1, maxFrm+1)):
            im = Image.open('%s/i_%06d.jpg'%(frameDir, t))
            im = im.resize((W, H), Image.BILINEAR)
            im = self.toTensor(im)
            video[i].copy_(im)
        video = video*255

        # sample clips: [flip x] frame x center[, corners]
        clips = []
        for i_start in np.linspace(0, maxFrm-T_, self.tLen, dtype='int'):
            for crp in self.cropChoice:
                clip_ = video[i_start:i_start+T_, :, crp[0]:crp[1]+1, crp[2]:crp[3]+1]
                clips.append(clip_.transpose(0, 1).unsqueeze(0))
        clips = torch.cat(clips, 0)

        if self.useFlip:
            clips_flip = clips.index_select(4, torch.LongTensor(range(W_-1, -1, -1)))
            clips = torch.cat([clips,clips_flip], 0)

        label = self.Lbs[index]
        cls = np.array([random.choice(np.nonzero(label>0)[0])]) # 0-based
        label = torch.ShortTensor(label).float() # shape: (12,), entry: 1 or -1
        cls = torch.LongTensor(cls) # shape: (1,)
        return clips, label, cls

    def __len__(self):
        return len(self.Vds)

if __name__ == '__main__':
    dataset = Hollywood2_rgb_train(subset='train')
    clip, label, cls = random.choice(dataset)
    TT.ToPILImage()(clip[:, 8, :, :]/255).show()
    dataset = Hollywood2_rgb_feature(subset='train')
    clip, label, cls = random.choice(dataset)
    TT.ToPILImage()(clip[0, :, 8, :, :]/255).show()
