import os, sys, glob, time, progressbar, argparse, numpy as np, cv2, imageio, random
from PIL import Image
from itertools import islice
from functools import partial
from easydict import EasyDict as edict
from skimage.measure import compare_ssim as comp_ssim

import torch
import torch.nn as nn
import torch.nn.functional as FF
import torch.optim as optim
import torch.utils.data as DD
import torchvision
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
except NameError:
    cwd = ''
sys.path.append(cwd + '../')
import data_pth as datasets
import model_pth as models
from pytoolkit import *

diff_xy = lambda x: torch.cat((x[:, :, 1:, 1:] - x[:, :, 1:, :-1],
                               x[:, :, 1:, 1:] - x[:, :, :-1, 1:]), dim=1)
pad_tl = lambda x: FF.pad(x, pad=(1, 0, 1, 0))
L1 = lambda x, y: (x - y).abs().mean()
Lp = lambda p: lambda x, y: (x - y).abs().pow(p).mean()
f_idl = L1
f_smooth = lambda x: L1(diff_xy(x), 0)
f_gdl = lambda x, y: L1(diff_xy(x).abs(), diff_xy(y).abs())
preprocess = lambda x: x.float()/127.5 - 1
postprocess = lambda x: (x+1)*127.5

def compute_D_loss(d_logits_real, d_logits_fake, method='GAN'):
    assert method in ('GAN', 'LSGAN')
    d_labels_real = torch.ones(d_logits_real.shape).to(d_logits_real.device)
    d_labels_fake = torch.zeros(d_logits_fake.shape).to(d_logits_fake.device)
    if method == 'GAN':
        func = FF.binary_cross_entropy_with_logits
    elif method == 'LSGAN':
        func = FF.mse_loss
    d_loss_real = func(d_logits_real, d_labels_real)
    d_loss_fake = func(d_logits_fake, d_labels_fake)
    d_loss = (d_loss_real + d_loss_fake) / 2.0
    return d_loss, d_loss_real, d_loss_fake

def compute_D_acc(d_logits_real, d_logits_fake, method='GAN'):
    assert method in ('GAN', 'LSGAN')
    if method == 'GAN':
        thresh = 0.5
    elif method == 'LSGAN':
        thresh = 0.5
    d_acc_real = ( d_logits_real > thresh ).float().mean()
    d_acc_fake = ( d_logits_fake < thresh ).float().mean()
    d_acc = (d_acc_real + d_acc_fake) / 2.0
    return d_acc, d_acc_real, d_acc_fake

def compute_G_loss(d_logits_fake, method='GAN'):
    assert method in ('GAN', 'LSGAN')
    g_labels = torch.ones(d_logits_fake.shape).to(d_logits_fake.device)
    if method == 'GAN':
        func = FF.binary_cross_entropy_with_logits
    elif method == 'LSGAN':
        func = FF.mse_loss
    g_loss = func(d_logits_fake, g_labels)
    return g_loss

def rmse2psnr(rmse, maxVal=1.0):
    if rmse == 0:
        return 100
    else:
        return 20 * np.log10(maxVal/rmse)

def psnr2rmse(psnr, maxVal=1.0):
    if psnr == 100:
        return 0
    else:
        return maxVal / 10 ** (psnr/20.0)

def manual_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_options(opts, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opts).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    mkdir(opts.saveDir)
    file_name = os.path.join(opts.saveDir, 'opts_{}.txt'.format(time.asctime()))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
