import os, sys, glob, time, progressbar, argparse, numpy as np, cv2, imageio, random
from PIL import Image
from itertools import islice
from functools import partial
from easydict import EasyDict as edict
import scipy.io as sio
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
sys.path.append(cwd + '../../')
from pytoolkit import *

def get_base_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--workers', default=2, type=int, help='number of workers for dataloader')
    parser.add_argument('--BtSz', default=8, type=int, help='batch size')
    # parser.add_argument('--BtMerge', default=1, type=int, help='batch step for merged gradient update')
    parser.add_argument('--trRatio', default=1, type=float, help='ratio of training data used per epoch')
    parser.add_argument('--OneBatch', default=False, action='store_true', dest='OneBatch', help='debug with one batch')
    # model
    # loss
    # optimizer
    parser.add_argument('--noOptimizer', default=[], type=str, nargs='+', dest='noOptimizer')
    parser.add_argument('--solver', default='adam', choices=['adam','sgd'], help='which solver')
    parser.add_argument('--MM', default=0.9, type=float, help='momentum')
    parser.add_argument('--Beta', default=0.999, type=float, help='beta for adam')
    parser.add_argument('--WD', default=1e-4, type=float, help='weight decay')
    # learning rate
    parser.add_argument('--LRPolicy', default='constant', type=str, choices=['constant', 'step', 'steps', 'exponential',], help='learning rate policy')
    parser.add_argument('--gamma', default=0.1, type=float, help='decay rate for learning rate')
    parser.add_argument('--LRStart', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--LRStep', default=100, type=int, help='steps to change learning rate')
    parser.add_argument('--LRSteps', default=[], type=int, nargs='+', dest='LRSteps', help='epochs before cutting learning rate')
    parser.add_argument('--nEpoch', default=500, type=int, help='total epochs to run')
    # init & checkpoint
    parser.add_argument('--initModel', default='', help='init model in absence of checkpoint')
    parser.add_argument('--checkpoint', default=0, type=int, help='resume from checkpoint')
    # save & display
    parser.add_argument('--saveDir', default='results/default/', help='directory to save/log experiments')
    parser.add_argument('--saveStep', default=5, type=int, help='epoch step for snapshot')
    parser.add_argument('--evalStep', default=50, type=int, help='epoch step for evaluation')
    parser.add_argument('--dispIter', default=50, type=int, help='batch step for tensorboard')
    # other mode
    parser.add_argument('--evalMode', default=False, action='store_true', dest='evalMode', help='evaluation mode')
    parser.add_argument('--valMode', default=False, action='store_true', dest='valMode', help='validation mode')
    parser.add_argument('--visMode', default=False, action='store_true', dest='visMode', help='visualization mode')
    parser.add_argument('--visDir', default='visual', type=str, help="dir to store visualization")
    parser.add_argument('--visNum', default=10, type=int, help="number of videos to visualize")
    parser.add_argument('--applyMode', default=False, action='store_true', dest='applyMode', help='apply model to one gif')
    parser.add_argument('--applyFile', default='', type=str, help='path to gif')
    # misc
    parser.add_argument('--seed', default=1, type=int, help='random seed for torch/numpy')
    return parser

kldiv_w = lambda p, m, w: (FF.kl_div((p+1e-8).log(), m, reduction='none') * w).mean() * m.size(1)
kldiv = lambda p, m: FF.kl_div((p+1e-8).log(), m) * m.size(1)
l2norm = lambda x, dim: x / (1e-8 + x.norm(dim=dim, keepdim=True))
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

def manual_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_plot_img(npArray):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.plot(npArray)
    canvas.draw()       # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return img

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
    return message

def create_optimizer(model, opts):
    optimizer = edict()
    if opts.solver == 'sgd':
        solver = partial(torch.optim.SGD, lr=opts.LRStart, momentum=opts.MM, weight_decay=opts.WD)
    elif opts.solver == 'adam':
        solver = partial(torch.optim.Adam, lr=opts.LRStart, betas=(opts.MM, opts.Beta), weight_decay=opts.WD)
    else:
        raise ValueError('optim solver "{}" not defined'.format(opts.solver))
    for key in model.keys():
        params = list(model[key].parameters())
        if params and key not in opts.noOptimizer: optimizer[key] = solver(params)
    return optimizer

def create_scheduler(optimizer, opts):
    scheduler = edict()
    for key in optimizer.keys():
        op = optimizer[key]
        if opts.LRPolicy == 'constant':
            scheduler[key] = optim.lr_scheduler.ExponentialLR(op, gamma=1.0)
        elif opts.LRPolicy == 'step':
            scheduler[key] = optim.lr_scheduler.StepLR(op, opts.LRStep, gamma=opts.gamma)
        elif opts.LRPolicy == 'steps':
            scheduler[key] = optim.lr_scheduler.MultiStepLR(op, opts.LRSteps, gamma=opts.gamma)
        elif opts.LRPolicy == 'exponential':
            scheduler[key] = optim.lr_scheduler.ExponentialLR(op, gamma=opts.gamma)
        else:
            raise ValueError('learning rate policy "{}" not defined'.format(opts.LRPolicy))
    return scheduler

def mkdir_save(state, state_file):
    mkdir(os.path.dirname(state_file))
    torch.save(state, state_file)

def save_checkpoint(epoch, model, optimizer, opts):
    print('save model & optimizer @ epoch %d'%(epoch))
    ckpt_file = '%s/ckpt/ep-%04d.pt'%(opts.saveDir, epoch)
    state = {}
    # an error occurs when I use edict
    # possible reason: optim.state_dict() has an 'state' attribute
    state['epoch'] = epoch
    for key in model.keys():
        _m = model[key]
        if hasattr(_m, 'module'): _m = _m.module
        state['model_'+key] = _m.state_dict()
    for key in optimizer.keys():
        state['optimizer_'+key] = optimizer[key].state_dict()
    mkdir_save(state, ckpt_file)

def resume_checkpoint(epoch, model, optimizer, opts):
    print('resume model & optimizer from epoch %d'%(epoch))
    ckpt_file = '%s/ckpt/ep-%04d.pt'%(opts.saveDir, epoch)
    if os.path.isfile(ckpt_file):
        L = torch.load(ckpt_file)
        for key in model.keys():
            _m = model[key]
            if hasattr(_m, 'module'): _m = _m.module
            if 'model_'+key in L: _m.load_state_dict(L['model_'+key])
        for key in optimizer.keys():
            if 'optimizer_'+key in L:
                optimizer[key].load_state_dict(L['optimizer_'+key])
    else:
        print('checkpoint "%s" not found'%(ckpt_file))
        quit()

def initialize(model, initModel):
    if initModel == '':
        print('no further initialization')
        return
    elif os.path.isfile(initModel):
        L = torch.load(initModel)
        for key in model.keys():
            _m = model[key]
            if hasattr(_m, 'module'): _m = _m.module
            if 'model_'+key in L: _m.load_state_dict(L['model_'+key], strict=False)
        print('model initialized using [%s]'%(initModel))
    else:
        print('[%s] not found'%(initModel))
        quit()

