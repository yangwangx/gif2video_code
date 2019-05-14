import numpy as np, random
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as FF

class ImagePool():
    """Image buffer for training generative models
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        # no pool
        if self.pool_size == 0:
            return images

        # with pool
        ret_images = []
        for image in images:
            image = torch.unsqueeze(image.detach(), dim=0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                ret_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randrange(0, self.pool_size)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    ret_images.append(tmp)
                else:
                    ret_images.append(image)
        ret_images = torch.cat(ret_images, 0)
        return ret_images

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def AverageMeters(keys):
    """Create a dictionary of AverageMeters"""
    AMs = edict()
    for key in keys:
        AMs[key] = AverageMeter()
    return AMs

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    Args: 
        output:  the predicted class-wise scores, torch tensor of shape (B, C)
        target:  the ground-truth class labels, torch tensor of shape (B,)
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def compute_D_loss(d_logits_real, d_logits_fake, method='GAN'):
    """Compute the discriminator loss.
    Args:
        d_logits_real: d's output logits for real data, torch tensor of shape (B, 1, ...)
        d_logits_fake: d's output logits for fake data, torch tensor of shape (B, 1, ...)
    """
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
    """Compute the discriminator accuracy.
    Args:
        d_logits_real: d's output logits for real data, torch tensor of shape (B, 1, ...)
        d_logits_fake: d's output logits for fake data, torch tensor of shape (B, 1, ...)
    """
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
    """Compute the generator loss.
    Args:
        d_logits_fake: d's output logits for `generated` data, torch tensor of shape (B, 1, ...)
    """
    assert method in ('GAN', 'LSGAN')
    g_labels = torch.ones(d_logits_fake.shape).to(d_logits_fake.device)
    if method == 'GAN':
        func = FF.binary_cross_entropy_with_logits
    elif method == 'LSGAN':
        func = FF.mse_loss
    g_loss = func(d_logits_fake, g_labels)
    return g_loss

def pairwise_distances(x, y=None):
    """ Computes the pairwise euclidean distances between rows of x and rows of y.
    Args:
        x: torch tensor of shape (m, d)
        y: torch tensor of shape (n, d), or None
    Returns:
        dist: torch tensor of shape (m, n), or (m, m)
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist
