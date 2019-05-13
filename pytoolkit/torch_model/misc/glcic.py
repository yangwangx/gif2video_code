import torch
import torch.nn as nn

def conv2d_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, leak=0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes, affine=True),
        nn.LeakyReLU(leak, inplace=True)
        )

def deconv2d_bn(in_planes, out_planes, kernel_size=4, stride=2, padding=1, leak=0):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes, affine=True),
        nn.LeakyReLU(leak, inplace=True)
        )

def glcic(in_ch=3, out_ch=3, ch=64, leak=0):
    return nn.Sequential(
        conv2d_bn( in_ch,  ch*1, kernel_size=5, stride=1, padding=2, leak=leak),
        conv2d_bn(  ch*1,  ch*2, kernel_size=3, stride=2, padding=1, leak=leak),
        conv2d_bn(  ch*2,  ch*2, kernel_size=3, stride=1, padding=1, leak=leak),
        conv2d_bn(  ch*2,  ch*4, kernel_size=3, stride=2, padding=1, leak=leak),
        conv2d_bn(  ch*4,  ch*4, kernel_size=3, stride=1, padding=1, leak=leak),
        conv2d_bn(  ch*4,  ch*4, kernel_size=3, stride=1, padding=1, leak=leak),
        conv2d_bn(  ch*4,  ch*4, kernel_size=3, stride=1, padding=2,  dilation=2, leak=leak),
        conv2d_bn(  ch*4,  ch*4, kernel_size=3, stride=1, padding=4,  dilation=4, leak=leak),
        conv2d_bn(  ch*4,  ch*4, kernel_size=3, stride=1, padding=8,  dilation=8, leak=leak),
        conv2d_bn(  ch*4,  ch*4, kernel_size=3, stride=1, padding=16, dilation=16, leak=leak),
        conv2d_bn(  ch*4,  ch*4, kernel_size=3, stride=1, padding=1, leak=leak),
        conv2d_bn(  ch*4,  ch*4, kernel_size=3, stride=1, padding=1, leak=leak),
        deconv2d_bn(ch*4,  ch*2, kernel_size=4, stride=2, padding=1, leak=leak),
        conv2d_bn(  ch*2,  ch*2, kernel_size=3, stride=1, padding=1, leak=leak),
        deconv2d_bn(ch*2,  ch*1, kernel_size=4, stride=2, padding=1, leak=leak),
        conv2d_bn(  ch*1,    32, kernel_size=3, stride=1, padding=1, leak=leak),
        nn.Conv2d(   32, out_ch, kernel_size=3, stride=1, padding=1),
        )

if __name__ == '__main__':
    model = glcic(in_ch=3, out_ch=3, ch=64, leak=0)
