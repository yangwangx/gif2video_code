import torch
import torch.nn as nn
import torch.nn.functional as FF

def double_conv(in_ch, out_ch, kernel_size=3, use_bn=True, leak=0.0):
    assert kernel_size%2==1, "kernel_size should be an odd number"
    padding = int(round((kernel_size - 1) / 2))
    if use_bn:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(leak, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.LeakyReLU(leak, inplace=True)
        )

def inconv(in_ch, out_ch, kernel_size=3):
    return double_conv(in_ch, out_ch, kernel_size)

def outconv(in_ch, out_ch, kernel_size=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size)

def downconv(in_ch, out_ch, kernel_size=3):
    return nn.Sequential(
        nn.AvgPool2d(2),
        double_conv(in_ch, out_ch, kernel_size=kernel_size)
    )

class upconv(nn.Module):
    def __init__(self, down_ch, up_ch, out_ch, bilinear=False):
        super(upconv, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(down_ch, down_ch, 2, stride=2)
        self.conv = double_conv(down_ch + up_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = FF.pad(x1, (diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet_simple(nn.Module):
    def __init__(self, in_ch, out_ch, ch=64):
        super(UNet_simple, self).__init__()
        self.inconv    =   inconv(in_ch, ch*1)
        self.downconv1 = downconv( ch*1, ch*2)
        self.downconv2 = downconv( ch*2, ch*4)
        self.downconv3 = downconv( ch*4, ch*8)
        self.downconv4 = downconv( ch*8, ch*8)

        self.upconv1 = upconv(ch*8, ch*8, ch*4)
        self.upconv2 = upconv(ch*4, ch*4, ch*2)
        self.upconv3 = upconv(ch*2, ch*2, ch*1)
        self.upconv4 = upconv(ch*1, ch*1, ch*1)
        self.outconv = outconv(ch*1, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)
        x = self.upconv1(x5, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.upconv4(x, x1)
        x = self.outconv(x)
        return x

if __name__ == '__main__':
    model = UNet_simple(in_ch=3, out_ch=3, ch=64)
