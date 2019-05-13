import torch
import torch.nn as nn
import torch.nn.functional as FF

class DoubleAttention2D(nn.Module):
    def __init__(self, in_channels, reduce_channels, att_channels):
        super(DoubleAttention2D, self).__init__()

        self.reduce = nn.Conv2d(in_channels, reduce_channels, kernel_size=1)
        self.extend = nn.Conv2d(reduce_channels, in_channels, kernel_size=1)
        self.attend1 = nn.Conv2d(in_channels, att_channels, kernel_size=1)
        self.attend2 = nn.Conv2d(in_channels, att_channels, kernel_size=1)

    def forward(self, x):
        _B, _C, _H, _W = x.shape
        A = self.reduce(x).view(_B, -1, _H*_W)
        B = torch.softmax(self.attend1(x).view(_B, -1, _H*_W), dim=-1).transpose(2, 1).contiguous()
        V = torch.softmax(self.attend2(x).view(_B, -1, _H*_W), dim=1)
        G = torch.bmm(A, B)
        Z = torch.bmm(G, V).view(_B, -1, _H, _W)
        Z = self.extend(Z)
        return x + Z

class DoubleAttention3D(nn.Module):
    def __init__(self, in_channels, reduce_channels, att_channels):
        super(DoubleAttention3D, self).__init__()

        self.reduce = nn.Conv3d(in_channels, reduce_channels, kernel_size=1)
        self.extend = nn.Conv3d(reduce_channels, in_channels, kernel_size=1)
        self.attend1 = nn.Conv3d(in_channels, att_channels, kernel_size=1)
        self.attend2 = nn.Conv3d(in_channels, att_channels, kernel_size=1)

    def forward(self, x):
        _B, _C, _T, _H, _W = x.shape
        A = self.reduce(x).view(_B, -1, _T*_H*_W)
        B = torch.softmax(self.attend1(x).view(_B, -1, _T*_H*_W), dim=-1).transpose(2, 1).contiguous()
        V = torch.softmax(self.attend2(x).view(_B, -1, _T*_H*_W), dim=1)
        G = torch.bmm(A, B)
        Z = torch.bmm(G, V).view(_B, -1, _T, _H, _W)
        Z = self.extend(Z)
        return x + Z

if __name__ == '__main__':
    if True:
        x = torch.rand(5, 20, 5, 5)
        m = DoubleAttention2D(20, 5, 5)
        y = m.cuda()(x.cuda())
        print(y.shape)
    if True:
        x = torch.rand(5, 20, 3, 5, 5)
        m = DoubleAttention3D(20, 5, 5)
        y = m.cuda()(x.cuda())
        print(y.shape)
