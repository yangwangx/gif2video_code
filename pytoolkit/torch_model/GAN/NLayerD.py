import torch
import torch.nn as nn

""" Defines the PatchGAN discriminator with the specified arguments. """

def NLayerDiscriminator(in_ch, ndf=64, n_layers=3):
    kw = 4
    padw = 1
    sequence = [
        nn.Conv2d(in_ch, ndf, kernel_size=kw, stride=2, padding=padw),
        nn.LeakyReLU(0.2, True)
    ]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
        nf_mult_prev = nf_mult
        nf_mult = min(2**n, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers, 8)
    sequence += [
        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                  kernel_size=kw, stride=1, padding=padw, bias=False),
        nn.BatchNorm2d(ndf * nf_mult),
        nn.LeakyReLU(0.2, True)
    ]

    sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

    return nn.Sequential(*sequence)

if __name__ == '__main__':
    model = NLayerDiscriminator(in_ch=6)
