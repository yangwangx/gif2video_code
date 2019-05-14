CUDA_VISIBLE_DEVICES=0 python ../main.py --visMode \
--dither_mode nodither \
--initModel pretrained/ccdnet1_gan_faces_nodither_ep30.pt \
--visDir ccdnet1_gan_faces_nodither

CUDA_VISIBLE_DEVICES=0 python ../main.py --visMode \
--dither_mode dither \
--initModel pretrained/ccdnet1_gan_faces_dither_ep30.pt \
--visDir ccdnet1_gan_faces_dither
