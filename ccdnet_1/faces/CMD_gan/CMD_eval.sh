CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode nodither \
--initModel pretrained/ccdnet1_gan_faces_nodither_ep30.pt


CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode dither \
--initModel pretrained/ccdnet1_gan_faces_dither_ep30.pt
