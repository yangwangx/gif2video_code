CUDA_VISIBLE_DEVICES=1 python ../main.py --visMode --subset_eval test --tDown 4 \
--dither_mode nodither \
--initModel pretrained/ccdnet1_gan_moments_nodither_ep30.pt \
--visDir ccdnet1_gan_moments_nodither
