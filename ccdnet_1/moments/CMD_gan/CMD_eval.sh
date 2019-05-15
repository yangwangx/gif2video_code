CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --subset_eval test --tDown 10 \
--dither_mode nodither \
--initModel pretrained/ccdnet1_gan_faces_nodither_ep20.pt


CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --subset_eval test --tDown 10 \
--dither_mode nodither \
--initModel pretrained/ccdnet1_gan_moments_nodither_ep30.pt
