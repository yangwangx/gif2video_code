CUDA_VISIBLE_DEVICES=0 python ../main.py \
--dither_mode nodither \
--PtSz 256 --BtSz 8 --trRatio 1 --tDown 8 \
--w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
--w_dgan 1 --w_ggan 1 --gan_loss 'GAN' \
--LRPolicy 'constant' --LRStart 0.0002 --nEpoch 30 \
--initModel pretrained/ccdnet1_gan_faces_nodither_ep30.pt \
--saveDir 'results/g32_nodith/idl_gdl_gan/lr0.0002/' --checkpoint 0
