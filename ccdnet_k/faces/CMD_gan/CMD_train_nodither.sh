CUDA_VISIBLE_DEVICES=0 python ../main.py \
--dither_mode 'nodither' \
--PtSz 256 --BtSz 8 --trRatio 0.03 --tDown 8 \
--w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
--w_dgan 1 --w_ggan 1 --gan_loss 'GAN' \
--LRPolicy 'constant' --LRStart 0.0002 --nEpoch 40 \
--base_model_file 'pretrained/ccdnet1_gan_faces_nodither_ep30.pt' \
--initModel '' \
--unroll 2 \
--saveDir 'results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,gan_d1_g1/unroll2/'  && \
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--dither_mode 'nodither' \
--PtSz 256 --BtSz 8 --trRatio 0.03 --tDown 8 \
--w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
--w_dgan 1 --w_ggan 1 --gan_loss 'GAN' \
--LRPolicy 'constant' --LRStart 0.0002 --nEpoch 40 \
--base_model_file 'pretrained/ccdnet1_gan_faces_nodither_ep30.pt' \
--initModel 'results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,gan_d1_g1/unroll2/ckpt/ep-0030.pt' \
--unroll 1 \
--saveDir 'results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,gan_d1_g1/unroll1/'
