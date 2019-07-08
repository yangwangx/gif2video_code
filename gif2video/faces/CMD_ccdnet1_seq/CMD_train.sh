# scheme1
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --sequential --unroll 0 \
--dither_mode 'nodither' \
--tCrop 9 --sCrop 256 --tStride 10 --BtSz 8 --trRatio 1 \
--gan_loss 'GAN'  --w_dgan 1 --w_ggan 1 --w_idl 0.5 --w_gdl 0.5 --w_warp 0.5 --w_smooth 1 \
--LRPolicy 'steps' --LRStart 0.0001 --gamma 0.1 --LRSteps 200 400 --nEpoch 500 \
--initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt' \
--saveDir 'results/s1_g32_nodith_t8/scheme1/' \
--L_warp_outlier 25
