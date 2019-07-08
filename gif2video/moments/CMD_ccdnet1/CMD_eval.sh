# temporal downsample by 2
CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode  --subset_eval test \
--dither_mode 'nodither' \
--initModel 'pretrained/gif2video_t4.ccdnet1_gan_moments_nodither_ep30.pt' \
--tCrop 3 --tStride 10

# temporal downsample by 4
CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode  --subset_eval test \
--dither_mode 'nodither' \
--initModel 'pretrained/gif2video_t4.ccdnet1_gan_moments_nodither_ep30.pt' \
--tCrop 5 --tStride 10

# temporal downsample by 8 
CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode  --subset_eval test \
--dither_mode 'nodither' \
--initModel 'pretrained/gif2video_t4.ccdnet1_gan_moments_nodither_ep30.pt' \
--tCrop 9 --tStride 10
