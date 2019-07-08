# temporal downsample by 8
CUDA_VISIBLE_DEVICES=0 python ../main.py --unroll 0 --evalMode \
--dither_mode 'nodither' \
--initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt' \
--tCrop 9 --tStride 9

# temporal downsample by 4
CUDA_VISIBLE_DEVICES=0 python ../main.py --unroll 0 --evalMode \
--dither_mode 'nodither' \
--initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt' \
--tCrop 5 --tStride 5

# temporal downsample by 2
CUDA_VISIBLE_DEVICES=0 python ../main.py --unroll 0 --evalMode \
--dither_mode 'nodither' \
--initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt' \
--tCrop 3 --tStride 3
