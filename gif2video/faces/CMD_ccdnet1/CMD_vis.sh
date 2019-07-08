# temporal downsample by 8
CUDA_VISIBLE_DEVICES=0 python ../main.py --unroll 0 --visMode --initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt' --visDir vis_ep800_t8 --visNum 1 --tCrop 9

# temporal downsample by 4
CUDA_VISIBLE_DEVICES=0 python ../main.py --unroll 0 --visMode --initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt' --visDir vis_ep800_t4 --visNum 1 --tCrop 5

# temporal downsample by 2
CUDA_VISIBLE_DEVICES=0 python ../main.py --unroll 0 --visMode --initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt' --visDir vis_ep800_t2 --visNum 1 --tCrop 3
