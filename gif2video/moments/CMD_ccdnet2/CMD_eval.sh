# temporal downsample by 2
CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --subset_eval test --tCrop 3 --tStride 10 \
--dither_mode 'nodither' \
--color_model1_file 'pretrained/ccdnet1_nogan_faces_nodither_ep60.pt' \
--color_model2_file 'pretrained/ccdnet2_nogan_faces_nodither_ep50.pt' --unroll 1 \
--initModel 'pretrained/gif2video_t4.ccdnet1_gan_moments_nodither_ep30.pt'
# --initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt'

# temporal downsample by 4
CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --subset_eval test --tCrop 5 --tStride 10 \
---dither_mode 'nodither' \
--color_model1_file 'pretrained/ccdnet1_nogan_faces_nodither_ep60.pt' \
--color_model2_file 'pretrained/ccdnet2_nogan_faces_nodither_ep50.pt' --unroll 1 \
--initModel 'pretrained/gif2video_t4.ccdnet1_gan_moments_nodither_ep30.pt'
# --initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt'

# temporal downsample by 8
CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --subset_eval test --tCrop 9 --tStride 10 \
--dither_mode 'nodither' \
--color_model1_file 'pretrained/ccdnet1_nogan_faces_nodither_ep60.pt' \
--color_model2_file 'pretrained/ccdnet2_nogan_faces_nodither_ep50.pt' --unroll 1 \
--initModel 'pretrained/gif2video_t4.ccdnet1_gan_moments_nodither_ep30.pt'
# --initModel 'pretrained/gif2video_t8.ccdnet1_gan_faces_nodither_ep30.pt'
