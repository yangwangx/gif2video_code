CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode 'nodither' \
--base_model_file 'pretrained/ccdnet1_gan_faces_nodither_ep30.pt' \
--initModel 'pretrained/ccdnet2_gan_faces_nodither_ep30.pt' --unroll 1

CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode 'dither' \
--base_model_file 'pretrained/ccdnet1_gan_faces_dither_ep30.pt' \
--initModel 'pretrained/ccdnet2_gan_faces_dither_ep40.pt' --unroll 1
