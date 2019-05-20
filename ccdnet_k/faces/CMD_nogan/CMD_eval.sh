CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode 'nodither' \
--base_model_file 'pretrained/ccdnet1_nogan_faces_nodither_ep60.pt' \
--initModel 'pretrained/ccdnet2_nogan_faces_nodither_ep50.pt' --unroll 1

CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode 'dither' --no_regif \
--base_model_file 'pretrained/ccdnet1_nogan_faces_dither_ep60.pt' \
--initModel 'pretrained/ccdnet2_nogan_faces_dither_ep50.pt' --unroll 1
