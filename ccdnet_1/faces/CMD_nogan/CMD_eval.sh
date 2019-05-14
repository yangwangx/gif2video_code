CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode nodither \
--initModel pretrained/ccdnet1_nogan_faces_nodither_ep60.pt


CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode dither \
--initModel pretrained/ccdnet1_nogan_faces_dither_ep60.pt
