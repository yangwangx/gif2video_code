CUDA_VISIBLE_DEVICES=1 python ../main.py --evalMode --subset_eval test --tDown 10 \
--dither_mode nodither \
--initModel pretrained/ccdnet1_nogan_faces_nodither_ep60.pt
