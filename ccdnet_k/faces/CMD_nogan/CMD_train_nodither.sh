CUDA_VISIBLE_DEVICES=0 python ../main.py --nogan \
--dither_mode 'nodither' \
--PtSz 256 --BtSz 8 --trRatio 0.03 --tDown 8 \
--w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
--LRPolicy 'step' --LRStart 0.001 --LRStep 10 --gamma 0.5 --nEpoch 60 \
--base_model_file 'pretrained/ccdnet1_nogan_faces_nodither_ep60.pt' \
--initModel '' \
--unroll 2 \
--saveDir 'results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll2/' && \
CUDA_VISIBLE_DEVICES=0 python ../main.py --nogan \
--dither_mode 'nodither' \
--PtSz 256 --BtSz 8 --trRatio 0.03 --tDown 8 \
--w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
--LRPolicy 'step' --LRStart 0.001 --LRStep 10 --gamma 0.5 --nEpoch 60 \
--base_model_file 'pretrained/ccdnet1_nogan_faces_nodither_ep60.pt' \
--initModel 'results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll2/ckpt/ep-0060.pt' \
--unroll 1 \
--saveDir 'results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll1/'
