CUDA_VISIBLE_DEVICES=1,2 python ../main.py --nogan \
--dither_mode nodither \
--PtSz 256 --BtSz 8 --trRatio 0.1 --tDown 8 \
--w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
--LRPolicy 'step' --LRStart 0.001 --LRStep 10 --gamma 0.5 --nEpoch 60 \
--initModel '' \
--saveDir 'results/g32_nodither_pt256_bt8_tr0.1/idl100_1,gdl100_1,nogan/' --checkpoint 0
