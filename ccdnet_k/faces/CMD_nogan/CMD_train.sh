# CUDA_VISIBLE_DEVICES=2 python ../main.py --nogan \
# --inputRoot '/mnt/disk/data/yangwang/Gif2Video/data,FF/face_gif_image/expand1.5_size256_s1_g32_nodither/' \
# --PtSz 256 --BtSz 8 --trRatio 0.03 --tDown 8 \
# --w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
# --LRPolicy 'step' --LRStart 0.001 --LRStep 10 --gamma 0.5 --nEpoch 60 \
# --initModel '' \
# --unroll 2 \
# --saveDir 'results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll2/' --checkpoint 10 \

# CUDA_VISIBLE_DEVICES=2 python ../main.py --nogan \
# --inputRoot '/mnt/disk/data/yangwang/Gif2Video/data,FF/face_gif_image/expand1.5_size256_s1_g32_nodither/' \
# --PtSz 256 --BtSz 8 --trRatio 0.03 --tDown 8 \
# --w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
# --LRPolicy 'step' --LRStart 0.001 --LRStep 10 --gamma 0.5 --nEpoch 60 \
# --initModel 'results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll2/ckpt/ep-0060.pt' \
# --unroll 1 \
# --saveDir 'results/g32_nodither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll1/'

CUDA_VISIBLE_DEVICES=1,2 python ../main_dither.py --nogan \
--inputRoot '/mnt/disk/data/yangwang/Gif2Video/data,FF/face_gif_image/expand1.5_size256_s1_g32_dither/' \
--PtSz 256 --BtSz 8 --trRatio 0.03 --tDown 8 \
--w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
--LRPolicy 'step' --LRStart 0.001 --LRStep 10 --gamma 0.5 --nEpoch 60 \
--initModel '' \
--unroll 2 \
--saveDir 'results/g32_dither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll2/' && \
CUDA_VISIBLE_DEVICES=1 python ../main_dither.py --nogan \
--inputRoot '/mnt/disk/data/yangwang/Gif2Video/data,FF/face_gif_image/expand1.5_size256_s1_g32_dither/' \
--PtSz 256 --BtSz 8 --trRatio 0.03 --tDown 8 \
--w_idl 100 --w_gdl 100 --p_idl 1 --p_gdl 1 \
--LRPolicy 'step' --LRStart 0.001 --LRStep 10 --gamma 0.5 --nEpoch 60 \
--initModel 'results/g32_dither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll2/ckpt/ep-0060.pt' \
--unroll 1 \
--saveDir 'results/g32_dither_pt256_bt8_tr0.03/idl100_1,gdl100_1,nogan/unroll1/'


