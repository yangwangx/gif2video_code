rm -rf results/onebatch

CUDA_VISIBLE_DEVICES=0 python ../main.py --OneBatch --saveDir 'results/onebatch' --nEpoch 500
