CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode nodither \
--initModel pretrained/ccdnet1_nogan_faces_nodither_ep60.pt

## {'PSNR': 32.83, 'PSNR_gif': 30.99, 'SSIM': 0.918, 'SSIM_gif': 0.868}

CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode dither \
--initModel pretrained/ccdnet1_nogan_faces_dither_ep60.pt

## {'PSNR': 33.93, 'PSNR_gif': 28.21, 'SSIM': 0.944, 'SSIM_gif': 0.746}
