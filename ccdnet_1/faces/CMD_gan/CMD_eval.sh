CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode nodither \
--initModel pretrained/ccdnet1_gan_faces_nodither_ep20.pt

##  {'PSNR': 32.11, 'PSNR_gif': 30.99, 'SSIM': 0.902, 'SSIM_gif': 0.868}

CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode dither \
--initModel pretrained/ccdnet1_gan_faces_dither_ep30.pt

## {'PSNR': 33.68, 'PSNR_gif': 28.21, 'SSIM': 0.940, 'SSIM_gif': 0.746}
