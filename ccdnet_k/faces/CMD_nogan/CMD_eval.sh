CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode 'nodither' \
--base_model_file 'pretrained/ccdnet1_nogan_faces_nodither_ep60.pt' \
--initModel 'pretrained/ccdnet2_nogan_faces_nodither_ep50.pt' --unroll 1

## {'PSNR': 34.06, 'PSNR_gif': 30.99, 'SSIM': 0.929, 'SSIM_gif': 0.868}

CUDA_VISIBLE_DEVICES=0 python ../main.py --evalMode --tDown 1 \
--dither_mode 'dither' --no_regif \
--base_model_file 'pretrained/ccdnet1_nogan_faces_dither_ep60.pt' \
--initModel 'pretrained/ccdnet2_nogan_faces_dither_ep50.pt' --unroll 1

## {'PSNR': 35.40, 'PSNR_gif': 28.21, 'SSIM': 0.954, 'SSIM_gif': 0.746}

## [issue] 'pretrained/ccdnet2_nogan_faces_**dither**_ep50.pt' is trained 
##         with the base model wrongly initialized using weights from 
##         'pretrained/ccdnet1_nogan_faces_**nodither**_ep60.pt'
