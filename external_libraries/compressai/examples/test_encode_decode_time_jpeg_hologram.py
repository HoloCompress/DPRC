import os 
import torch 
from compressai.utils.bench.codecs import JPEG
from skimage.io import imread 
from PIL import Image, ImageOps 
import numpy as np 

args = {}
coder = JPEG(args)

channel_strs = ['red', 'green', 'blue']
writer = open('time_for_jpeg_compress_hologram_imgs_new.txt', 'a')
for quality in range(100, 85, -5):
    writer.write(f'_____________________________quality{quality}_________________________________\n')
    all_time_enc = [[],[],[]]
    all_time_dec = [[],[],[]]
    all_bpp = [[],[],[]]
    count = 0 
    for i in range(801, 901):
        img_name = os.path.join(f'/jixie/yujie_data/DIV2K_valid_HR/{i:04d}-gt-target.png')
        blue_name  = f'/jixie/yujie_codes/holonet_compression/HoloNet_yujie_BLUE_5firstVGG_compression_multiScale_MSE_WFFT1e-8_VGG_2x_msssim_again/train_log/results/test-ckpt-17-rescale/{i:04d}-pred-phase.png'
        img = Image.open(img_name)
        red_name = f'/jixie/yujie_codes/holonet_compression/HoloNet_yujie_red_5firstVGG_compression_multiScale_MSE_WFFT1e-8_VGG_2x_msssim_again/train_log/results/test-ckpt-67-rescale/{i:04d}-pred-phase.png'
        green_name = f'/jixie/yujie_codes/holonet_compression/HoloNet_yujie_GREEN_5firstVGG_compression_multiScale_MSE_WFFT1e-8_VGG_2x_msssim/train_log/results/test-ckpt-30-rescale/{i:04d}-pred-phase.png'
        red = Image.open(red_name)
        green = Image.open(green_name)
        blue = Image.open(blue_name)
        blue_np = np.array(blue)
        print(blue_np.shape)
        #red, green, blue = img.split()
        for m, img in enumerate([red, green, blue]):
            output = coder._run(img, quality=quality)
            count += 1
            img_np = np.array(img)
            #print(img_np.shape)
            all_time_enc[m].append(output['encoding_time'])
            all_time_dec[m].append(output['decoding_time'])
            all_bpp[m].append(output['bpp'])
        
    for m in range(3):
        avg_time_enc = sum(all_time_enc[m]) / len(all_time_enc[m])
        avg_time_dec = sum(all_time_dec[m]) / len(all_time_dec[m])
        avg_bpp = sum(all_bpp[m]) / len(all_bpp[m])
        writer.write(f'-----channel: {channel_strs[m]}-------\n')
        writer.write(f'quality: {quality}\n')
        writer.write(f'avg enc time is: {avg_time_enc:.4f}\n')
        writer.write(f'avg dec time is: {avg_time_dec:.4f}\n')
        writer.write(f'avg bpp is {avg_bpp:.4f}\n')
        writer.write(f'count is {count:d}\n')

writer.close()

