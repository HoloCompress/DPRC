import os 
import torch 
from compressai.utils.bench.codecs import JPEG
from skimage.io import imread 
from PIL import Image, ImageOps 
import numpy as np 

args = {}
coder = JPEG(args)

channel_strs = ['red', 'green', 'blue']
writer = open('time_for_jpeg_compress_div2k_imgs_RGB_split-new.txt', 'a')
for quality in range(85, 80, -5):
    writer.write(f'_____________________________quality{quality}_________________________________\n')
    all_time_enc = [[],[],[]]
    all_time_dec = [[],[],[]]
    all_bpp = [[],[],[]]
    count = 0 
    for i in range(801, 901):
        img_name = os.path.join(f'/jixie/yujie_data/DIV2K_valid_HR/{i:04d}-gt-target.png')
        img = Image.open(img_name)
        red, green, blue = img.split()
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

