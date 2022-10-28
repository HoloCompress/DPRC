import os 
import torch 
from compressai.utils.bench.codecs import JPEG
from skimage.io import imread 
from PIL import Image

args = {}
coder = JPEG(args)

encode_time_total = 0 
decode_time_total = 0
count = 0 

'''
out = {
"psnr": psnr_val,
"ms-ssim": msssim_val,
"bpp": bpp_val,
"encoding_time": enc_time,
"decoding_time": dec_time,
'''


chan_strs=['r', 'g', 'b']

data_dir = '/jixie/yujie_data/video_dataset_1080x1920/data/1920x1080/qp00_videos/'
eval_file = 'eval_video_list.txt'
eval_file_path = os.path.join(data_dir, eval_file)
all_files = []
with open(eval_file_path, 'r') as reader:
    for line in reader:
        line = line.strip()
        splits = line.split(',')
        vid = splits[0]
        start = int(splits[1])
        end = int(splits[2])
        for k in range(start, end):
            all_files.append(os.path.join(data_dir, f'{vid}-frame{k}.png'))

writer = open(f'jpeg_eval_video_data_RGB_split.txt', 'a')
for quality in [95, 90, 80, 70, 60]:
    writer.write(f'_____________________________quality{quality}_________________________________\n')
    all_time_enc = [[],[],[]]
    all_time_dec = [[],[],[]]
    all_bpp = [[],[],[]]
    for n, img_name in enumerate(all_files):
        img = Image.open(img_name)
        red, green, blue = img.split()
        for m, img in enumerate([red, green, blue]):
            jpeg_name = str.replace(img_name, '.png', f'-{chan_strs[m]}-quality{quality}.jpg')
            print(jpeg_name)
            img.save(jpeg_name,format='jpeg',  quality=quality)
            output = coder._run(img, quality=quality)
            count += 1
            all_time_enc[m].append(output['encoding_time'])
            all_time_dec[m].append(output['decoding_time'])
            all_bpp[m].append(output['bpp'])

    for m in range(3):
        avg_time_enc = sum(all_time_enc[m]) / len(all_time_enc[m])
        avg_time_dec = sum(all_time_dec[m]) / len(all_time_dec[m])
        avg_bpp = sum(all_bpp[m]) / len(all_bpp[m])
        writer.write(f'-----channel: {chan_strs[m]}-------\n')
        writer.write(f'quality: {quality}\n')
        writer.write(f'avg enc time is: {avg_time_enc:.4f}\n')
        writer.write(f'avg dec time is: {avg_time_dec:.4f}\n')
        writer.write(f'avg bpp is {avg_bpp:.4f}\n')
        writer.write(f'count is {count:d}\n')

writer.close()

writer.close()