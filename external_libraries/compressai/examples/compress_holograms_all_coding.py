#!/usr/bin/env python
# coding: utf-8

# <!-- Copyright 2020 InterDigital Communications, Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# # CompressAI inference demo
import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from pytorch_msssim import ms_ssim
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018,cheng2020_anchor
from ipywidgets import interact, widgets

from functools import partial
from matplotlib import pyplot as plt
import os
import cv2

# ## Load a pretrained model

'''
functions
'''
def pillow_encode(img, fmt='jpeg', quality=10):
    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp)
    return rec, bpp

def find_closest_bpp(target, img, fmt='jpeg'):
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp

def find_closest_psnr(target, img, fmt='jpeg'):
    lower = 0
    upper = 100
    prev_mid = upper
    
    def _psnr(a, b):
        a = np.asarray(a).astype(np.float32)
        b = np.asarray(b).astype(np.float32)
        mse = np.mean(np.square(a - b))
        return 20*math.log10(255.) -10. * math.log10(mse)
    
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        prev_mid = mid
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        psnr_val = _psnr(rec, img)
        if psnr_val > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp, psnr_val

def find_closest_msssim(target, img, fmt='jpeg'):
    lower = 0
    upper = 100
    prev_mid = upper
    
    def _mssim(a, b):
        a = torch.from_numpy(np.asarray(a).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        b = torch.from_numpy(np.asarray(b).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        return ms_ssim(a, b, data_range=255.).item()

    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        prev_mid = mid
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        msssim_val = _mssim(rec, img)
        if msssim_val > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp, msssim_val


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

# ### Per-latent bit-rate result
def detailed_bpp(out):
    size = out['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    for name, values in out_net['likelihoods'].items():
        bpp_val = torch.log(values).sum() / (-math.log(2) * num_pixels)
        print(f'- "{name}" latent bit-rate: {bpp_val:.3f} bpp')
########################################################################################################


root_dir = '/jixie/yujie_codes/holonet_compression/our_final_results_0821/all_phase_data/'
rgb_holo_path = os.path.join(root_dir, 'ours_step1_rgb_PRN')
if not os.path.isdir(rgb_holo_path):
    os.mkdir(rgb_holo_path)

ConstantPad = torch.nn.ConstantPad2d(padding=(0, 0, 4, 4), value=0)
#model_dict={'hyperprior18':bmshj2018_hyperprior, 'chen_anchor20': cheng2020_anchor, 'jointAuto18':mbt2018}
model_dict={'jointAuto18':mbt2018, 'hyperprior18':bmshj2018_hyperprior}

for key, value in model_dict.items():
    result_sub_dir = f'rgb_step1_{key}_again'
    result_dir = os.path.join(root_dir, result_sub_dir)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    
    device = 'cuda' 

    
    if key != 'chen2020-anchor':
        quality_list = list(range(1, 9))
    else: 
        quality_list = list(range(1, 7))
    

    for quality in quality_list:

        writer = open(os.path.join(result_dir, f'quantitatives-quality{quality}.txt'), 'a')
        writer.write('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        psnr_list = [] 
        ssim_list = []
        bpp_list = []
        count = []
        
        net = value(quality=quality, pretrained=True).eval().to(device)
        print(f'Parameters: {sum(p.numel() for p in net.parameters())}')
        print(f'Entropy bottleneck(s) parameters: {sum(p.numel() for p in net.aux_parameters())}')
        writer.write(f'Parameters: {sum(p.numel() for p in net.parameters())}\n')
        writer.write(f'Entropy bottleneck(s) parameters: {sum(p.numel() for p in net.aux_parameters())}\n')


        # ## 1. Inference

        # ### Load image and convert to 4D float tensor

        # First, we need to load an RGB image and convert it to a 4D floating point tensor, as the network expectes an input tensor of size: `(batch_size, 3, height, width)`.
        for target_idx in range(801, 901):
            channel_list = []
            for sub_dir in ['red_step1_again', 'green_step1_again',  'blue_step1_again', 'color_step1_again']:
                if 'color' not in sub_dir:
                    #img_path = os.path.join(root_dir, sub_dir, f'{target_idx:04d}-pred-phase.png')
                    
                    img = Image.open(img_path)
                    img = transforms.ToTensor()(img).unsqueeze(0)
                    img = img.repeat(1,3,1,1)
                else: 
                    img_path = os.path.join(root_dir, sub_dir, f'{target_idx:04d}-pred-phase-rgb.png')
                    img_path = '/jixie/yujie_data/DIV2K_valid_HR/0801-gt-target.png'
                    img = Image.open(img_path)
                    img = transforms.ToTensor()(img).unsqueeze(0)

                x = img
                x = ConstantPad(x)
                print(x.shape)
                with torch.no_grad():
                    x = x.to(device)
                    out_net = net.forward(x)
                    #x_hat_y_hat = out_net['x_hat'].clamp_(0, 1)
                
                    if key == 'hyperprior18':
                        compress_out = net.compress(x)
                        decompress_out = net.decompress(compress_out['strings'], compress_out['shape'])
                        x_hat_y_hat = decompress_out['x_hat'].clamp(0,1)
                    else: 
                        compress_out = net.compress(x)
                        decompress_out = net.decompress(compress_out['strings'], compress_out['shape'])
                        x_hat_y_hat = decompress_out['x_hat'].clamp(0,1)

                    # ## 2. Comparison to classical codecs

                    psnr = compute_psnr(x, x_hat_y_hat)
                    ssim = compute_msssim(x, x_hat_y_hat)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    target_bpp = compute_bpp(out_net)
                    bpp_list.append(target_bpp)
                    # ## 3. Latent visualization                
                    detailed_bpp(out_net)
                    print('Decoded quantized latent:')
                    print(f'PSNR: {psnr:.4}dB')
                    print(f'MS-SSIM: {ssim:.4f}')
                    print(f'Bit-rate: {target_bpp:.3f} bpp')

                    writer.write(f'Id: {target_idx:04d}: \n')
                    writer.write(f'PSNR: {psnr:.4f}\t MS-SSIM: {ssim:.4f} \t  Bit-rate: {target_bpp:.4f} \n')
                
                    result = x_hat_y_hat
                    result = result.cpu().numpy()
                    result = result.squeeze().transpose((1,2,0))[8:-8,:,::-1]
                    print(result.shape)
                    result_8bit = (result * 255).astype(np.uint8)
                    save_name = os.path.join(result_dir, f'{target_idx:04d}-pred-phase-{key}-quality{quality}.png')
                    cv2.imwrite(save_name, result_8bit)
            
                writer.write('**********************\n')
                avg_bpp = sum(bpp_list) / len(bpp_list)
                avg_ssim = sum(ssim_list) / len(ssim_list)
                avg_psnr = sum(psnr_list) / len(psnr_list)

                writer.write(f'avg PSNR: {avg_psnr:.4f} \n')
                writer.write(f'avg MS-SSIM: {avg_ssim:.4f} \n')
                writer.write(f'avg bit-rate: {avg_bpp:.4f} \n')
                writer.write(f'Count: {len(ssim_list)}\n')


                writer.close()




