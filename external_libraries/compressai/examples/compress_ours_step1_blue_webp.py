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
import time
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


result_root_dir = '/jixie/yujie_codes/holonet_compression/our_final_results/'
#data_dir = '/mnt/mnt2/codes/yujie_codes/holonet_compression/hardware_experiment/SGD_ASM/'

ConstantPad = torch.nn.ConstantPad2d(padding=(0, 0, 8, 8), value=0)
model_dict = {'webp': pillow_encode}
for key, value in model_dict.items():
    device='cuda'
    quality_list = list(range(5, 101, 5))
    # ### Load image and convert to 4D float tensor

    # First, we need to load an RGB image and convert it to a 4D floating point tensor, as the network expectes an input tensor of size: `(batch_size, 3, height, width)`.
    for sub_dir in ['blue']:
        result_sub_dir = f'ours_step1_{sub_dir}_{key}_0511'
        #result_dir = os.path.join(root_dir, result_sub_dir)
        result_dir = os.path.join(result_root_dir, result_sub_dir)
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        blue_data_dir = '/jixie/yujie_codes/holonet_compression/HoloNet_yujie_BLUE_5firstVGG_compression_multiScale_MSE_WFFT1e-8_VGG_2x_msssim_again/train_log/results/test-ckpt-13-rescale'
        img_list = os.listdir(blue_data_dir)
        img_list = [os.path.join(blue_data_dir, file) for file in img_list if 'pred-phase.png' in file]
        img_list.sort()
        print(len(img_list))
        for img_file in img_list:
            img = Image.open(img_file)
            #img = cv2.imread(img_file)
            for quality in quality_list:
                rec, bpp = model_dict[key](img, fmt=key, quality=quality)
                file_name = img_file.split('/')[-1][:-4]
                save_name = os.path.join(result_dir, file_name+'_quality%d.webp'%(quality))
                rec.save(save_name)
                #rec.save
                print(save_name)
                '''
                file_name = img_file.split('/')[-1][:-4]
                save_name = os.path.join(result_dir, file_name+'_quality%d.jp2'%(quality))
                print(save_name)
                cv2.imwrite(save_name, img,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                '''




