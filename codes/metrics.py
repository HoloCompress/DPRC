from collections import defaultdict
from datetime import datetime
import os
import sys
import math
import random
import json
import glob
import logging
import shutil
import torch.nn.functional as F
from math import exp
import numpy as np
import torch
from torchvision import transforms

from PIL import Image, ImageFont, ImageDraw
#from skimage.measure import compare_psnr, compare_ssim

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

import cv2

# from pytorch_msssim import ssim as ssim_pth


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).cuda()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().cuda()
    return window

def calc_psnr(pred, gt, mask=None):
    '''
        Here we assume quantized(0-255) arguments.
    '''
    diff = (pred - gt).div(255)

    if mask is not None:
        mse = diff.pow(2).sum() / (3 * mask.sum())
    else:
        mse = diff.pow(2).mean() + 1e-8    # mse can (surprisingly!) reach 0, which results in math domain error

    return -10 * math.log10(mse)

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    
    # mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    # mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1 = F.conv2d(F.pad(img1, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)
    mu2 = F.conv2d(F.pad(img2, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    # sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    sigma1_sq = F.conv2d(F.pad(img1 * img1, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(img2 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(F.pad(img1 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def init_lpips_eval():
    LPIPS_dir = "../PerceptualSimilarity"
    LPIPS_net = "squeeze"
    sys.path.append(LPIPS_dir)
    from models import dist_model as dm
    print("Initialize Distance model from %s" % LPIPS_net)
    lpips_model = dm.DistModel()
    lpips_model.initialize(model='net-lin',net='squeeze', use_gpu=True,
        model_path=os.path.join(LPIPS_dir, 'weights/v0.1/%s.pth' % LPIPS_net))
    return lpips_model


##########################
# Evaluations
##########################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split('+')
    for l in loss_list:
        _, loss_type = l.split('*')
        loss_specifics[loss_type] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics


def init_meters(loss_str):
    losses = init_losses(loss_str)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    lpips = AverageMeter()
    return losses, psnrs, ssims, lpips


def quantize(img, rgb_range=255):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


def calc_psnr(pred, gt, mask=None):
    '''
        Here we assume quantized(0-255) arguments.
    '''
    diff = (pred - gt).div(255)

    if mask is not None:
        mse = diff.pow(2).sum() / (3 * mask.sum())
    else:
        mse = diff.pow(2).mean() + 1e-8    # mse can (surprisingly!) reach 0, which results in math domain error

    return -10 * math.log10(mse)


def calc_ssim(img1, img2, datarange=255.):
    im1 = img1.numpy().transpose(1, 2, 0).astype(np.uint8)
    im2 = img2.numpy().transpose(1, 2, 0).astype(np.uint8)
    return compare_ssim(im1, im2, datarange=datarange, multichannel=True, gaussian_weights=True)


def calc_metrics(im_pred, im_gt, mask=None):
    q_im_pred = quantize(im_pred.data, rgb_range=1.)
    q_im_gt = quantize(im_gt.data, rgb_range=1.)
    if mask is not None:
        q_im_pred = q_im_pred * mask
        q_im_gt = q_im_gt * mask
    psnr = calc_psnr(q_im_pred, q_im_gt, mask=mask)
    # ssim = calc_ssim(q_im_pred.cpu(), q_im_gt.cpu())
    ssim = ssim_matlab(q_im_pred.unsqueeze(0), q_im_gt.unsqueeze(0), val_range=255)
    return psnr, ssim


def eval_LPIPS(model, im_pred, im_gt):
    im_pred = 2.0 * im_pred - 1
    im_gt = 2.0 * im_gt - 1
    dist = model.forward(im_pred, im_gt)[0]
    return dist


def eval_metrics(output, gt, psnrs, ssims, lpips, lpips_model=None, mask=None, psnrs_masked=None, ssims_masked=None):
    # PSNR should be calculated for each image
    for b in range(gt.size(0)):
        psnr, ssim = calc_metrics(output[b], gt[b], None)
        psnrs.update(psnr)
        ssims.update(ssim)
        if mask is not None:
            psnr_masked, ssim_masked = calc_metrics(output[b], gt[b], mask[b])
            psnrs_masked.update(psnr_masked)
            ssims_masked.update(ssim_masked)
        if lpips_model is not None:
            _lpips = eval_LPIPS(lpips_model, output[b].unsqueeze(0), gt[b].unsqueeze(0))
            lpips.update(_lpips)


##########################
# ETC
##########################

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def makedirs(path):
    if not os.path.exists(path):
        print("[*] Make directories : {}".format(path))
        os.makedirs(path)

def remove_file(path):
    if os.path.exists(path):
        print("[*] Removed: {}".format(path))
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    print("[*] {} has backup: {}".format(path, new_path))

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# TensorBoard
def log_tensorboard(writer, losses, psnr, ssim, lpips, lr, timestep, mode='train'):
    for k, v in losses.items():
        writer.add_scalar('Loss/%s/%s' % (mode, k), v.avg, timestep)
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    if lpips is not None:
        writer.add_scalar('LPIPS/%s' % mode, lpips, timestep)
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)


###########################
###### VISUALIZATIONS #####
###########################

def save_image(img, path):
    # img : torch Tensor of size (C, H, W)
    q_im = quantize(img.data.mul(255))
    if len(img.size()) == 2:    # grayscale image
        im = Image.fromarray(q_im.cpu().numpy().astype(np.uint8), 'L')
    elif len(img.size()) == 3:
        im = Image.fromarray(q_im.permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
    else:
        pass
    im.save(path)

def save_batch_images(output, imgpath, save_dir, alpha=0.5):
    GEN = save_dir.find('-gen') >= 0 or save_dir.find('stereo') >= 0
    q_im_output = [quantize(o.data, rgb_range=1.) for o in output]
    for b in range(output[0].size(0)):
        paths = imgpath[0][b].split('/')
        if GEN:
            save_path = save_dir
        else:
            save_path = os.path.join(save_dir, paths[-3], paths[-2])
        makedirs(save_path)
        for o in range(len(output)):
            if o % 2 == 1 or len(output) == 1:
                output_img = Image.fromarray(q_im_output[o][b].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
                if GEN:
                    _imgname = imgpath[o//2][b].split('/')[-1]
                    imgname = "%s-%.04f.png" % (_imgname, alpha)
                else:
                    imgname = imgpath[o//2][b].split('/')[-1]

                if save_dir.find('voxelflow') >= 0:
                    #imgname = imgname.replace('gt', 'ours')
                    imgname = 'frame_01_ours.png'
                elif save_dir.find('middlebury') >= 0:
                    imgname = 'frame10i11.png'
                
                output_img.save(os.path.join(save_path, imgname))


def save_batch_images_test(output, imgpath, save_dir, alpha=0.5):
    GEN = save_dir.find('-gen') >= 0 or save_dir.find('stereo') >= 0
    q_im_output = [quantize(o.data, rgb_range=1.) for o in output]
    for b in range(output[0].size(0)):
        paths = imgpath[0][b].split('/')
        if GEN:
            save_path = save_dir
        else:
            save_path = os.path.join(save_dir, paths[-3], paths[-2])
        makedirs(save_path)
        for o in range(len(output)):
            # if o % 2 == 1 or len(output) == 1:
                # print("   ", o, b, imgpath[o][b])
                output_img = Image.fromarray(q_im_output[o][b].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
                if GEN:
                    _imgname = imgpath[o][b].split('/')[-1]
                    imgname = "%s-%.04f.png" % (_imgname, alpha)
                else:
                    imgname = imgpath[o][b].split('/')[-1]

                if save_dir.find('voxelflow') >= 0:
                    #imgname = imgname.replace('gt', 'ours')
                    imgname = 'frame_01_ours.png'
                elif save_dir.find('middlebury') >= 0:
                    imgname = 'frame10i11.png'

                output_img.save(os.path.join(save_path, imgname))


def save_images_test(output, imgpath, save_dir, alpha=0.5):
    q_im_output = [quantize(o.data, rgb_range=1.) for o in output]
    for b in range(output[0].size(0)):
        paths = imgpath[1][b].split('/')
        save_path = os.path.join(save_dir, paths[-3], paths[-2])
        makedirs(save_path)
        # Output length is one
        output_img = Image.fromarray(q_im_output[0][b].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
        imgname = imgpath[1][b].split('/')[-1]

        # if save_dir.find('voxelflow') >= 0:
        #     imgname = 'frame_01_ours.png'
        # elif save_dir.find('middlebury') >= 0:
        #     imgname = 'frame10i11.png'

        output_img.save(os.path.join(save_path, imgname))


def save_images_multi(output, imgpath, save_dir, idx=1):
    q_im_output = [quantize(o.data, rgb_range=1.) for o in output]
    for b in range(output[0].size(0)):
        paths = imgpath[0][b].split('/')
        # save_path = os.path.join(save_dir, paths[-3], paths[-2])
        # makedirs(save_path)
        # Output length is one
        output_img = Image.fromarray(q_im_output[0][b].permute(1, 2, 0).cpu().numpy().astype(np.uint8), 'RGB')
        # imgname = imgpath[idx][b].split('/')[-1]
        imgname = '%s_%03d.png' % (paths[-1], idx)

        output_img.save(os.path.join(save_dir, imgname))


def make_video(out_dir, gt_dir, gt_first=False):
    gt_ext = '/*.png'
    frames_all = sorted(glob.glob(out_dir + '/*.png') + glob.glob(gt_dir + gt_ext), \
        key=lambda frame: frame.split('/')[-1])
    print("# of total frames : %d" % len(frames_all))
    if gt_first:
        print("Appending GT in front..")
        frames_all = sorted(glob.glob(gt_dir + gt_ext)) + frames_all
        print("# of total frames : %d" % len(frames_all))

    # Read the first image to determine height and width
    frame = cv2.imread(frames_all[0])
    h, w, _ = frame.shape

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_dir + '/slomo.mp4', fourcc, 30, (w, h))
    for p in frames_all:
        #print(p)
        # TODO: add captions (e.g. 'GT', 'slow motion x4')
        frame = cv2.imread(p)
        fh, fw = frame.shape[:2]
        #print(fh, fw, h, w)
        if fh != h or fw != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        out.write(frame)

def check_already_extracted(vid):
    return bool(os.path.exists(vid + '/00001.png'))
