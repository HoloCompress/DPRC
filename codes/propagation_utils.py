"""
This is the script containing all tuility functions used for the implementation.

This file is borrowed from the implementation of the following paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import math
import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.modules.loss as ll

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def mul_complex(t1, t2):
    """multiply two complex valued tensors element-wise. the two last dimensions are
    assumed to be the real and imaginary part

    complex multiplication: (a+bi)(c+di) = (ac-bd) + (bc+ad)i
    """
    # real and imaginary parts of first tensor
    a, b = t1.split(1, 4)
    # real and imaginary parts of second tensor
    c, d = t2.split(1, 4)

    # multiply out
    return torch.cat((a * c - b * d, b * c + a * d), 4)


def div_complex(t1, t2):
    """divide two complex valued tensors element-wise. the two last dimensions are
    assumed to be the real and imaginary part

    complex division: (a+bi) / (c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2) i
    """
    # real and imaginary parts of first tensor
    (a, b) = t1.split(1, 4)
    # real and imaginary parts of second tensor
    (c, d) = t2.split(1, 4)

    # get magnitude
    mag = torch.mul(c, c) + torch.mul(d, d)

    # multiply out
    return torch.cat(((a * c + b * d) / mag, (b * c - a * d) / mag), 4)


def reciprocal_complex(t):
    """element-wise inverse of complex-valued tensor

    reciprocal of complex number z=a+bi:
    1/z = a / (a^2 + b^2) - ( b / (a^2 + b^2) ) i
    """
    # real and imaginary parts of first tensor
    (a, b) = t.split(1, 4)

    # get magnitude
    mag = torch.mul(a, a) + torch.mul(b, b)

    # multiply out
    return torch.cat((a / mag, -(b / mag)), 4)


def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


def replace_amplitude(field, amplitude):
    """takes a complex-valued tensor with real/imag channels, converts to
    amplitude/phase, replaces amplitude, then converts back to real/imag

    resolution of both tensors should be (M, N, height, width, 1 or 2)
    """
    # convert to amplitude and phase
    field_amp, field_phase = rect_to_polar(field[..., 0], field[..., 1])

    # get number of dimensions
    ndims = np.shape(list(amplitude.size()))[0]
    if ndims < 5:
        amplitude = amplitude.unsqueeze(4)

    # replace amplitude with target amplitude and convert back to real/imag
    real, imag = polar_to_rect(amplitude, field_phase.unsqueeze(4))

    # concatenate
    return torch.cat((real, imag), 4)


def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted


def ifft2(tensor_re, tensor_im, shift=False):
    """Applies a 2D ifft to the complex tensor represented by tensor_re and _im"""
    tensor_out = torch.stack((tensor_re, tensor_im), 4)

    if shift:
        tensor_out = ifftshift(tensor_out)
    (tensor_out_re, tensor_out_im) = torch.ifft(tensor_out, 2, True).split(1, 4)

    tensor_out_re = tensor_out_re.squeeze(4)
    tensor_out_im = tensor_out_im.squeeze(4)

    return tensor_out_re, tensor_out_im


def fft2(tensor_re, tensor_im, shift=False):
    """Applies a 2D fft to the complex tensor represented by tensor_re and _im"""
    # fft2
    (tensor_out_re, tensor_out_im) = torch.fft(torch.stack((tensor_re, tensor_im), 4), 2, True).split(1, 4)

    tensor_out_re = tensor_out_re.squeeze(4)
    tensor_out_im = tensor_out_im.squeeze(4)

    # apply fftshift
    if shift:
        tensor_out_re = fftshift(tensor_out_re)
        tensor_out_im = fftshift(tensor_out_im)

    return tensor_out_re, tensor_out_im


def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def pad_stacked_complex(field, pad_width, padval=0, mode='constant'):
    """Helper for pad_image() that pads a real padval in a complex-aware manner"""
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked_complex dimension
        return nn.functional.pad(field, pad_width, mode=mode)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, mode=mode, value=padval)
        imag = nn.functional.pad(imag, pad_width, mode=mode, value=0)
        return torch.stack((real, imag), -1)


def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field


def gen_mask(field, target_shape, pytorch=True, stacked_complex=True):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]

        field[(..., *crop_slices)] = 0.0
        mask = 1.0 - field
        #print('return from here')
        return mask
    else:
        return field

def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field


def srgb_gamma2lin(im_in):
    """converts from sRGB to linear color space"""
    thresh = 0.04045
    im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    return im_out


def srgb_lin2gamma(im_in):
    """converts from linear to sRGB color space"""
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def phasemap_8bit(phasemap, inverted=True):
    """convert a phasemap tensor into a numpy 8bit phasemap that can be directly displayed

    Input
    -----
    :param phasemap: input phasemap tensor, which is supposed to be in the range of [-pi, pi].
    :param inverted: a boolean value that indicates whether the phasemap is inverted.

    Output
    ------
    :return: output phasemap, with uint8 dtype (in [0, 255])
    """

    out_phase = phasemap.cpu().detach().squeeze().numpy()
    out_phase = ((out_phase + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - out_phase) * 255).round().astype(np.uint8)  # quantized to 8 bits
    else:
        phase_out_8bit = ((out_phase) * 255).round().astype(np.uint8)  # quantized to 8 bits

    return phase_out_8bit


def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def propagate_field(input_field, propagator, prop_dist=0.2, wavelength=520e-9, feature_size=(6.4e-6, 6.4e-6),
                    prop_model='ASM', dtype=torch.float32, precomputed_H=None):
    """
    A wrapper for various propagation methods, including the parameterized model.
    Note that input_field is supposed to be in Cartesian coordinate, not polar!

    Input
    -----
    :param input_field: pytorch tensor shape of (1, C, H, W, 2), the field before propagation, in X, Y coordinates
    :param prop_dist: propagation distance in m.
    :param wavelength: wavelength of the wave in m.
    :param feature_size: pixel pitch
    :param prop_model: propagation model ('ASM', 'MODEL', 'fresnel', ...)
    :param trained_model: function or model instance for propagation
    :param dtype: torch.float32 by default
    :param precomputed_H: Propagation Kernel in Fourier domain (could be calculated at the very first time and reuse)

    Output
    -----
    :return: output_field: pytorch tensor shape of (1, C, H, W, 2), the field after propagation, in X, Y coordinates
    """

    if prop_model == 'ASM':
        output_field = propagator(u_in=input_field, z=prop_dist, feature_size=feature_size, wavelength=wavelength,
                                  dtype=dtype, precomped_H=precomputed_H)
    elif 'MODEL' in prop_model:
        # forward propagate through our citl-calibrated model
        _, input_phase = rect_to_polar(input_field[..., 0], input_field[..., 1])
        output_field = propagator(input_phase)
    else:
        raise ValueError('Unexpected prop_model value')

    return output_field


def write_sgd_summary(slm_phase, out_amp, target_amp, k,
                      writer=None, path=None, s=0., prefix='test'):
    """tensorboard summary for SGD

    :param slm_phase: Use it if you want to save intermediate phases during optimization.
    :param out_amp: PyTorch Tensor, Field amplitude at the image plane.
    :param target_amp: PyTorch Tensor, Ground Truth target Amplitude.
    :param k: iteration number.
    :param writer: SummaryWriter instance.
    :param path: path to save image files.
    :param s: scale for SGD algorithm.
    :param prefix:
    :return:
    """
    loss = nn.MSELoss().to(out_amp.device)
    loss_value = loss(s * out_amp, target_amp)
    psnr_value = psnr(target_amp.squeeze().cpu().detach().numpy(), (s * out_amp).squeeze().cpu().detach().numpy())
    ssim_value = ssim(target_amp.squeeze().cpu().detach().numpy(), (s * out_amp).squeeze().cpu().detach().numpy())

    if writer is not None:
        writer.add_image(f'{prefix}_Recon/amp', (s * out_amp).squeeze(0), k)
        writer.add_scalar(f'{prefix}_loss', loss_value, k)
        writer.add_scalar(f'{prefix}_psnr', psnr_value, k)
        writer.add_scalar(f'{prefix}_ssim', ssim_value, k)


def write_gs_summary(slm_field, recon_field, target_amp, k, writer, roi=(880, 1600), prefix='test'):
    """tensorboard summary for GS"""
    _, slm_phase = rect_to_polar(slm_field[..., 0], slm_field[..., 1])
    recon_amp, recon_phase = rect_to_polar(recon_field[..., 0], recon_field[..., 1])
    loss = nn.MSELoss().to(recon_amp.device)

    recon_amp = crop_image(recon_amp, target_shape=roi, stacked_complex=False)
    target_amp = crop_image(target_amp, target_shape=roi, stacked_complex=False)

    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                  / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))

    loss_value = loss(recon_amp, target_amp)
    psnr_value = psnr(target_amp.squeeze().cpu().detach().numpy(), recon_amp.squeeze().cpu().detach().numpy())
    ssim_value = ssim(target_amp.squeeze().cpu().detach().numpy(), recon_amp.squeeze().cpu().detach().numpy())

    if writer is not None:
        writer.add_image(f'{prefix}_Recon/amp', recon_amp.squeeze(0), k)
        writer.add_scalar(f'{prefix}_loss', loss_value, k)
        writer.add_scalar(f'{prefix}_psnr', psnr_value, k)
        writer.add_scalar(f'{prefix}_ssim', ssim_value, k)


def get_psnr_ssim(recon_amp, target_amp, multichannel=False):
    """get PSNR and SSIM metrics"""
    psnrs, ssims = {}, {}

    # amplitude
    psnrs['amp'] = psnr(target_amp, recon_amp)
    ssims['amp'] = ssim(target_amp, recon_amp, multichannel=multichannel)

    # linear
    target_linear = target_amp**2
    recon_linear = recon_amp**2
    psnrs['lin'] = psnr(target_linear, recon_linear)
    ssims['lin'] = ssim(target_linear, recon_linear, multichannel=multichannel)

    # srgb
    target_srgb = srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_linear, 0.0, 1.0))
    psnrs['srgb'] = psnr(target_srgb, recon_srgb)
    ssims['srgb'] = ssim(target_srgb, recon_srgb, multichannel=multichannel)

    return psnrs, ssims


        
