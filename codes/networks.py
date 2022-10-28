import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import importlib
import src.hyperprior as hyperprior
import utils.utils as utils
from propagation_ASM import propagation_ASM
from utils.pytorch_prototyping.pytorch_prototyping import Unet 
from utils.extra_modules import Generator, MultiScale_Encoder
import time
grads = {}


# Functions
##############################################################################
def get_network(config):
    if config.compress:
        is_stage2 = True
    else:
        is_stage2 = False 
    
    if config.is_train:
        entropy_code = False
    else:
        entropy_code = True

    hyper_prior = hyperprior.Hyperprior(bottleneck_capacity=8,
            likelihood_type='gaussian', entropy_code=entropy_code)
      
    net = Model(config.prop_dist, config.wavelength,  feature_size=config.feature_size, is_stage2=is_stage2, initial_phase=InitialPhaseUnet(4, 16),
                                   final_phase_only=ComPhaseNet(num_in=2), hyper_prior=hyper_prior)
    return net



def save_grad(name):
    def hook(grad):
        grads[name] =  grad 
    return hook


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

# Classes 
##############################################################################
# The Intial Phase Predictor (IP)
class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d):
        super(InitialPhaseUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase

# The Phase encoder and Phase decoder (Ep and Dp)
class ComPhaseNet(nn.Module):
    def __init__(self, num_in=2, latent_channels=8, num_down=2, num_up=2, norm=nn.BatchNorm2d, use_dropout=False):
        super(ComPhaseNet, self).__init__()

        self.encoder = MultiScale_Encoder(per_layer_out_ch=[32, 64], in_channels=num_in, use_dropout=use_dropout,
                                     out_channels=latent_channels, norm=norm)
        self.decoder = Generator(filters=[64, 32, 16], in_channels=latent_channels, activation='relu', 
                                n_residual_blocks=4)

    def forward(self, input_x, return_time=False):
        latent_map = self.encode(input_x)
        output = self.decode(latent_map) 
        return latent_map, output
    
    def encode(self, input_x):
        latent_map = self.encoder(input_x)
        return latent_map 


    def decode(self, latent_map):
        output = self.decoder(latent_map)
        return output


# Full Framework
class Model(nn.Module):
    """
    Class initialization parameters
    -------------------------------
    distance: propagation distance between the SLM and the target plane 
    wavelength: the wavelength of the laser 
    feature_size: the pixel pitch of the SLM 
    initial_phase:  a module that is used to predict the initial phase at the target plane 
    final_phase_only: the module that is used to encode the complex wavefield at SLM plane into a phase-only hologram
    proptype: chooses the propagation operator. Default ASM.
    linear_conv: if True, pads for linear conv for propagation. Default True
    """
    def __init__(self, distance=0.1, wavelength=520e-9, feature_size=6.4e-6, is_stage2=False, 
                  initial_phase=None, final_phase_only=None, proptype='ASM', linear_conv=True, hyper_prior=None):

        super(Model, self).__init__()

        # submodules
        self.initial_phase = initial_phase
        self.final_phase_only = final_phase_only
        self.is_stage2 = is_stage2

        # propagation parameters
        self.wavelength = wavelength
        self.feature_size = (feature_size
                              if hasattr(feature_size, '__len__')
                              else [feature_size] * 2)
        self.distance = -distance


        # objects to precompute
        self.precomped_H = None
        self.hyper_prior = hyper_prior

        # change out the propagation operator
        if proptype == 'ASM':
            self.prop = propagation_ASM
        else:
            ValueError(f'Unsupported prop type {proptype}')

        self.linear_conv = linear_conv
        self.device = torch.device('cpu')

        s = torch.tensor(0.95, requires_grad=True)
        self.s = torch.nn.Parameter(s)
          

    def forward(self, target_amp, is_train=True):
        # compute some initial phase, convert to real+imag representation
        start = time.time()
        init_phase = self.initial_phase(target_amp)
        real, imag = utils.polar_to_rect(target_amp, init_phase)
        target_complex = torch.stack((real, imag), -1)

        # precompute the propagation kernel once it is not provided
        if self.precomped_H is None:
            self.precomped_H = self.prop(target_complex,
                                         self.feature_size,
                                         self.wavelength,
                                         self.distance,
                                         return_H=True,
                                         linear_conv=self.linear_conv)
            self.precomped_H = self.precomped_H.to(self.device).detach()
            self.precomped_H.requires_grad = False


        # Propagate the wavefield to the SLM plane 
        slm_field = self.prop(target_complex, self.feature_size,
                              self.wavelength, self.distance,
                              precomped_H=self.precomped_H,
                              linear_conv=self.linear_conv)

        # Transform it to amplitude-phase form
        amp, ang = utils.rect_to_polar(slm_field[..., 0], slm_field[..., 1])
        slm_amp_phase = torch.cat((amp, ang), -3)

        first_stage_end = time.time() 
        first_elapsed = first_stage_end - start

        # if the hyper_prior is None, it indicates that we are trainint the sub-network PRN.
        if not self.is_stage2:
            latent_map, pred_phase = self.final_phase_only(slm_amp_phase)
            return latent_map, pred_phase
        else:
            if is_train:
                # During trainng stage, we didn't conduct entropy coding process
                latent_map = self.final_phase_only.encode(slm_amp_phase)   
                hyperinfo = self.hyper_prior(latent_map, spatial_shape=target_amp.size()[2:])     
                latents_quantized = hyperinfo.decoded
                total_nbpp = hyperinfo.total_nbpp
                total_qbpp = hyperinfo.total_qbpp      
                decode_phase = self.final_phase_only.decode(latents_quantized)
                return latent_map, decode_phase, total_nbpp, total_qbpp
                
            else:
                # In test stage, we need to conduct entropy coding to transform the latent maps into binary files.
                encode_start = time.time()
                latent_map = self.final_phase_only.encode(slm_amp_phase)  
                #latent_map = latent_map.to("cuda:1")
                compression_output = self.hyper_prior.compress_forward(latent_map, spatial_shape=target_amp.size()[2:])
                encode_elapsed = time.time() - encode_start

                decode_start = time.time()
                latents_decoded = self.hyper_prior.decompress_forward(compression_output, device=latent_map.device)
                #latents_decoded = latents_decoded.to("cuda:0")
                new_start = time.time()
                pred_phase = self.final_phase_only.decode(latents_decoded)
                decode_elapsed = time.time() - decode_start  
                net_decode_time = time.time() - new_start 
                #decode_elapsed = time.time() - decode_start                
                return [latent_map, pred_phase, compression_output, init_phase, amp, ang, first_elapsed, net_decode_time, decode_elapsed]
               
        
    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.precomped_H is not None:
            slf.precomped_H = slf.precomped_H.to(*args, **kwargs)
        if self.hyper_prior is not None:
            self.hyper_prior = self.hyper_prior.to(*args, **kwargs)

        # try setting dev based on some parameter, default to cpu
        try:
            slf.device = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.device = device_arg
        return slf


if __name__ == '__main__':
    hyper_prior = hyperprior.Hyperprior(bottleneck_capacity=8,
        likelihood_type='gaussian', entropy_code=False)
    hyper_prior = None
    net = Model(200 * 1e-2, 638 * 1e-9,  feature_size=6.4e-6, initial_phase=InitialPhaseUnet(4, 16),
                                   final_phase_only=ComPhaseNet(num_in=2), hyper_prior=hyper_prior)
    
    net = net.to('cuda:0')
    print(net)
    input_data = torch.randn(2, 1, 1072, 1920)
    input_data = input_data.cuda()
    with torch.no_grad():
      output = net(input_data, is_train=False)
      print(output[0].shape, output[1].shape)