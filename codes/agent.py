import torch
import torch.nn as nn
from abc import abstractmethod
import numpy as np
from networks import get_network
from losses import *
from reconstruction import holo_propagator
import torch.nn.functional as F
from losses import VGGLoss
from loss.loss_provider import LossProvider
from pytorch_msssim import ms_ssim
from src.loss import losses
from base_agent import BaseAgent



def get_agent(config):
    return HologramAgent(config)

class HologramAgent(BaseAgent):
    def build_net(self, config):
        net = get_network(config)
        net = net.to(self.device)
        return net

    def set_loss_function(self):
        self.criterionMSE = nn.MSELoss(size_average=True).to(self.device) # mse loss criterion
        self.criterionGrad = Grad_loss().to(self.device)  # gradient loss criterion
        self.criterionVGG_recon = VGGLoss(device=self.device)  # vgg loss criterion
        provider = LossProvider()
        self.criterionWFFT = provider.get_loss_function('watson-fft', deterministic=False, colorspace='grey',
                                                             pretrained=True, reduction='sum')  # watson-fft loss criterion
        self.criterionWFFT = self.criterionWFFT.to(self.device)
        self.criterionMS_SSIM = ms_ssim # ms-ssim loss criterion
        # ASM propagator used for reconstruction
        self.propagator = holo_propagator(self.config.wavelength, self.config.prop_dist, self.config.feature_size) 

    def scale(self, x, label, weight=None):  
        '''
        A function used for adjust the scale between the input image (param: x) and the target image (param: label)
        '''
        ##### x.size() should be NxCxHxW
        alpha = torch.cuda.FloatTensor(x.size())

        denominator = torch.mul(x, label)
        numerator = torch.mul(x, x)

        alpha_vector = torch.div(denominator.sum(-1).sum(-1),numerator.sum(-1).sum(-1))
        #print(alpha_vector)
        alpha_vector[alpha_vector != alpha_vector] = 0  #### changes nan to 0
        alpha_vector = torch.clamp(alpha_vector, min=0.1, max=10.0)

        for i in range(x.size(0)):
            for j in range(x.size(1)):
                alpha[i][j].fill_(alpha_vector[i][j])

        x = torch.mul(torch.autograd.Variable(alpha, requires_grad=False), x)
    
        if weight is not None:
            x = torch.mul(torch.autograd.Variable(weight,requires_grad=False), x)
            label = torch.mul(torch.autograd.Variable(weight, requires_grad=False), label)

            tensor1 = torch.mul(weight, weight)
            self.weight_normal = tensor1.sum()
            if self.weight_normal != 0:
                self.weight_normal = 1.0 / self.weight_normal
    
        return x
 
    def forward(self, data):
        target_amp  = data[0].cuda()
        mask = data[1].cuda()
    
        #mask = data[3].cuda()
        loss_dict = {}

        if not self.config.compress:
            # if config.compress is False, we are in the first stage training
            output = self.net(target_amp, self.config.is_train)
            latent_map = output[0]
            pred_phase = output[1]
        else:
            if self.config.is_train:
                latent_map, pred_phase, total_nbpp, total_qbpp = self.net(target_amp)
                weighted_rate, rate_penalty = losses.weighted_rate_loss(self.config, total_nbpp=total_nbpp,
                                total_qbpp=total_qbpp, step_counter=self.clock.step, ignore_schedule=False)
                rate_loss = weighted_rate.to("cuda:0")  # this is the rate loss used for lowering the bit-rate
                total_qbpp = total_qbpp.detach() # just for evaluation and curve visualization, with no gradient back-propagation
                total_nbpp = total_nbpp.detach() # just for evaluation and curve visualization, with no gradient back-propgation 
                loss_dict.update({'rate_loss': rate_loss, 'eval_bpp':total_qbpp, 'eval_nbpp':total_nbpp})
            else:
                output = self.net(target_amp, self.config.is_train)
                pred_phase = output[1]   # fetch the predicted phase for reconstruction
                 
        # propagate the hologram to the target plane
        recon_img = self.propagator(pred_phase)
        # adjust the scale of the recontruction image
        recon_img_scale = torch.sqrt(torch.pow(recon_img, 2) * 0.95)
        recon_img_scale = recon_img_scale * mask  

        # Multiply with a mask so that the target and reconstruction just contain the region of interest
        recon_img = recon_img * mask
        target_amp = target_amp * mask
        
        # calculate loss terms for reconstruction
        wfft_loss = self.config.w_wfft * self.criterionWFFT(recon_img_scale, target_amp)        
        mse_loss = self.config.w_mse * self.criterionMSE(recon_img_scale, target_amp)

        if not self.config.is_train:
            # for speeding up test and saving memory cost, turn off vgg loss in test.
            vgg_loss = torch.tensor([0.0]).to(self.device) 
        else:
            vgg_loss = self.config.w_vgg * self.criterionVGG_recon(recon_img_scale, target_amp)
        
        msssim_loss = self.config.w_ssim * (1-self.criterionMS_SSIM(recon_img_scale, target_amp, data_range=1))

        recon_losses = {'watson-fft': wfft_loss, 'mse': mse_loss, 'ms_ssim':msssim_loss, 
                         'vgg': vgg_loss}
        
        # if we are in stage2 training, multiply the recon loss terms with an amplifier
        for key, value in recon_losses.items():
            recon_losses[key] = value * self.config.w_recon 
        
        # update the loss terms for reconstruction into the loss dict
        loss_dict.update(recon_losses)

        # In training procese, just output the these items with a fixed order, which will be used for tensorboard visualization
        if self.config.is_train:
            output = [latent_map, pred_phase, recon_img, recon_img_scale, target_amp]   
        else:
            # In test stage, just append the recon image and scaled recon image in the output list
            output.extend([recon_img, recon_img_scale])
        return output, loss_dict


    def infer(self, data):
        self.net.eval()
        data = data.cuda()
        with torch.no_grad():
           output = self.net(data)
        return output

    def visualize_batch(self, data, mode, outputs=None):
        '''The function used for visualizing the image produced from the network and hologram reconstruction'''
        tb = self.train_tb if mode == 'train' else self.val_tb
        input_img = data[0][0,...]
        input_img = (input_img + 1 )/2.0
        
        latent_map, pred_phase, recon_img,recon_img_scale, target_amp = outputs

        target_recon = target_amp[0,...].cpu()
        pred_phase = pred_phase[0,...].detach().cpu()
     
        pred_phase = F.interpolate(pred_phase.unsqueeze(0), scale_factor=0.25)[0,...]
        pred_phase = (pred_phase +  np.pi ) / (2 * np.pi) 
        target_recon = F.interpolate(target_recon.unsqueeze(0), scale_factor=0.25)[0,...]
        recon_img = F.interpolate(recon_img, scale_factor=0.25)[0,...]
        recon_img_scale = F.interpolate(recon_img_scale, scale_factor=0.25)[0,...]
        latent_map1 = latent_map[0,0,...].detach().cpu()
        latent_map8 = latent_map[0,7,...].detach().cpu()

        tb.add_image('pred_phase', pred_phase, self.clock.step, dataformats='CHW')
        tb.add_image('recon_img', recon_img, self.clock.step, dataformats='CHW')
        tb.add_image('recon_img_scale', recon_img_scale, self.clock.step, dataformats='CHW')
        tb.add_image('target_amp', target_recon, self.clock.step, dataformats='CHW')
        tb.add_image('latent_map1', latent_map1, self.clock.step, dataformats='HW')
        tb.add_image('latent_map8', latent_map8, self.clock.step, dataformats='HW')

    def bgr2rgb(self, input):
        '''
        A funtion used for change the channel order for the input image.
        Since the image read by cv2.imread() have channels arranged in [b,g,r] order.
        The tensorboard visualization receive color image with channel order: [r,g,b]
        '''
        temp_input = input.clone()
        input[0,...] = temp_input[2,...]
        input[2,...] = temp_input[0,...]
        return input

    