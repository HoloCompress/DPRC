import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.model_utils import Vgg19, Vgg19_conv4
from torch.autograd import Variable
import random 
import torch.nn.functional as F


class lightNess_loss(nn.Module):
    def __init__(self):
        super(lightNess_loss, self).__init__()        
        self.vgg = Vgg19_conv4().cuda()
        self.criterion = nn.MSELoss()     

    def forward(self, x, y):       
        if x.size(1) == 1:
            x = x.repeat(1,3,1,1)
        if  y.size(1) == 1:
            y = y.repeat(1,3,1,1)       
        x_vgg_conv4, y_vgg_conv4 = self.vgg(x), self.vgg(y)
        loss = self.criterion(x_vgg_conv4, y_vgg_conv4.detach())        
        return loss


class siMSELoss(nn.Module):
    def __init__(self, scale_invariant=True):
        super(siMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.scale_invariant = scale_invariant
        self.weight_normal = 1.0

    def compute_loss(self, x, label, weight=None):  ##### x.size() should be NxCxHxW
        if self.scale_invariant:
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
        

        loss = self.criterion(x,label)
        return loss

    def __call__(self, pred, target):  #### targets should contain [label, weight_map]
        '''
        label = targets[0]
        weight = targets[1]
        '''
        result_loss = self.compute_loss(pred, target)
        return result_loss


class Grad_loss(nn.Module):
    def __init__(self):
        super(Grad_loss,self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        input_padY = torch.nn.functional.pad(input, (0,0,0,1), 'constant')
        input_padX = torch.nn.functional.pad(input, (0,1,0,0), 'constant') 
      
        target_padY = torch.nn.functional.pad(target, (0,0,0,1), 'constant')
        target_padX = torch.nn.functional.pad(target, (0,1,0,0), 'constant') 

        input_padX = torch.abs(input_padX)
        input_padY = torch.abs(input_padY)
        target_padX = torch.abs(target_padX)
        target_padY = torch.abs(target_padY)

        input_diffx = (input_padX[:,:,:,1:]) / (input_padX[:,:,:,:-1] +  1e-5)
        input_diffy = (input_padY[:,:,1:,:]) / (input_padY[:,:,:-1,:] + 1e-5)
        target_diffx = (target_padX[:,:,:,1:]) / (target_padX[:,:,:,:-1] + 1e-5)
        target_diffy = (target_padY[:,:,1:,:]) / (target_padY[:,:,:-1,:] + 1e-5)

        grad_map_input = torch.norm(torch.stack([input_diffx, input_diffy], dim=1), p=2, dim=1)
        grad_map_target = torch.norm(torch.stack([target_diffx, target_diffy], dim=1), p=2, dim=1)
        grad_map_input = grad_map_input / torch.max(torch.max(grad_map_input, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        grad_map_target = grad_map_target / torch.max(torch.max(grad_map_target, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        grad_mask = torch.lt(torch.abs(grad_map_target-1), 0.1)

        loss = torch.mean(torch.mul(torch.abs(grad_map_input - grad_map_target), grad_mask), dim=(0,1,2,3))

        return loss 

    def compute_grad(self, x):
        x_diffx = x[:,:,:,1:] - x[:,:,:,:-1]
        x_diffy = x[:,:,1:,:] - x[:,:,:-1,:]

        return x_diffx, x_diffy

#### VGG Loss, borrowed from Pix2pixHD
#### https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
#####

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()   
        self.device = device
        self.vgg = Vgg19().to(self.device)
        self.criterion = nn.L1Loss()
        #self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]      

    def forward(self, x, y):   
        if x.size(1) == 1 and y.size(1) == 1:
            x = x.repeat(1,3,1,1)
            y = y.repeat(1,3,1,1) 
        if self.device != "cuda:0":
            x = x.to(self.device)
            y = y.to(self.device) 
        #print(x.device)     
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = input.new(input.size()).fill_(self.real_label)
                self.real_label_var.requires_grad=False
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = input.new(input.size()).fill_(self.fake_label)
                self.fake_label_var.requires_grad=False
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
