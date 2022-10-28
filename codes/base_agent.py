import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import numpy as np
from utility import TrainClock
from tensorboardX import SummaryWriter
from losses import *




def get_agent(config):
    return HologramAgent(config)


class BaseAgent(object):
    """Base trainer that provides commom training behavior.
        All trainer should be subclass of this class. 
    """
    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()
        self.device = torch.device(config.device)
        self.batch_size = config.batch_size
        self.config = config

        # build network
        self.net = self.build_net(config)

        if config.compress and config.pretrain_path != '':
            self.load_ckpt(load_path=config.pretrain_path)  # if we don't provide a pretrain path, we will train the full framework from scratch.


        
        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    @abstractmethod
    def build_net(self, config):
        raise NotImplementedError

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterion = nn.MSELoss().cuda()


    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size, gamma=0.98)


    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Checkpoint saved at {}".format(save_path))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if isinstance(self.net, nn.DataParallel):
            torch.save({
                'clock': self.clock.make_checkpoint(),
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }, save_path)
        else:
            torch.save({
                'clock': self.clock.make_checkpoint(),
                'model_state_dict': self.net.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }, save_path)
        self.net.to("cuda:0")

    def load_ckpt(self, name=None, load_path=''):
        """load checkpoint from saved checkpoint"""
        if load_path == '':
            name = name if name == 'latest' else "ckpt_epoch{}".format(name)
            load_path = os.path.join(self.model_dir, "{}.pth".format(name))
            
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)
        print("Checkpoint loaded from {}".format(load_path))
        if self.config.is_train:
            strict = True 
        else:
            strict = False
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
        if self.config.cont and self.config.is_train:  
            self.clock.restore_checkpoint(checkpoint['clock'])  
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    @abstractmethod
    def forward(self, data):
        pass

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss_list = [loss_dict[item] for item in loss_dict.keys() if 'eval' not in item]
        loss = sum(loss_list)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 


    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        self.scheduler.step(self.clock.epoch)


    def record_losses(self, loss_dict, mode='train'):
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        # record loss to tensorboard
        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)

    def train_func(self, data):
        """one step of training"""
        self.net.train()

        outputs, losses = self.forward(data)

        self.update_network(losses)
        self.record_losses(losses, 'train')

        return outputs, losses

    def val_func(self, data):
        """one step of validation"""
        self.net.eval()
        #self.net.train()

        with torch.no_grad():
            outputs, losses = self.forward(data)

        self.record_losses(losses, 'validation')

        return outputs, losses

    def visualize_batch(self, data, mode, **kwargs):
        """write visualization results to tensorboard writer"""
        raise NotImplementedError
