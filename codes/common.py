import os
from utility import ensure_dirs
import argparse
import json
import shutil
import torch

def get_config(phase):
    config = Config(phase)
    return config


class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """
    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
        wave_dict = {'b':450*nm, 'g':520 * nm, 'r':638*nm} # pre-defined wavelength for each color channel

        self.chan_strs = ('red', 'green', 'blue')
        self.prop_dist = self.prop_dist * cm
        self.feature_size = (args.pixel_pitch * um, args.pixel_pitch * um) 
        self.wavelength = wave_dict[self.channel]  # select the wavelength used for wave propagation calculation

        ''' add compression training config '''
        self.lambda_schedule = dict(vals=[1, 0.5], steps=[10000]) 
        self.lambda_A = 2**(-4)
        self.lambda_B = 2**(-5)
        self.target_rate = 2.
        self.target_schedule = dict(vals=[0.20/0.14, 1.], steps=[10000])  # Rate allowance

        if not self.compress:
            self.w_recon = 1.0
            self.model_name = 'stage1'
        else:
            self.w_recon = {'high':10, 'mid':5, 'low':1}[self.quality]
            self.model_name = 'stage2'

        # experiment log/model paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if self.model_name == '':
            self.log_dir = os.path.join(self.exp_dir, 'log')
            self.model_dir = os.path.join(self.exp_dir, 'model')
        else:
            self.log_dir = os.path.join(self.exp_dir, 'log_{}'.format(self.model_name))
            self.model_dir = os.path.join(self.exp_dir, 'model_{}'.format(self.model_name))


        print("----Experiment Configuration-----")
        for k, v in self.__dict__.items():
            print("{0:20}".format(k), v)
            
        if phase == "train" and args.cont is not True and os.path.exists(self.log_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.log_dir)
            shutil.rmtree(self.model_dir)
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
            self.device = 'cuda:0'

        # create soft link to experiment log directory
        if not os.path.exists('run_log'):
            os.symlink(self.exp_dir, 'run_log')
        
            
        # save this configuration
        if self.is_train:
            if self.model_name != '':
                log_file = '{}/config_{}.txt'.format(self.exp_dir, self.model_name)
            else:
                log_file = '{}/config.txt'.format(self.exp_dir)
            with open(log_file, 'w') as f:
                json.dump(self.__dict__, f, indent=2)


    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        
        # basic configuration
        self._add_basic_config_(parser)

        # dataset configuration
        self._add_dataset_config_(parser)

        # display parameter configuration
        self._add_display_config_(parser)

        # model configuration
        self._add_network_config_(parser)

        # loss weight configuration
        self._add_loss_config_(parser)

        # training or testing configuration
        self._add_training_config_(parser)
        
        self._add_test_config_(parser)

        # additional parameters if needed
        pass

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="../running", help="path to project folder where models and logs will be saved")
        group.add_argument('--holo_data_root', type=str, default='../data')
        group.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-2], help="name of this experiment")
        group.add_argument('-g', '--gpu_ids', type=str, default='2,3', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    def _add_dataset_config_(self, parser):
        """add hyperparameters for dataset configuration"""        
        group = parser.add_argument_group('dataset')
        group.add_argument('--dataset', type=str, default='DIV2K', choices=['DIV2K', 'collected'], 
                          help='which dataset to train or test on, DIV2K for DIV2K dataset or (collected) for self-collected test images')
        group.add_argument('--batch_size', type=int, default=1, help="batch size")
        group.add_argument('--num_workers', type=int, default=4, help="number of workers for data loading")
        group.add_argument('--channel', type=str, choices=['r','g','b'], help="select which channel to train")

    def _add_display_config_(self, parser):
        """add parameters for displaying settings"""
        group = parser.add_argument_group('display')
        group.add_argument('--prop_dist', type=int, default=20, help="the propagation distance in cm")
        group.add_argument('--pixel_pitch', type=float, default=6.4, help='pixel pitch for SLM')
        
    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument('--pretrain_path', type=str, default='')
        group.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        group.add_argument('--compress', action='store_true', default=False, help='whether to integrate compression modules into the framework')

    def _add_loss_config_(self, parser):
        """loss balancing parameters"""
        group = parser.add_argument_group('loss')
        group.add_argument('--w_mse', type=float, default=1.0, help='the weight parameter for MSE loss term')
        group.add_argument('--w_vgg', type=float, default=0.025, help='the balancing weight for VGG perceptual loss')
        group.add_argument('--w_ssim', type=float, default=0.05, help=' the balancing weight for MS-SSIM loss')
        group.add_argument('--w_wfft', type=float, default=1e-8, help='the balancing weihgt for Watson-FFT loss')
        group.add_argument('--quality', type=str,  default='high', choices=['high', 'mid', 'low'], help='the compression quality level set for stage2\'s training')
    
    
    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--nr_epochs', type=int, default=1000, help="total number of epochs to train")
        group.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        group.add_argument('--lr_s', type=float, default=5e-5, help="initial learning rate")
        group.add_argument('--lr_step_size', type=int, default=5, help="step size for learning rate decay")
        group.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        group.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        group.add_argument('--save_frequency', type=int, default=1, help="save models every x epochs")
        group.add_argument('--val_frequency', type=int, default=5, help="run validation every x iterations")
        group.add_argument('--vis_frequency', type=int, default=40, help="visualize output every x iterations")
        group.add_argument('--model_name', type=str, default='', help='specify a mdoel name for save the model and log')
        
        
    def _add_test_config_(self, parser):
        """testing configuration"""
        group = parser.add_argument_group('testing')
        group.add_argument('-o', '--output', type=str, default='output folder to save results')
        group.add_argument('--postfix', type=str, default='', help='postfix for name the output')

