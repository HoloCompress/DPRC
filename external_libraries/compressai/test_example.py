from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
import math 
import torch 
import numpy as np 
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F
import time

from compressai.models import JointAutoregressiveHierarchicalPriors, MeanScaleHyperprior



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out



class Network(CompressionModel):
    def __init__(self, N=128):
        super().__init__(N)
        self.encode = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
        )

        self.decode = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, x):
       y = self.encode(x)
       y_hat, y_likelihoods = self.entropy_bottleneck(y)
       x_hat = self.decode(y_hat)
       return x_hat, y_likelihoods



if __name__ == '__main__':
    x = torch.rand(1, 1, 1024, 1920)
    N, _, H, W = x.size()
    num_pixels = N * H * W
    x = x.cuda()
    
    net = MeanScaleHyperprior(N=80, M=80, in_channels=1)
    net = net.cuda()

    lmbda = 0.1
    criterion = RateDistortionLoss(lmbda)
    # bitrate of the quantized latent


    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    aux_optimizer = optim.Adam(net.aux_parameters(), lr=1e-3)

    for i in range(10):
        optimizer.zero_grad()
        #bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)

        # mean square error
        output = net(x)
        losses = criterion(output, x)
        loss = losses['loss']
        bpp_loss = losses['bpp_loss']

        loss.backward(retain_graph=True)
        optimizer.step()

        print('iteration %d, loss is %.4f, bpp loss is %.4f'%(i, loss , bpp_loss))

    ####### compress ################
    
    

    net.update()
    net.eval()
    start = time.time()
    x = torch.rand(1,1, 2048, 3840)
    x = x.cuda()
    output = net.compress(x)
    strings = output['strings']
    shape = output['shape']
    #strings = net.entropy_bottleneck.compress(y)
    elapsed = time.time() - start 
    print('encoding elapsed is ', elapsed)
    #########decompress #############
    decode_start = time.time()
    #$shape = y.size()[2:]    
    x_recon = net.decompress(strings, shape)['x_hat']
    # y_hat = net.entropy_bottleneck.decompress(strings, shape)
    # print(y_hat.shape)
    # x_hat = net.decode(y_hat)
    # print(x_hat.shape)
    decode_elapsed = time.time() - decode_start
    print('decoding elapsed is ', decode_elapsed)