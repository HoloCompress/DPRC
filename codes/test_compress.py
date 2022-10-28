from tqdm import tqdm
from dataset import get_dataloader
from common import get_config
from agent import get_agent
from utility import ensure_dir
import numpy as np
import os
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import io as sio
from propagation_utils import crop_image 
from operator import itemgetter
from src.helpers import utils, datasets, metrics
from src.compression import compression_utils
from src.helpers import utils
import torchvision 
import time 
import scipy

def load_and_decompress(model, compressed_format_path, out_path):
    # Decompress single image from compressed format on disk

    compressed_output = compression_utils.load_compressed_format(compressed_format_path)
    start_time = time.time()
    with torch.no_grad():
        reconstruction = model.decompress(compressed_output)

    torchvision.utils.save_image(reconstruction, out_path, normalize=True)
    delta_t = time.time() - start_time
    model.logger.info('Decoding time: {:.2f} s'.format(delta_t))
    model.logger.info(f'Reconstruction saved to {out_path}')

def make_deterministic(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  
    np.random.seed(seed)
    

def scale(x, label, weight=None):  ##### x.size() should be NxCxHxW
    #alpha = torch.cuda.FloatTensor(x.size())
    alpha = x.new(x.size())

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
    return x


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


def save_img(img_array, file_name):
    img = (img_array * 255).round().astype(np.uint8)
    sio.imsave(file_name, img)


def test(test_loader, tr_agent, save_dir, postifx=''):
    psnr_list = []
    psnr_scale_list = [] 
    ssim_list = [] 
    ssim_scale_list = []
    count = 0
    #print('tr_agent.net.s is ', tr_agent.net.s)
   
    writer = open(os.path.join(save_dir, 'errors.txt'), 'a')
    loss_writer = open(os.path.join(save_dir, 'loss_analysis.txt'), 'a')
    writer.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
    loss_writer.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
    loss_all = []
    errors_all = []
    first_time_list = []
    actual_bpp_list = []
    theoretical_bpp_list = []

    encode_time_list = []
    decode_time_list = []
    
    tr_agent.net.hyper_prior.hyperprior_entropy_model.build_tables()
    tr_agent.net.to("cuda:0")
    tr_agent.save_ckpt('latest_evaluateMode.pt')


    for i, data in enumerate(test_loader):
        batch_size = data[0].size(0)
        start_id = i * batch_size
        output, losses = tr_agent.val_func(data)
        first_time_list.append(output[-5])
        encode_time_list.append(output[-4])
        decode_time_list.append(output[-3])

        for j in range(output[0].size(0)):
            index = data[-1][j].split('/')[-1]
            target_amp = data[0].cuda()
            mask = data[1].cuda()

            compress_path = os.path.join(save_dir, '%s-compressed.dprc'%(index))
            compression_output = output[2]
            actual_bpp, theoretical_bpp = compression_utils.save_compressed_format(compression_output,
                out_path=compress_path)


            target_amp = crop_image(target_amp, [880,1600], True, False)
            pred_phase = output[1][j,...]
              
            recon_img = output[-2]
            recon_img = crop_image(recon_img, [880,1600], True, False)            
            recon_img_scale = output[-1]
            recon_img_scale = crop_image(recon_img_scale, [880, 1600], True, False)

            recon_img *= (torch.sum(recon_img * target_amp, (-2, -1), keepdim=True)
                        / torch.sum(recon_img * recon_img, (-2, -1), keepdim=True))
            recon_img_scale *= (torch.sum(recon_img_scale * target_amp, (-2, -1), keepdim=True)
                        / torch.sum(recon_img_scale * recon_img_scale, (-2, -1), keepdim=True))


            target_amp = target_amp[j,...].cpu().squeeze().numpy()
            pred_phase = pred_phase.detach().cpu().squeeze().numpy()
            recon_img = recon_img[j,...].detach().cpu().squeeze().numpy()
            recon_img_scale = recon_img_scale[j,...].detach().cpu().squeeze().numpy()

            if len(recon_img.shape) == 3 and recon_img.shape[0] == 3:
                recon_img = recon_img.transpose((1,2,0))
                recon_img_scale = recon_img_scale.transpose((1,2,0))
                pred_phase = pred_phase.transpose((1,2,0))
                for c in range(3):
                    recon_img[:,:,c] = recon_img[:,:,c]
                    recon_img_scale[:,:,c] = recon_img_scale[:,:,c] 
                    
                target_amp = target_amp.transpose((1,2,0))
                multichannel = True  # for computing ssim value 
            else:
                recon_img = recon_img 
                recon_img_scale = recon_img_scale 
                multichannel = False  # for computing ssim value 
            
            print('pred_phase.dtype is ', pred_phase.dtype)
            cur_psnr_ori = psnr(target_amp, recon_img)
            cur_psnr_scale = psnr(target_amp, recon_img_scale)
            cur_ssim_ori = ssim(target_amp, recon_img, multichannel=multichannel)
            cur_ssim_scale = ssim(target_amp, recon_img_scale, multichannel=multichannel)
            psnr_list.append(cur_psnr_ori)
            psnr_scale_list.append(cur_psnr_scale)
            ssim_list.append(cur_ssim_ori)
            ssim_scale_list.append(cur_ssim_scale)

            count += 1


            recon_img = recon_img / np.max(recon_img)
            recon_img_scale = recon_img_scale / np.max(recon_img_scale)
            recon_name = os.path.join(save_dir, index + '-recon-img.png')
            recon_name_scale = os.path.join(save_dir, index  + '-recon-img-scale.png')
            phase_name = os.path.join(save_dir, index  + '-pred-phase-DPRC.png')
            invert_phase_name = os.path.join(save_dir, index + '-pred-phase-invert.png')
            target_name = os.path.join(save_dir, index  + '-target-amp.png')



            loss_all.append({'id': index, 'mse': losses['mse'].item(), 'vgg':losses['vgg'].item(), 'wfft':losses['watson-fft'].item(),
                            'total': losses['mse'].item() + losses['vgg'].item(),
                            'psnr': cur_psnr_scale, 'ssim': cur_ssim_scale , 
                            'actual_bpp':actual_bpp, 'theoretical_bpp': theoretical_bpp})

            errors_all.append({'id': index, 'psnr':cur_psnr_ori, 'psnr_scale':cur_psnr_scale, 
                            'ssim':cur_ssim_ori, 'ssim_scale':cur_ssim_scale, 
                            'actual_bpp':actual_bpp, 'theoretical_bpp': theoretical_bpp})
            print('index: %s   psnr %.4f,  ssim %.4f  ac_bpp %.4f, the_bpp: %.4f'%(index, cur_psnr_scale, cur_ssim_scale,
                                                                                                             actual_bpp, theoretical_bpp))

            actual_bpp_list.append(actual_bpp)
            theoretical_bpp_list.append(theoretical_bpp)

            scipy.io.savemat(str.replace(phase_name, '.png', '.mat'), {'phase': pred_phase})
            pred_phase = ((pred_phase + np.pi) % (2 * np.pi)) / (2 * np.pi)
            print('after transformation' , np.max(pred_phase), np.min(pred_phase))
            invert_phase = 1-pred_phase
            

            print(recon_name)
            save_img(recon_img_scale, recon_name_scale)
            save_img(pred_phase, phase_name)
            save_img(invert_phase, invert_phase_name)
       

    errors_all = sorted(errors_all, key=itemgetter('psnr_scale'), reverse=True)
    for item in errors_all:
        writer.write('Index: %s \n' % (item['id']))
        writer.write('PSNR %.4f  PSNR (scale) %.4f   SSIM %.4f  SSIM (scale) %.4f  ac_bpp: %.4f, the_bpp: %.4f \n'
                                 %(item['psnr'], item['psnr_scale'], item['ssim'], item['ssim_scale'], 
                                 item['actual_bpp'],  item['theoretical_bpp']))


    avg_psnr = sum(psnr_list) / count 
    avg_psnr_scale = sum(psnr_scale_list) / count
    avg_ssim = sum(ssim_list) / count 
    avg_ssim_scale = sum(ssim_scale_list) / count
    avg_first_time = sum(first_time_list) / count 
    avg_encode_time = sum(encode_time_list) / count 
    avg_decode_time = sum(decode_time_list) / count 
    avg_ac_bpp = sum(actual_bpp_list) / count 
    avg_the_bpp = sum(theoretical_bpp_list) / count 
    
    print('avg_psnr: ',  avg_psnr)
    print('avg_psnr scale: ', avg_psnr_scale)
    print('avg_ssim : ', avg_ssim)
    print('avg_ssim scale: ',avg_ssim_scale)
    print('avg first stage time :', avg_first_time)
    print('avg_encode time : ', avg_encode_time)
    print('avg decode time : ', avg_decode_time)
    print('avg actual bpp :', avg_ac_bpp)
    print('avg theoretical_bpp bpp : ', avg_the_bpp)
        

    writer.write('avg PSNR: %.4f \n'%(avg_psnr))
    writer.write('avg PSNR (scale): %.4f  \n'%(avg_psnr_scale))
    writer.write('avg SSIM: %.4f \n'%(avg_ssim))
    writer.write('avg SSIM (scale): %.4f \n'%(avg_ssim_scale))
    writer.write('avg first stage time %.4f\t'%(avg_first_time))
    writer.write('avg encode time %.4f\t'%(avg_encode_time))
    writer.write('avg decode time %.4f\t'%(avg_decode_time))
    writer.write('avg actual bpp %.4f\t'%(avg_ac_bpp))
    writer.write('avg theoretical bpp %.4f\t'%(avg_the_bpp))
    writer.close()


    loss_all = sorted(loss_all, key=itemgetter('total'), reverse=True)
    for item in loss_all:
        loss_writer.write('id: %s total (mse + vgg):  %.4f  vgg: %.4f  mse: %.4f wfft: %.4f, ac_bpp: %.4f, the_bpp: %.4f,  psnr: %.4f  ssim: %.4f\n'%(item['id'], 
                        item['total'], item['vgg'], item['mse'], item['wfft'], item['actual_bpp'],  item['theoretical_bpp'],
                        item['psnr'], item['ssim']))
    loss_writer.close()
    

def main():
    # create experiment config containing all hyperparameters
    config = get_config('test')
    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint
    model_path = f'../pretrain_networks/model_{config.channel}.pth'
    print(model_path)
    tr_agent.load_ckpt(load_path=model_path)

    # create dataloader
    mode = 'test'
    test_loader = get_dataloader(mode, config)
    if config.model_name != '':
        save_dir = "{}/results/{}-{}-{}-ckpt-{}-compress-{}".format(config.exp_dir, mode, config.dataset, config.model_name, config.ckpt, config.channel)
    else:
        save_dir = "{}/results/{}-{}-ckpt-{}-compress-{}".format(config.exp_dir, mode, config.dataset, config.ckpt, config.channel) 
    ensure_dir(save_dir)
    test(test_loader, tr_agent, save_dir)



if __name__ == '__main__':
    main()
