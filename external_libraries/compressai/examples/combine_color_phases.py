import cv2 
import numpy as np 
import os 

results_dir_list = {}
# results_dir_list['green'] = '/jixie/yujie_codes/holonet_compression/HoloNet_yujie_GREEN_5firstVGG_compression_multiScale_MSE_WFFT1e-8_VGG_2x_msssim/train_log/results/test-ckpt-30-rescale'
# results_dir_list['blue']='/jixie/yujie_codes/holonet_compression/HoloNet_yujie_BLUE_5firstVGG_compression_multiScale_MSE_WFFT1e-8_VGG_2x_msssim_again/train_log/results/test-ckpt-13-rescale'
# results_dir_list['red']='/jixie/yujie_codes/holonet_compression/HoloNet_yujie_red_5firstVGG_compression_multiScale_MSE_WFFT1e-8_VGG_2x_msssim_again/train_log/results/test-ckpt-67-rescale'


results_dir_list['blue'] = '/jixie/yujie_codes/holonet_compression/retrain_HardTanh/ours_step1_blue_hardTanh_scratch_scale0.95_correct_kpl/train_log/results/test-ckpt-48-compress-rescale-0520-b-2'
results_dir_list['green'] = '/jixie/yujie_codes/holonet_compression/retrain_HardTanh/ours_step1_green_hardTanh_scale0.95_correct_again/train_log/results/test-ckpt-59-compress-rescale-0520-g-2'
results_dir_list['red'] = '/jixie/yujie_codes/holonet_compression/retrain_HardTanh/red_step1_temp_rescale/ours_step1_red_hardTanh_scale0.95_scratch_correct_again_again/train_log/results/test-ckpt-41-compress-rescale-0520-r-2'

result_root_dir = '/jixie/yujie_codes/holonet_compression/our_final_results_0821/'
#color_img_dir = os.path.join(result_root_dir, 'color_step1_0511')

color_img_dir = '/jixie/yujie_codes/holonet_compression/our_final_results_0821/all_phase_data/ours_step1_rgb_PRN'
if not os.path.isdir(color_img_dir):
    os.mkdir(color_img_dir)
for idx in range(801, 901):
    channels = []
    for ch in ['blue', 'green', 'red']:
        cur_phase_name = f'{results_dir_list[ch]}/{idx:04d}-pred-phase.png'
        cur_img = cv2.imread(cur_phase_name, cv2.IMREAD_GRAYSCALE)
        print(cur_img.shape)
        channels.append(cur_img)
    color_img = np.stack(channels, 2)
    print(f'{color_img_dir}/{idx:04d}-phase-color-PRN.png')
    cv2.imwrite(f'{color_img_dir}/{idx:04d}-phase-color-PRN.png', color_img)
