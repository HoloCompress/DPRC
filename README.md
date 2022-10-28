 ## Joint Neural Phase Retrieval and Compression for Energy- and Computation-Efficient Holography on the Edge

![Python](https://img.shields.io/badge/Python->=3.6-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch-==1.6.0-Red?logo=pytorch)

This repository contains the source codes for the paper [Joint Neural Phase Retrieval and Compression for Energy- and Computation-Efficient Holography on the Edge](https://dl.acm.org/doi/10.1145/3528223.3530070)

## Prepare the environment
To train/test with the codes, please first prepare the conda environment first.
#### Prepare the Python environment:
     
1) Install anaconda on your machine. (For example, you can install [MiniConda](https://docs.conda.io/en/latest/miniconda.html).) 

2) After installing conda, run the following commands in the terminal:
        
    ```
    conda env create -f environment_dprc.yaml  
    conda activate dprc
    ```
3) Install the library [compressAI](https://github.com/InterDigitalInc/CompressAI) with our modifications for entropy coding and entropy decoding.
    ```
    cd external_libraries/compressai
    python setup.py install 
    ```
    

## Test on your own data:
If you want to run tests on the images collected by yourself, just put the images in ```./data/collected```, then open a terminal within current folder and run the following command:

```
cd codes; sh scripts/compress_tests.sh
```
After running, you'll get the test results in folder ```./running/DPRC/results```. And the results for 'R,G,B' channels are stored in three different folders. 


## Training

Please dowload DIV2K dataset and put it under ```./data``` folder, then the final training data path is ```./data/DIV2K_train_HR/```. Note that the framework needs to be trained on GPUs with more than 15 GB memory when the batch size is set to 1.

#### Trigger the training by running the scripts:
1) Stage1:
    ```
    cd codes; sh scripts/train_stage1.sh
    ```
2) Stage2: Train the full framework (DPRC):
    ```
    cd codes; sh scripts/train_stage2.sh
    ```
We have provided a few data samples in ```./data``` for a quick start when the full dataset is not ready yet.


## Some parameters
In the training scripts, we can adjust the parameters by specifying:
*  ```-g '1'```:  specify the index of gpu on which the codes will be run, e.g, we specify '1' here.
* ```--dataset```: specify which dataset to train/test on, use '--dataset DIV2K' for train/test on DIV2K dataet and use '--dataset collected' to test on self-collected images.
* ```--pixel_pitch 6.4```: this will make the model used a pixel pitch of 6.4 um.
* ```--prop_dist 20```: specify a reconstruction distance at 20 cm. You can change propagation distance according to your needs.
* ```--pretrain_path```: since we adopt a two-stage training, it is used to specify the location of the pretrained model file produced by stage 1. 
* ```--vis```: if specified, we'll visualize the output images in tensorboard during training.
* ```--batch_size 1```: the batch size used for training or test.
* ```--lr 1e-4```: set the learning rate used for training the framework.
* ```--channel r```: specify the channel of training image and corresponding wavelength, the supported choices include 'r', 'g' and 'b'.
* ```--compress```: this must be specified during training and test in stage2, which will let the network integrate compression related modules.



## Bibtex
We are happy to be cited when our codes are useful in your projects.
```
@article{DPRC,
author = {Wang, Yujie and Chakravarthula, Praneeth and Sun, Qi and Chen, Baoquan},
title = {Joint Neural Phase Retrieval and Compression for Energy- and Computation-Efficient Holography on the Edge},
year = {2022},
volume = {41},
number = {4},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {110},
numpages = {16}
}
```

## Acknowledgement
Some codes are adapted from [neural-holography](https://github.com/computational-imaging/neural-holography) by @ [Suyeon Choi](https://github.com/choisuyeon) and [HiFiC](https://github.com/Justin-Tan/high-fidelity-generative-compression) by @ [Justin-Tan](
https://github.com/Justin-Tan), we thank the authors for opensourcing their awesome resources.

