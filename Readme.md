
# Cross Adversarial Consistency Self-Prediction Learning for Unsupervised Domain Adaptation Person Re-Identification.

This package contains the source code which is associated with the following paper:

Huafeng Li, Jian Pang, Zhengtao Yu, and Dapeng Tao, “Cross Adversarial Consistency Self-Prediction Learning for Unsupervised Domain Adaptation Person Re-Identification.” Elsevier:Information Sciences, 2020.(under review)

Edited by Jian Pang

Usage of this code is free for research purposes only. 

Thank you.

# Requirements:
    CUDA  10
    Python  3.7
    Pytorch  1.0
    torchvision  0.4
    numpy  1.17

# Get Started
## 1.Install:
    download the code
    git clone https://github.com/PangJian123/CAC-CSP.git
    cd CAC-CSP
    
## 2.Prepare datasets and the pre-trained model: please refer to https://github.com/michuanhaohao/reid-strong-baseline

## 3.Run the training file:
        python tools/train_stage_1.py (for the first stage)
        python tools/train_stage_2.py (for the second stage)

# Contact:
    Don't hesitate to contact me if you meet any problems when using this code.

    Jian Pang
    Faculty of Information Engineering and Automation
    Kunming University of Science and Technology                                                           
    Email: pangjian@stu.kust.edu.cn

# Acknowledgements
Our code is based on https://github.com/michuanhaohao/reid-strong-baseline.
