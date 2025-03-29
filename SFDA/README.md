# <p align="center"> Person Re-Identification (Person ReID)</p>

## Getting Started

### 📦 Installation

To set up the environment and baseline code, follow the installation guide provided in the original repository:

🔗 [AdversarialNAS](https://github.com/chengaopro/AdversarialNAS)


### Mask Generation

**U^2-NET** is used to generate the foreground segmentation mask.


#### 1. Install Dependencies
Set up U^2-NET: U Square Net by following the instructions here: 🔗 [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

#### 2. Generate Masks

Run the following script to create segmentation masks for human foregrounds:
```bash
python u2net_maskgenerator.py
```

## Training and Testing

All training and testing instructions are available in the `train_test_instructions.sh` file. 

### Baseline (Market-1501 with ResNet18)

Train:
```bash
python train.py --gpu_ids 0  --name use_rn18_market_1 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --total_epoch 60
```
Test:
```bash
python test.py  --gpu_ids 0  --name use_rn18_market_1 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last
```

### With Foreground-Background Augmentation

Train (with --ours parameter):
```bash
python train.py --gpu_ids 0  --name use_rn18_market_4 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --ours 0.5 --total_epoch 60
```
Test:
```bash
python test.py  --gpu_ids 0  --name use_rn18_market_4 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last
```

## 📁 Codebase

1. Baseline Code: [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) ![GitHub stars](https://img.shields.io/github/stars/layumi/Person_reID_baseline_pytorch.svg?style=flat&label=Star)
2. Masking Generation Framework: [SAM 2](https://github.com/facebookresearch/sam2) ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/sam2.svg?style=flat&label=Star)
