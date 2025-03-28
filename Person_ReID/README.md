# <p align="center"> Person Re-Identification (Person ReID)</p>

## Getting Started

### 📦 Installation

To set up the environment and baseline code, follow the installation guide provided in the original repository:

🔗 [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)


### Mask Generation

**MediaPipe** is used to generate an initial mask, which serves as input to the **SAM 2** model for refining the foreground segmentation.


#### 1. Install Dependencies
Install **MediaPipe**:
  ```bash
  pip install mediapipe
  ```
Set up SAM 2 (Segment Anything in Images and Videos) by following the instructions here: 🔗 [facebookresearch/sam2](https://github.com/facebookresearch/sam2)

#### 2. Generate Masks
Run the following script to create segmentation masks for human foregrounds:
```bash
python sam2_maskgenerator.py
```

## Training and Testing 
All training and testing instructions are available in the train_test_instructions.sh file. 

Baseline (Market-1501 with ResNet18)
Train:
```bash
python train.py --gpu_ids 0  --name use_rn18_market_1 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --total_epoch 60
```
Test:
```bash
python test.py  --gpu_ids 0  --name use_rn18_market_1 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last
```

With Foreground-Background Augmentation
Train (with --ours parameter):
```bash
python train.py --gpu_ids 0  --name use_rn18_market_4 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --ours 0.5 --total_epoch 60
```
Test:
```bash
python test.py  --gpu_ids 0  --name use_rn18_market_4 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last
```

## Codebase
1. Baseline Code: [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) ![GitHub stars](https://img.shields.io/github/stars/layumi/Person_reID_baseline_pytorch.svg?style=flat&label=Star)
2. Masking Generation Framework: [SAM 2](https://github.com/facebookresearch/sam2) ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/sam2.svg?style=flat&label=Star)
