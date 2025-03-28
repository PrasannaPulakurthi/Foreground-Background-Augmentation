<h1 align="center"> Person ReID </h1>


## Getting started
### Installation
Follow the instructions at [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) for installing relavent pacages

### Mask Generation
To generate the masks, we use mediapipe to generate a rough mask and refine it using the "SAM 2: Segment Anything in Images and Videos" model. 

Instructions on how to install the SAM 2 model can be found [here](https://github.com/facebookresearch/sam2). 
Install mediapipe using the following command. 
```bash
pip install mediapipe
```

Run the following command to create a human foreground segmentation mask. 
```bash
python sam2_maskgenerator.py
```

## Training and Testing 
All the training and testing instructions are present in the train_test_instructions.sh file. 

The commands to train and test the baseline Market dataset with Resnet18. 
```bash
python train.py --gpu_ids 0  --name use_rn18_market_1 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_market_1 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last
```

The commands to train and test the baseline Market dataset with Resnet18, using our augmentation. 
```bash
python train.py --gpu_ids 0  --name use_rn18_market_4 --use_rn18 --batchsize 32  --data_dir data/Market/pytorch --ours 0.5 --total_epoch 60
python test.py  --gpu_ids 0  --name use_rn18_market_4 --use_rn18 --batchsize 32  --test_dir data/Market/pytorch --which_epoch last
```

## Codebase
1. [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) ![GitHub stars](https://img.shields.io/github/stars/layumi/Person_reID_baseline_pytorch.svg?style=flat&label=Star)
2. [SAM 2](https://github.com/facebookresearch/sam2) ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/sam2.svg?style=flat&label=Star)
