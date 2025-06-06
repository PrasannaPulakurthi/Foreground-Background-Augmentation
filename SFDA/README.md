# <p align="center"> Source-Free Domain Adaptation (SFDA)</p>

## Getting Started

### Installation

To set up the environment and baseline code, follow the installation guide provided in the original repository: [AdaContrast](https://github.com/DianCh/AdaContrast)

### Download and Organize the PACS Dataset

Download the PACS dataset from [Kaggle](https://www.kaggle.com/datasets/ma3ple/pacs-dataset/data) and extract it into the `datasets/` directory.
The dataset should be organized as follows:
```text
datasets/
└── PACS/
    ├── art_painting/
    ├── cartoon/
    ├── photo/
    └── sketch/
  ```

### Mask Generation

**U^2-NET** is used to generate the foreground segmentation mask.


#### 1. Install Dependencies
Set up U^2-NET: U Square Net by following the instructions here: 🔗 [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
```bash
git clone https://github.com/xuebinqin/U-2-Net
rename U-2-Net u2net
```

Download the pre-trained model u2net.pth (176.3 MB) from [GoogleDrive](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view) to `./u2net/saved_models/u2net/`

#### 2. Generate Masks

Run the following script to create segmentation masks:
```bash
python u2net_maskgenerator.py
```

## Training and Testing

Source training instructions are present in the `scripts_win/train_PACS_source.sh` file. We also provide the pre-trained source models, which can be downloaded from [Hugging Face](https://huggingface.co/prasannareddyp/DRA-PACS/tree/main).

Single and Multi-target training instructions are present in the `scripts_win/train_PACS_target.sh` file. 


## Codebase

1. Baseline Code: [AdversarialNAS](https://github.com/chengaopro/AdversarialNAS) ![GitHub stars](https://img.shields.io/github/stars/chengaopro/AdversarialNAS.svg?style=flat&label=Star)
2. Related Work: [SPM](https://github.com/PrasannaPulakurthi/SPM) ![GitHub stars](https://img.shields.io/github/stars/PrasannaPulakurthi/SPM.svg?style=flat&label=Star)

