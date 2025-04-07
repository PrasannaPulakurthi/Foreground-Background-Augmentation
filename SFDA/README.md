# <p align="center"> Source-Free Domain Adaptation (SFDA)</p>

## Getting Started

### 📦 Installation

To set up the environment and baseline code, follow the installation guide provided in the original repository:

🔗 [AdversarialNAS](https://github.com/chengaopro/AdversarialNAS)

### Download and organize the PACS dataset

Download the PACS dataset from [Kaggle](https://www.kaggle.com/datasets/ma3ple/pacs-dataset/data) and extract it into the `datasets/` directory so that it follows the structure below:
Download the PACS dataset from [Kaggle](https://www.kaggle.com/datasets/ma3ple/pacs-dataset/data) and place it in `datasets/PACS/`.
The dataset should be organized as follows

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
```

#### 2. Generate Masks

Run the following script to create segmentation masks for human foregrounds:
```bash
python u2net_maskgenerator.py
```

## Training and Testing

Source training instructions are present in the `scripts_win/train_PACS_source.sh` file. 

Single and Multi-target training instructions are present in the `scripts_win/train_PACS_target.sh` file. 


## 📁 Codebase

1. Baseline Code: [AdversarialNAS](https://github.com/chengaopro/AdversarialNAS) ![GitHub stars](https://img.shields.io/github/stars/chengaopro/AdversarialNAS.svg?style=flat&label=Star)

