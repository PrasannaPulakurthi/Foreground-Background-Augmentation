<h1 align="center"> Person ReID </h1>

To generate the masks, we use mediapipe to generate a rough mask and refine it using the "SAM 2: Segment Anything in Images and Videos" model. 

Instructions on how to install the SAM 2 model can be found here. 

Run the following command to create a human foreground segmentation mask. 
```bash
python sam2_maskgenerator.py
```
## Codebase
1. [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) ![GitHub stars](https://img.shields.io/github/stars/layumi/Person_reID_baseline_pytorch.svg?style=flat&label=Star)
2. [SAM 2](https://github.com/facebookresearch/sam2) ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/sam2.svg?style=flat&label=Star)
