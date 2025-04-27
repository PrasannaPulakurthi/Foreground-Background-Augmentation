# <p align="center"> Foreground Background Augmentation </p>

[![arXiv](https://img.shields.io/badge/arXiv-2504.13077-b31b1b.svg)](https://arxiv.org/abs/2504.13077)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for our 2025 paper "**Effective dual-region augmentation for reduced reliance on large amounts of labeled data**,"
by [Prasanna Reddy Pulakurthi](https://www.prasannapulakurthi.com/), [Majid Rabbani](https://www.rit.edu/directory/mxreee-majid-rabbani), [Celso M. de Melo](https://celsodemelo.net/), [Sohail A. Dianat](https://www.rit.edu/directory/sadeee-sohail-dianat), and [Raghuveer Rao](https://ieeexplore.ieee.org/author/37281258600). [[PDF]](https://arxiv.org/abs/2504.13077)

**Keywords:** Data Augmentation, Classification, Source-Free Domain Adaptation, Person Re-Identification

This paper introduces a novel dual-region augmentation approach designed to reduce reliance on large-scale labeled datasets while improving model robustness and adaptability across diverse computer vision tasks, including source-free domain adaptation (SFDA) and person re-identification (ReID). Our method performs targeted data transformations by applying random noise perturbations to foreground objects and spatially shuffling background patches. This effectively increases the diversity of the training data, improving model robustness and generalization. Evaluations on the PACS dataset for SFDA demonstrate that our augmentation strategy consistently outperforms existing methods, achieving significant accuracy improvements in both single-target and multi-target adaptation settings. By augmenting training data through structured transformations, our method enables model generalization across domains, providing a scalable solution for reducing reliance on manually annotated datasets. Furthermore, experiments on Market-1501 and DukeMTMC-reID datasets validate the effectiveness of our approach for person ReID, surpassing traditional augmentation techniques.

![examples](assets/examples.png)

## Method
![method](assets/method.png)
<p align="center"><i>Overall pipeline of the proposed Foreground-Background Augmentation method.</i></p>


## Applications

### Source-Free Domain Adaptation
Implementation details and training scripts for SFDA experiments can be found in [`./SFDA/`](./SFDA/).

### Person Re-Identification
Implementation details and training scripts for ReID experiments can be found in [`./Person_ReID/`](./Person_ReID/).

## Results

Our Foreground-Background Augmentation method achieves strong performance on both Source-Free Domain Adaptation (SFDA) and Person Re-Identification (ReID) benchmarks.

### Source-Free Domain Adaptation (SFDA) - PACS Dataset

We evaluate on the PACS dataset for both single-target and multi-target domain adaptation. Our method outperforms existing source-free domain adaptation approaches.

<p align="center">
  <img src="assets/SFDA.png" alt="SFDA Results" width="800"/>
</p>

<p align="center"><i>Classification accuracy (%) for SFDA on the PACS dataset. Our method achieves the highest accuracy across single-target and multi-target settings.</i></p>

---

### Person Re-Identification (ReID) - Market-1501 and DukeMTMC-reID Datasets

We evaluate our augmentation strategy across different backbones (ResNet-18 and EfficientNet-b4) and show consistent improvements over baseline and other augmentation strategies.

<p align="center">
  <img src="assets/Person-reid.png" alt="Person ReID Results" width="800"/>
</p>

<p align="center"><i>Comparison of person ReID performance on Market-1501 and DukeMTMC-reID datasets. Our augmentation method achieves the best performance across all evaluated metrics.</i></p>


## Citation
If you find this work useful for your research, please cite:

```bibtex
@misc{pulakurthi2025effectivedualregionaugmentationreduced,
      title={Effective Dual-Region Augmentation for Reduced Reliance on Large Amounts of Labeled Data}, 
      author={Prasanna Reddy Pulakurthi and Majid Rabbani and Celso M. de Melo and Sohail A. Dianat and Raghuveer M. Rao},
      year={2025},
      eprint={2504.13077},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.13077}, 
}
