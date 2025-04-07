# <p align="center"> Foreground Background Augmentation </p>

Code for our 2025 paper "**Exploring effective synthetic data generation for reduced reliance on large amounts of labeled data**,"
by [Prasanna Reddy Pulakurthi](https://www.prasannapulakurthi.com/), [Majid Rabbani](https://www.rit.edu/directory/mxreee-majid-rabbani), [Celso M. de Melo](https://celsodemelo.net/), [Sohail A. Dianat](https://www.rit.edu/directory/sadeee-sohail-dianat), and [Raghuveer Rao](https://ieeexplore.ieee.org/author/37281258600).

**Keywords:** Synthetic Data Generation, Data Augmentation, Classification, Source-Free Domain Adaptation, Person Re-Identification

This paper introduces a novel synthetic data generation approach designed to mitigate the reliance on large-scale labeled datasets while improving model robustness and adaptability across diverse computer vision tasks, including source-free domain adaptation (SFDA) and person re-identification (ReID). Our method performs targeted data transformations by applying random noise perturbations to foreground objects and spatially shuffling background patches. This effectively increases the diversity of the training data, improving model robustness and generalization. Evaluations on the PACS dataset for SFDA demonstrate that our augmentation strategy consistently outperforms existing methods, achieving significant accuracy improvements in both single-target and multi-target adaptation settings. By generating diverse training samples, our method enables model generalization across unseen domains, providing a scalable solution for reducing reliance on manually annotated datasets. Furthermore, experiments on Market-1501 and DukeMTMC-reID datasets validate the effectiveness of our approach in person ReID, surpassing traditional augmentation techniques.

![examples](assets/examples.png)

## Method
![method](assets/method.png)

## Applications

### Source-Free Domain Adaptation
Find more information in `./SFDA/`

### Person Re-Identification
Find more information in `./Person_ReID/`
