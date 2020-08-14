# Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment

This is the `Pytorch` demo code for **[Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment (DRMEA) (AAAI 2020)](https://www.aaai.org/ojs/index.php/AAAI/article/view/5943)** 

## Overview

*"DRMEA describes the domains by a sequence of abstract manifolds, and develops a Riemannian manifold learning framework to achieve transferability and discriminability consistently. "*

#### Network Architectures
![NetworkArchitectures](https://github.com/LavieLuo/Datasets/blob/master/NetworkArchitectures_AAAI.png)

#### Experiment Result
ImageCLEF       |    I→P     |    P→I     |    I→C     |    C→I     |    C→P     |    P→C     | Avg. 
:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:
| ResNet-50     | 74.8 ± 0.3 | 83.9 ± 0.1 | 91.5 ± 0.3 | 78.0 ± 0.2 | 65.5 ± 0.3 | 91.2 ± 0.3 | 80.7
| DAN           | 74.5 ± 0.4 | 82.2 ± 0.2 | 92.8 ± 0.2 | 86.3 ± 0.4 | 69.2 ± 0.4 | 89.8 ± 0.4 | 82.5
| DANN          | 75.0 ± 0.3 | 86.0 ± 0.3 | 96.2 ± 0.4 | 87.0 ± 0.5 | 74.3 ± 0.5 | 91.5 ± 0.6 | 85.0
| JAN           | 76.8 ± 0.4 | 88.0 ± 0.2 | 94.7 ± 0.2 | 89.5 ± 0.3 | 74.2 ± 0.3 | 91.7 ± 0.3 | 85.8
| CDAN          | 76.7 ± 0.3 | 90.6 ± 0.3 | 97.0 ± 0.4 | 90.5 ± 0.4 | 74.5 ± 0.3 | 93.5 ± 0.4 | 87.1
| CDAN+E        | 77.7 ± 0.3 | 90.7 ± 0.2 | 97.7 ± 0.3 | 91.3 ± 0.3 | 74.2 ± 0.2 | 94.3 ± 0.3 | 87.7
| **DRMEA (No AL)** | 78.0 ± 0.1 | 91.1 ± 0.1 | 95.6 ± 0.2 | 88.7 ± 0.3 | 74.8 ± 0.1 | 94.8 ± 0.2 | 87.3
| **DRMEA (No DS)** | 78.9 ± 0.1 | 90.5 ± 0.2 | 94.0 ± 0.1 | 87.8 ± 0.1 | 76.7 ± 0.2 | 93.0 ± 0.1 | 86.8
| **DRMEA**         | 80.7 ± 0.1 | 92.5 ± 0.1 | 97.2 ± 0.1 | 90.5 ± 0.1 | 77.7 ± 0.2 | 96.2 ± 0.2 | 89.1

## Requirements
- python 3.6
- PyTorch 1.0

## Dataset
- The dataset should be placed in `./Dataset`, e.g.,

  `./Dataset/ImageCLEF`

- The structure of the datasets should be like
```
Image-CLEF (Dataset)
|- I (Domain)
|  |- aeroplane (Class)
|     |- XXXX.jpg (Sample) 
|     |- ...
|  |- bike (Class)
|  |- ...
|- P (Domain)
|- C (Domain)
```

## Usage
- Download the `Image-CLEF` dataset from **[Google Drive](https://drive.google.com/file/d/1_-XuTxmmGH3ayDIgPBzdaq8EpeLH2gvp/view?usp=sharing)**

- Training with config

  `python main.py --dset ImageCLEF --mEpo 50 --ExpTime 10 --BatchSize 32`
  
- Experiment results refer to *Variables*: 

  **ACC_Recorder** and **Total_Result**

- Best model and epxerimental logs can be found in folder `./Model_Log/...`

## Citation
If this repository is helpful for you, please cite our paper:
```
@inproceedings{luo2020unsupervised,
  title={Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment},
  author={You-Wei Luo, and Chuan-Xian Ren, and Pengfei Ge, and Ke-kun Huang, and Yu-Feng Yu},
  booktitle={AAAI},
  year={2020}
}
```

## Contact
If you have any questions, please feel free contact me via **luoyw28@mail2.sysu.edu.cn**.
