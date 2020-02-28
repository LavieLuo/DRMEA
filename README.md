# Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment

This is the `Pytorch` demo code for **[Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment (DRMEA) (AAAI 2020 Oral)](https://arxiv.org/abs/2002.08675)** 

## Requirements
- python 3.6
- PyTorch 1.0

## Dataset
The dataset should be placed in `./Dataset`, e.g.,

  `./Dataset/ImageCLEF`

The structure of the datasets should be like
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
