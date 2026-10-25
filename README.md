## [TMLR: 2026] Precision by Plasticity: Exemplar-free Analytic Adaptation for Continual Learning
We provide the complete codebase and pretrained models used to benchmark continual learning performance on three popular benchmark datasets for continual learning.

## Download Datasets
We conduct our experiments on the following datasets:
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet-R](https://github.com/hendrycks/imagenet-r)
- [Tiny-ImageNet](https://www.kaggle.com/datasets/ambityga/tiny-imagenet)
  
Place the dataset file inside the ```local_datasets``` directory

## Download the Pretrained Backbone
We use a **DeiT ViT** backbone pretrained on 611 ImageNet classes (excluding 389 overlapping classes with CIFAR and Tiny-ImageNet to avoid data leakage) for all experiments.
Please download the pre-trained deit vit network provided by Learnability and Algorithm for Continual Learning from
The pretrained model is sourced from [CLOOD](https://github.com/k-gyuhak/CLOOD)

Save the *.pth file to ./models/best_checkpoint.pth path.

Refer: Learnability and Algorithm for Continual Learning. ICML, 2023
**[Download DeiT ViT checkpoint.pth](https://drive.google.com/file/d/1uEpqe6xo--8jdpOgR_YX3JqTHqj34oX6/view?usp=sharing)**

### System Configuration
```
Python 3.12.3
Ubuntu 24.04.1 LTS
CUDA Version: 12.4
```
### Train Model
```
Tested Model: [vit,resnet18,resnet34,resnet50,efficientnet_b0,vitB16]
Tested Datset: [cifar100,tinyimagenet,imagenet-r,imagenet1000]
Mode Type: [direct or woodbury]

======== Run AACL ================
#python aacl.py --model_name <name> --dataset <name> --seeds <value> --eta <value> --beta <value> --buffer_size <value> --gamma_main <value> --mode_type <value>
```

## Citation
* Please cite this work if you find it useful for your research.
```
```

## Acknowledgement
This repo is built upon the following project. We sincerely appreciate for their contributions.
1. [GACL](https://github.com/CHEN-YIZHU/GACL)  

