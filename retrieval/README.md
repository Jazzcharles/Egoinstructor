# Egoinstructor retrieval module
Official Pytorch implementation for the crossview retrieval module in Egoinstructor at CVPR 2024

> **Retrieval-Augmented Egocentric Video Captioning**<br>
> [Jilan Xu](https://jazzcharles.github.io/), [Yifei Huang](https://hyf015.github.io/), [Junlin Hou](https://houjunlin.github.io/), [Guo Chen](https://scholar.google.com/citations?user=lRj3moAAAAAJ), [Yuejie Zhang](https://cs.fudan.edu.cn/3e/d0/c25921a278224/page.htm), [Rui Feng](https://faculty.fudan.edu.cn/fengrui/zh_CN/index.htm), [Weidi Xie](https://weidixie.github.io/)<br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2024<br>

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2401.00789)
[![Project Page](https://img.shields.io/badge/Homepage-Website-green)](https://jazzcharles.github.io/Egoinstructor/)

The retrieval module is trained on pseudo paired egocentric videos (Ego4d) and exocentric videos (HowTo100M) using a EgoExoNCE loss. 

<div align="center">
<img src="../assets/retrieval.png" width="100%">
</div>



## Preparing Pretrain data
Please refer to [docs/data.md](docs/data.md).

## Training and Evaluation 
Training with slurm script
```
./scripts/train_slurm.sh
```
or run
```
python main_pretrain_contrastive.py --config ./configs/egohowto.yml
```

To evaluate the model's retrieval performance, modify the resume checkpoint path in ./configs/test.yml
```
resume: /path/to/the/trained/checkpoint.pt
```
and run
```
python main_pretrain_contrastive.py --config ./configs/test.yml
```
