# DuMLP-Pin

This repo is the official implementation of "DuMLP-Pin: A Dual-MLP-dot-product Permutation-invariant Network for Set Feature Extraction" (AAAI 2022). It includes codes and models for the following tasks:

| Task | Dataset | Evaluation Metric | Value (%)
| :------: | :------: | :------: | :------: |
| Point Cloud Classification | ModelNet40 | Overall Accuracy | 92.26 |
| Point Cloud Part Segmentation | ShapeNetPart | Mean IoU | <div style="width: 80pt">S: 83.43 L: 84.92</div> |

## Getting Started

All experiments are done with one RTX 3090.

### Install

- Clone this repo.

<!--```bash
git clone https://github.com/*.git
cd DuMLP-Pin
```-->

- Create a conda virtual environment and activate it:

```bash
conda create -n dmpp python=3.8
conda activate dmpp
```

- Install `CUDA==11.1` with `cudnn8` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `pytorch`:

```bash
conda install pytorch==1.8.1 cudatoolkit=11.1
```

- Install `h5py`, `tqdm` and `tensorboard`:

```bash
conda install h5py tqdm==4.59.0 tensorboard==2.4.0
```

### Quick Start

We provide several examples in `my_script.sh`, which includes some commands to reproduce our results in the paper. Please fill in the value of `MODEL_LOAD_PATH` for your usage.

The template is:

```
./my_script.sh [MODE] [TASK] [GPU_INDEX] [S=small, L=large, optional for ShapeNetPart]
```

Some examples:
```
chmod +x my_script.sh
./my_script.sh train ModelNet40 0
./my_script.sh train ShapeNetPart 0 S
./my_script.sh eval ShapeNetPart -1 L # -1 for CPU
```

### Structure

Models and datasets are grouped as follows. For ShapeNetPart, we have two models of different sizes: small in `S` and large in `L`. We do each single experiment twice to validate the reproducibility.
```
 DuMLP_Pin
    ├─ModelNet40
    │  ├─datasets
    │  └─models
    │      ├─2021-10-09-18-58-28
    │      └─2021-10-09-20-57-57
    └─ShapeNetPart
        ├─datasets
        └─models
            ├─L
            │  ├─2021-10-17-11-29-54
            │  └─2021-10-18-15-33-17
            └─S
                ├─2021-10-18-15-34-37
                └─2021-10-19-04-49-45
```

### Citation

```
@article{Fei_Zhu_Liu_Deng_Li_Deng_Zhang_2022,
  title={DuMLP-Pin: A Dual-MLP-Dot-Product Permutation-Invariant Network for Set Feature Extraction},
  volume={36},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/19939},
  DOI={10.1609/aaai.v36i1.19939},
  number={1},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Fei, Jiajun and Zhu, Ziyu and Liu, Wenlei and Deng, Zhidong and Li, Mingyang and Deng, Huanjun and Zhang, Shuo},
  year={2022},
  month={Jun.},
  pages={598-606}
}
```

### Contact

Feel free to contact Jiajun Fei [feijj20@mails.tsinghua.edu.cn](feijj20@mails.tsinghua.edu.cn) if you have some questions.