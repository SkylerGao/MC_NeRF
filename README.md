# Version_1.0:
1. This version is only intended for use with synthetic datasets.

# MC_NeRF: Multi-Camera Neural Radiance Fields for Multi-Camera Image Acquisition System
Project page: https://in2-viaun.github.io/MC-NeRF/  
arXiv preprint: https://arxiv.org/abs/2309.07846

## Abstract
Neural Radiance Fields (NeRF) employ multi-view images for 3D scene representation and have shown remarkable performance. As one of the primary sources of multi-view images, multi-camera systems encounter challenges such as varying intrinsic parameters and frequent pose changes. Most previous NeRF-based methods often assume a global unique camera and seldom consider scenarios with multiple cameras. Besides, some pose-robust methods still remain susceptible to suboptimal solutions when poses are poor initialized. In this paper, we propose MC-NeRF, a method can jointly optimize both intrinsic and extrinsic parameters for bundle-adjusting Neural Radiance Fields. Firstly, we conduct a theoretical analysis to tackle the degenerate case and coupling issue that arise from the joint optimization between intrinsic and extrinsic parameters. Secondly, based on the proposed solutions, we introduce an efficient calibration image acquisition scheme for multi-camera systems, including the design of calibration object. Lastly, we present a global end-to-end network with training sequence that enables the regression of intrinsic and extrinsic parameters, along with the rendering network. Moreover, most existing datasets are designed for unique camera, we create a new dataset that includes four different styles of multi-camera acquisition systems, allowing readers to generate custom datasets. Experiments confirm the effectiveness of our method when each image corresponds to different camera parameters. Specifically, we adopt up to 110 images with 110 different intrinsic and extrinsic parameters, to achieve 3D scene representation without providing initial poses.

## Overview
![image](https://github.com/IN2-ViAUn/MC-NeRF/blob/main/image/overview.png)


## Prerequisites
This code is developed with `python3.9.13`. PyTorch 2.0.1 and cuda 11.7 are required.  
It is recommended use `Anaconda` to set up the environment. Install the dependencies and activate the environment `mc-env` with
```
conda env create --file requirements.yaml python=3.9.13
conda activate mc-env
```

## Dataset
To play with other scenes presented in the paper, download the data [here](https://drive.google.com/drive/folders/1VKElczwt7TdWOyiWnHZIaxKYlycA-dPZ). Place the downloaded dataset according to the following directory structure(The following are created in the root directory):
```
├── config         
│   ├── ...                                                   
│                                                         
├── data                                             
│   ├── dataset_Array                                             
│   │   └── Array_Computer        
│   │   └── Array_Ficus 
│   │   └── Array_Gate
|   |   └── Array_Lego
|   |   └── Array_Materials
|   |   └── Array_Snowtruck
|   |   └── Array_Statue
|   |   └── Array_Train
|   ├── data_Ball
|   |   └── Ball_Computer
|   |   └── Ball_Ficus
|   |   └── ...
│   ├── data_HalfBall   
│   │   └── HalfBall_Computer
|   |   └── HalfBall_Ficus
|   |   └── ...
|   ├── data_Room
|   |   └── Room_Computer
|   |   └── Room_Ficus
|   |   └── ...
|   ├── ...
```
The folder synthetic_dataset_code contains a Blender script for customizing a synthetic multi-camera dataset. Readers can modify the types of objects, as well as the number and parameters of the cameras, according to their needs.
## Running the code
To train MC_NeRF(recommended for two-GPU mode):
```
# <CONFIG> <DEVICE_NUM> and <START_DEVICE> can be set to your likes 
# replace {Style} with Array | Ball | HalfBall | Room  
# replace {Dataset} with Computer | Ficus | Gate | Lego | Materials | Snowtruck | Statue | Train

# single-GPU mode
python main.py --train --config=<CONFIG> --root_data=./data/dataset_{Style} --data_name={Style}_{Dataset} --start_device=<START_DEVICE>  
eg: python main.py --train --root_data=dataset_Ball --data_name=Ball_Computer --start_device=1

# multi-GPU mode
python -m torch.distributed.launch --nproc_per_node=<DEVICE_NUM> --use_env main.py --train --config=<CONFIG> --root_data=./data/dataset_{Style} --data_name={Style}_{Dataset} --start_device=<START_DEVICE>  
eg: python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --train --root_data=dataset_Ball --data_name=Ball_Computer --start_device=1  
```
---
To test MC_NeRF(only single-GPU mode):
```
# <CONFIG> and <START_DEVICE> can be set to your likes
# replace {Style} with Array | Ball | HalfBall | Room  
# replace {Dataset} with Computer | Ficus | Gate | Lego | Materials | Snowtruck | Statue | Train

python main.py --demo --config=<CONFIG> --root_data=./data/dataset_{Style} --data_name={Style}_{Dataset} --start_device=<START_DEVICE>
eg: python main.py --demo --root_data=dataset_Ball --data_name=Ball_Computer --start_device=1
```
---
All the results will be stored in the directory `results`, all the neural network weight parameters will be stored in the directory `weights`.

If you want to save log information to log.txt file: add `--log`.  
If you want to use tensorboard tools to show training results: add `--tensorboard`

## Citation
If you find this implementation or pre-trained models helpful, please consider to cite:
```
@misc{gao2023mcnerf,
  title={MC-NeRF: Multi-Camera Neural Radiance Fields for Multi-Camera Image Acquisition Systems}, 
  author={Yu Gao and Lutong Su and Hao Liang and Yufeng Yue and Yi Yang and Mengyin Fu},
  year={2023},
  eprint={2309.07846},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
