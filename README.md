# MVP_Interpolation
Multimodal Virtual Point Interpolation
## Create Data labels
```
# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-mini" --nsweeps=10 --virtual True 
```
version="v1.0-trainval" also can be   
create train and val labels data
## Generate Virtual Points
```
python virtual_gen.py --info_path data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl  MODEL.WEIGHTS centernet2_checkpoint.pth 
```
## Train & Test
Train for centerpoint_voxelnet using one gpu
```
python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py ./configs/mvp/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_scale_virtual.py 
```
Test using one gpu
```
python ./tools/dist_test.py ./configs/mvp/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_scale_virtual.py --work_dir work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_scale_virtual --checkpoint work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_scale_virtual/latest.pth --speed_test 
```
## Results
**mini data set = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100', 'scene-0103', 'scene-0916']**   
   
**mini_val** = ['scene-0103', 'scene-0916']   
| Interpolation Method | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| No Virutal | 0.2306 | 0.5476 | 0.5076 | 0.9312 | 0.5974 | 0.3147 | 0.3254 |
| Nearest z1 | 0.3832 | 0.5067 | **0.5116** | 0.9215 | 1.1191 | **0.3615** | 0.3615 |
| Center of z1 and z2 | **0.3894** | 0.5053 | 0.5193 | 0.9097 | 0.9507 | 0.3745 | 0.3688 |
| Linear Interpolation z1 and z2 | 0.3455 | 0.5103 | 0.5214 | 0.9946 | **0.7867** | 0.3742 | 0.3540 |
| **Bilinear** | 0.3453 | 0.5076 | 0.5226 | **0.7755** | **0.8217** | 0.4044 | **0.3695** |
| Bilinear + Nearest z1 | 0.3771 | 0.5165 | 0.5162 | 0.8600 | 0.9765 | 0.3714 | 0.3645 |
| **Bilinear + Center of z1 and z2** | 0.3756 | **0.5044** | 0.5195 | 0.9162 | 0.8514 | 0.3905 | **0.3696** |

<br/><br/>
**mini_val** = ['scene-0061', 'scene-0553']   
| Interpolation Method | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| No Virutal | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
| Nearest z1 | 0.6640 | 0.3122 | 0.2431 | 0.6934 | 1.3206 | 0.5517 | 0.5519 |
| Center of z1 and z2 | 0.6695 | 0.2812 | 0.2314 | 0.4937 | 1.1579 | 0.5983 | **0.5743** |
| Tri_centroid | **0.6707** | 0.2943 | 0.2389 | 0.5422 | 1.1072 | 0.5721 | 0.5706 |
| Plane + Nearest z1 | 0.6654 | 0.2869 | 0.2317 | 0.5674 | 1.1466 | 0.5929 | 0.5648 |
| Bilinear | 0.6022 | 0.2999 | 0.2390 | 0.5890 | 1.3650 | 0.6455 | 0.5238 |
| Bilinear + Nearest z1 | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
| Bilinear + Center of z1 and z2 | 0.6581 | **0.2654** | **0.2265** | **0.6304** | **0.8880** | 0.5860 | 0.5694 |

<br/><br/>
**mini_val** = ['scene-0655', 'scene-0757']   
| Interpolation Method | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Nearest z1 | 0.5404 | 0.4203 | 0.3904 | 0.6496 | 1.1804 | 0.2914 | 0.4950 |
| Center of z1 and z2 | 0.5177 | 0.4210 | **0.3795** | **0.5105** | **0.7955** | **0.2548** | 0.5227 |
| Tri_centroid | **0.5816** | **0.4047** | 0.3809 | 0.5159 | 1.0277 | 0.3025 | **0.5304** |
| Plane + Nearest z1 | 0.5305 | 0.4011 | 0.3706 | 0.5708 | 0.8818 | 0.2907 | 0.5137 |
| Bilinear | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
| Bilinear + Nearest z1 | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
| Bilinear + Center of z1 and z2 | 0. | 0. | 0. | 0. | 0. | 0. | 0. |

<br/><br/>
**mini_val** = ['scene-0796', 'scene-1077']   
| Interpolation Method | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Nearest z1 | 0.4711 | 0.5391 | 0.4614 | 0.4477 | 2.2477 | 0.5931 | 0.4314 |
| Center of z1 and z2 | 0.4739 | 0.4842 | 0.4471 | 0.5452 | 2.2592 | 0.5541 | 0.4349 |
| Tri_centroid | **0.4767** | 0.5154 | 0.4644 | 0.4101 | 2.0647 | 0.6653 | 0.4329 |
| Bilinear | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
| Bilinear + Nearest z1 | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
| Bilinear + Center of z1 and z2 | 0. | 0. | 0. | 0. | 0. | 0. | 0. |

<br/><br/>
**mini_val** = ['scene-1094', 'scene-1100']   
| Interpolation Method | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Nearest z1 | 0.3067 | 0.6179 | 0.6022 | 0.8060 | 1.0189 | **0.4929** | **0.3015** |
| Center of z1 and z2 | 0.2853 | 0.6257 | 0.6014 | **0.7908** | **0.9941** | 0.5025 | 0.2912 |
| Tri_centroid | **0.3077** | **0.6122** | **0.6011** | 0.8136 | 1.0093 | 0.5440 | 0.2968 |
| Bilinear | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
| Bilinear + Nearest z1 | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
| Bilinear + Center of z1 and z2 | 0. | 0. | 0. | 0. | 0. | 0. | 0. |

<br/><br/>
**Overall mAP, NDS**
| Interpolation Method | mAP | NDS |
|:----------:|:----------:|:----------:|
| Nearest z1 | 0.47308 | 0.42826 |
| Center of z1 and z2 | 0. | 0.43838 |
| Tri_centroid | 0. | 0.42826 |
## Reference
https://github.com/tianweiy/MVP
