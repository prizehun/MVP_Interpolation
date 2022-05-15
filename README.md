# MVP_interpolation
Multi modal virtual point interpolation
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
| Nearest z1 | 0.3832 | 0.5067 | **0.5116** | 0.9215 | 1.1191 | **0.3615** | 0.3615 |
| Center of z1 and z2 | **0.3894** | **0.5053** | 0.5193 | 0.9097 | 0.9507 | 0.3745 | 0.3688 |
| Linear Interpolation z1 and z2 | 0.3455 | 0.5103 | 0.5214 | 0.9946 | 0.7867 | 0.3742 | 0.3540 |
| Bilinear | 0.3453 | 0.5076 | 0.5226 | 0.7755 | **0.8217** | 0.4044 | **0.3695** |
| Bilinear + z1 | 0.3771 | 0.5165 | 0.5162 | 0.8600 | 0.9765 | 0.3714 | 0.3645 |
| Bilinear + Center of z1 and z2 | 0. | 0. | 0. | 0. | 0. | 0. | 0. |

<br/><br/>
**mini_val** = ['scene-0061', 'scene-0553']   
| Interpolation Method | mAP | mATE | mASE | mAOE | mAVE | mAAE | NDS |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| Nearest z1 | 0.3832 | 0.5067 | 0.5116 | 0.9215 | 1.1191 | 0.3615 | 0.3615 |
| Center of z1 and z2 | 0.3894 | 0.5053 | 0.5193 | 0.9097 | 0.9507 | 0.3745 | 0.3688 |
| Linear Interpolation z1 and z2 | 0.3455 | 0.5103 | 0.5214 | 0.9946 | 0.7867 | 0.3742 | 0.3540 |
| Bilinear | 0.3453 | 0.5076 | 0.5226 | 0.7755 | 0.8217 | 0.4044 | 0.3695 |
| Bilinear + z1 | 0.3771 | 0.5165 | 0.5162 | 0.8600 | 0.9765 | 0.3714 | 0.3645 |
| Bilinear + Center of z1 and z2 | 0. | 0. | 0. | 0. | 0. | 0. | 0. |
## Reference
https://github.com/tianweiy/MVP
