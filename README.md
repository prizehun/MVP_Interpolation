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
**mini data set**   
mini_train = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']   
mini_val = ['scene-0103', 'scene-0916']   
| 드라마 제목 | 주연 배우 | 방영일 |
|:----------|:----------:|----------:|
| **호텔 델루나** | 이지은, 여진구 | ~~2019.07.13. ~ 2019.09.01.~~ |
| 타인은 지옥이다 | 임시완, 이동욱, 이현욱, 이정은 | 2019.08.31. ~ |
| 멜로가 체질 | 천우희, 안재홍, 전여빈, 공명 | 2019.08.09. ~ |
## Reference
https://github.com/tianweiy/MVP
