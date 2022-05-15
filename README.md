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
## Reference
https://github.com/tianweiy/MVP
