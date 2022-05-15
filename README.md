# MVP_interpolation
Multi modal virtual point interpolation
## Train & Test
```
python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py ./configs/mvp/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_scale_virtual.py 
```

## Reference
https://github.com/tianweiy/MVP
