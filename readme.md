# STCLoc
本项目仅用于复现代码记录使用 

## Environment

环境同SGLoc 和DiffLoc 不用重新配置

## Train  / Test 过程中  Hercules要改的数据集
参考posepn++的基础上 加了一些东西


## Run

### Hercules
- 多模态数据集 单独测radar/lidar部分
#### train  -- 1 GPU

```
python train_hercules_lidar.py //  python train_hercules_radar.py 
```
**其他运行训练的时候在代码里要改的**
- --gpu_id 0 对应代码里os.environ["CUDA_VISIBLE_DEVICES"] = '3'
- --全局变量 **SEQUENCE_NAME**
- 序列名 `data.hercules_lidar.py` or `data.hercules_radar.py`   里面的`sequence_name`
- `data.composition` 里面的要把引入的 `radar / lidar` 版本的 `Hercules类` 进行切换

**以下在代码里默认设置好的（参考原本代码run oxford数据集的参数）**
- --decay_step 500
- --log_dir 文件后缀记得更改`f'STCLoc_{SEQUENCE_NAME}_Lidar/'` or `f'STCLoc_{SEQUENCE_NAME}_Radar/'`
- --batch_size 80  注意更改
- --val_batch_size 80  注意更改
- --skip 2
- --dataset Hercules 
- --num_loc 10 --num_ang 10 
- --mac epoch 用100轮

-------


#### test  -- 1 GPU
```
python eval_hercules_lidar.py //   python eval_hercules_radar.py
```
**其他运行测试的时候在代码里要改的**
- --gpu_id 0 对应代码里os.environ["CUDA_VISIBLE_DEVICES"] = '3'
- --resume_model checkpoint_epochxx.tar 权重文件
- --全局变量 **SEQUENCE_NAME**
- 序列名`data.hercules_lidar.py` or `data.hercules_radar.py`   里面的`sequence_name`
- `data.composition` 里面的要把引入的 `radar / lidar` 版本的 `Hercules类` 进行切换


**以下在代码里默认设置好的（参考原本代码run oxford数据集的参数）**
- --gpu_id 0 代码里os.environ["CUDA_VISIBLE_DEVICES"] = '3'
- --resume_model checkpoint_epochxx.tar
- --log_dir `f'STCLoc_{SEQUENCE_NAME}_Lidar/'` or `f'STCLoc_{SEQUENCE_NAME}_Radar/'`
- --val_batch_size 40 **注意这个值等于1 会测试的很慢**
- --dataset Hercules
- --skip 2
- --num_loc 10 --num_ang 10 

--------
### Snail_Radar



当前数据集各种代码还没改
-----
- train  -- 1 GPU
```
python train.py --gpu_id 0 --batch_size 80 --val_batch_size 80 --decay_step 500 --log_dir log-oxford/ --dataset Oxford --num_loc 10 --num_ang 10 --skip 2
 ```
- test  -- 1 GPU
```
python eval.py --gpu_id 0 --val_batch_size 40 --log_dir log-oxford/ --dataset Oxford --num_loc 10 --num_ang 10 --skip 2 --resume_model checkpoint_epochxx.tar
```

## Acknowledgement

We appreciate the code of PointNet++ and AtLoc they shared.

## Citation

```
@ARTICLE{9928031,
  author={Yu, Shangshu and Wang, Cheng and Lin, Yitai and Wen, Chenglu and Cheng, Ming and Hu, Guosheng},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={STCLoc: Deep LiDAR Localization With Spatio-Temporal Constraints}, 
  year={2023},
  volume={24},
  number={1},
  pages={489-500},
  doi={10.1109/TITS.2022.3213311}}
```
