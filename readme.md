# STCLoc
本项目仅用于复现代码记录使用 

## Environment

环境同SGLoc 和DiffLoc 不用重新配置

## Train  / Test  代码要修改的部分 以兼容HerCULES 和 Snail_Radar
参考`posepn++`的代码基础上https://github.com/herry0415/PosePN-series 加了一些东西--num_loc 10 --num_ang 10  可以分为四个部分进行修改
- data.hercules_lidar.py // data.hercules_radar.py 数据集类加载 
- data.composition 部分 
- train_hercules_lidar.py //  python train_hercules_radar.py 训练文件
- python eval_hercules_lidar.py //   python eval_hercules_radar.py 测试文件



### hercules_lidar.py //  hercules_radar.py
----
- 先把posepn++项目的数据集 文件复制一下然后对着该项目的`OxfordVelodyne_datagenerator.py` 函数的todo部分改
- 数据集需要加几个参数  分区和分角度
- 返回值由2个变为4个了


### composition
----
- MF部分 `offsets = offsets.astype(int) `  1 `np.int` 已启用 改一下
- *核心部分*： 加载`Hercules`部分 要注意lidar or radar的区分 头文件


### train_hercules_lidar.py //  train_hercules_radar.py
----
- 先把该项目的`train.py` 文件复制一下 然后对着posepn++ train函数的todo部分改
- `pcs_tensor                           = val_data.to(device, dtype=torch.float32) ` Radar 部分有问题 需要转换为torch.float32

### eval_hercules_lidar.py //   eval_hercules_radar.py
----
- 先把该项目的`test.py` 文件复制一下 然后对着posepn++ eval函数的todo部分改
- 注意图片保存名字的后缀切换 `lidar / radar`  ctrl+f一下 lidar/radar

**改完代码注意 注意检查**：
- SEQUENCE_NAME 是否都正确加载了
- lidar / radar 数据集路径/保存名称的 后缀是否更改完毕ctrl+f一下 lidar/radar

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
- --序列名 `data.hercules_lidar.py` or `data.hercules_radar.py`   里面的`sequence_name`
- `data.composition` 里面的要把引入的 `radar / lidar` 版本的 `Hercules类` 进行切换

**以下在代码里默认设置好的（参考原本代码run oxford数据集的参数）**
- --decay_step 500
- --log_dir 文件后缀记得更改`f'STCLoc_{SEQUENCE_NAME}_Lidar/'` or `f'STCLoc_{SEQUENCE_NAME}_Radar/'`
- --batch_size 80  注意更改
- --val_batch_size 80  注意更改
- --skip 2
- --dataset Hercules 
- --num_loc 10 --num_ang 10 
- --mac epoch **用100轮**

-------


#### test  -- 1 GPU
```
python eval_hercules_lidar.py //   python eval_hercules_radar.py
```
**其他运行测试的时候在代码里要改的**
- --gpu_id 0 对应代码里`os.environ["CUDA_VISIBLE_DEVICES"] = '3'`
- --resume_model `checkpoint_epochxx.tar` 权重文件
- --全局变量 **SEQUENCE_NAME**
- --序列名`data.hercules_lidar.py` or `data.hercules_radar.py`   里面的`sequence_name`
- `data.composition` 里面的要把引入的 `radar / lidar` 版本的 `Hercules类` 进行切换


**以下在代码里默认设置好的（参考原本代码run oxford数据集的参数）**
- --log_dir `f'STCLoc_{SEQUENCE_NAME}_Lidar/'` or `f'STCLoc_{SEQUENCE_NAME}_Radar/'`
- --val_batch_size 40 **注意这个值等于1 会测试的很慢**
- --dataset `Hercules``
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
