# Video_Analysis

消防员操法打分项目，利用视频分类对消防员操法中出现的错误进行检测。

-----------------------------------------

### 实现效果概述

- 消防员一开始甩水带→显示水带未压线 
  - (检测**压线**)
- 然后挂水带奔跑→显示未掉落
  - (检测**掉落**)
- 然后爬梯→显示第一步正确，未同节，未跨节
  - (检测**首步**、**同节**、**跨节**)
- 然后拉水带→显示未过分止线
- 操练结束

检测上述黑色部分的扣分项，输出最后分类结果。

------------------------------------
### 说明
C3D为C3D的模型代码，包含训练文件和测试文件，具体详见`./C3D/README.md`.

------------------

### 进度：
- [x] C3D模型各部分训练，训练模型在他们提供后上传。
- [ ] 利用训练好的C3D模型对给定的测试视频，测试视频已经发到微信群。数据读取采用Yidadaa的LSTM读取方式。
- [ ] （如果来得及）结合光流法训练网络。
- [ ] 最低目标是7月24日之前提供一个整合方案用于对给定测试视频的预测。

### 测试脚本使用说明

#### 如何使用
进入C3D目录，运行如下命令：

```bash
python eval.py -i [要测试的视频路径] -r [存放检查点的路径]
```

测试完成后，会在当前目录生成一个`csv`文件以及视频文件，以供查看。

一个典型的适用目录长下面这个样子：
```bash
./
├── C3D
│   ├── assets/
│   ├── ckpt/ # 检查点文件存放处
│   ├── dataloaders/
│   ├── eval.py # 脚本存放处
│   ├── inference.py
│   ├── LICENSE
│   ├── mypath.py
│   ├── network/
│   ├── __pycache__/
│   ├── README.md
│   ├── test.py
│   ├── train.py
│   └── 说明.md
├── False1-cross-no-fall_output.csv # 输出文件所在地
├── False1-cross-no-fall_output.mp4
├── README.md
└── TestData # 测试文件存放处
    ├── False1-cross-no-fall.MOV
    ├── False2-fall.mp4
    ├── True-all-right-2.mp4
    ├── True-all-right.MOV
    └── True-no-fall-no-cross.mov
```
则应该使用命令：`python ./C3D/eval.py -i ./TestData/True-all-right.MOV -r ./C3D/ckpt`。


