# YOLOv12-BoT-SORT-ReID 多无人机跟踪系统

## 📖 项目简介

本项目是一个完整的**多无人机目标检测与跟踪系统**，基于最新的 **YOLOv12** 目标检测模型和 **BoT-SORT**（Bottleneck Transformer SORT）多目标跟踪算法，结合 **Fast-ReID** 重识别技术，实现对无人机集群（UAV Swarm）的高精度检测、跟踪与识别。

### 🎯 核心特性

- ✅ **YOLOv12 目标检测**：最新的 YOLO 系列模型，提供高精度实时检测
- ✅ **BoT-SORT 跟踪算法**：结合相机运动补偿（GMC）的先进多目标跟踪
- ✅ **Fast-ReID 重识别**：基于深度学习的目标重识别，提高跟踪鲁棒性
- ✅ **多数据格式支持**：支持 LabelMe、UAVSwarm、MOT Challenge 等多种数据格式
- ✅ **完整训练流程**：从数据准备到模型训练、推理、评估的全流程支持
- ✅ **批量处理能力**：支持批量视频处理和多场景跟踪
- ✅ **轨迹插值优化**：提供轨迹插值算法，优化跟踪连续性

---

## 📁 项目结构

```
YOLOv12-BoT-SORT-ReID/
├── convert_labelme_to_yolo.py      # LabelMe → YOLO 格式转换
├── convert_uavswarm_to_yolo.py     # UAVSwarm → YOLO 格式转换
├── evaluate_detections.py          # 检测结果评估脚本
├── test_uavswarm.py                # 图像序列检测测试
├── BoT-SORT/                       # BoT-SORT 跟踪系统核心模块
│   ├── tracker/                    # 跟踪器实现
│   │   ├── bot_sort.py            # BoT-SORT 主算法
│   │   ├── mc_bot_sort.py         # 多摄像头 BoT-SORT
│   │   ├── kalman_filter.py       # 卡尔曼滤波器
│   │   ├── gmc.py                 # 全局运动补偿
│   │   └── matching.py            # 数据关联匹配
│   ├── fast_reid/                  # Fast-ReID 重识别模块
│   │   ├── fast_reid_interfece.py # ReID 接口
│   │   └── fastreid/              # ReID 模型实现
│   ├── yolov12/                    # YOLOv12 检测模型
│   │   ├── train.py               # 模型训练脚本
│   │   ├── ultralytics/           # Ultralytics 库
│   │   ├── weights/               # 预训练权重
│   │   └── *.yaml                 # 数据集配置文件
│   ├── tools/                      # 工具脚本集合
│   │   ├── inference.py           # 视频推理与跟踪
│   │   ├── track.py               # MOT 评估跟踪
│   │   ├── demo.py                # 演示脚本
│   │   ├── interpolation.py       # 轨迹插值
│   │   ├── predict_track*.py      # 多赛道预测脚本
│   │   └── mota.py                # MOTA 指标计算
│   ├── batch_process_videos.py    # 批量视频处理
│   └── getInfo.py                 # 数据集统计分析
├── data/                           # 数据集目录
│   ├── images/                    # 原始图像数据
│   ├── uav_custom/                # 自定义 UAV 数据集
│   ├── uavswarm_yolo/             # UAVSwarm YOLO 格式
│   ├── MultiUAV_Train/            # 多 UAV 训练数据
│   ├── MultiUAV_Test/             # 多 UAV 测试数据
│   └── MOT/                       # MOT 格式数据
├── test_results/                   # 测试结果输出
└── TrackEval/                      # 跟踪评估工具
    ├── trackeval/                 # 评估指标实现
    └── scripts/                   # 评估脚本

```

---

## 🛠️ 环境安装

### 系统要求

- **操作系统**：Windows 10/11, Linux (Ubuntu 18.04+)
- **Python**：3.8 - 3.11（推荐 3.11）
- **GPU**：NVIDIA GPU with CUDA 11.0+ (推荐 RTX 3060 及以上)
- **内存**：16GB RAM 及以上
- **硬盘**：20GB 可用空间

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-repo/YOLOv12-BoT-SORT-ReID.git
cd YOLOv12-BoT-SORT-ReID
```

2. **创建虚拟环境**（推荐）
```bash
conda create -n uav python=3.11
conda activate uav
```

3. **安装依赖包**
```bash
cd BoT-SORT
pip install -r requirements.txt
```

### 核心依赖库

```
ultralytics          # YOLOv12 官方库
torch>=2.0.0         # PyTorch 深度学习框架
torchvision>=0.15.0  # 视觉库
opencv-python        # 图像处理
numpy                # 数值计算
scipy                # 科学计算
filterpy             # 卡尔曼滤波
lap                  # 线性分配问题求解器
motmetrics           # MOT 评估指标
loguru               # 日志记录
tqdm                 # 进度条显示
scikit-learn         # 机器学习工具
matplotlib           # 可视化
Pillow               # 图像处理
easydict             # 配置管理
pyyaml               # YAML 解析
```

---

## 📂 数据准备

### 1. LabelMe 标注数据转换

**脚本**：`convert_labelme_to_yolo.py`

**功能说明**：
- 将 LabelMe 工具标注的 JSON 文件转换为 YOLO 训练格式
- 自动遍历多级目录，收集所有标注数据
- 随机划分训练集（70%）和验证集（30%）
- 生成标准 YOLO 目录结构

**使用方法**：

1. 修改脚本中的路径配置：
```python
base_dir = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\images"
output_images_train = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\images\train"
output_images_val = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\images\val"
output_labels_train = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\labels\train"
output_labels_val = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\labels\val"
```

2. 运行转换脚本：
```bash
python convert_labelme_to_yolo.py
```

**输入格式**：
- LabelMe JSON 文件（矩形标注）
- 对应的 JPG/PNG 图像文件

**输出格式**：
```
uav_custom/
├── images/
│   ├── train/          # 训练图像
│   └── val/            # 验证图像
└── labels/
    ├── train/          # 训练标签 (YOLO 格式)
    └── val/            # 验证标签 (YOLO 格式)
```

**YOLO 标签格式**：
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.4  # 归一化坐标（0-1）
```

---

### 2. UAVSwarm 数据集转换

**脚本**：`convert_uavswarm_to_yolo.py`

**功能说明**：
- 将 UAVSwarm 数据集的 MOT 格式（gt.txt）转换为 YOLO 格式
- 支持可见性和置信度过滤（visibility > 0.3, conf > 0）
- 按帧组织数据，为每一帧生成对应的标注文件
- 自动划分训练集和验证集

**MOT 格式说明**：
```
frame_id, track_id, x, y, w, h, conf, class_id, visibility
1, 1, 100, 200, 50, 80, 1.0, 0, 0.8
```

**使用方法**：

1. 修改脚本配置：
```python
train_base_dir = "D:/UAV/YOLOv12-BoT-SORT-ReID/data/UAVSwarm-dataset-master/train"
test_base_dir = "D:/UAV/YOLOv12-BoT-SORT-ReID/data/UAVSwarm-dataset-master/test"
output_base = "D:/UAV/YOLOv12-BoT-SORT-ReID/data/uavswarm_yolo"
```

2. 运行转换：
```bash
python convert_uavswarm_to_yolo.py
```

**输出结构**：
```
uavswarm_yolo/
├── images/
│   ├── train/          # 训练图像（从视频序列提取）
│   └── val/            # 验证图像
└── labels/
    ├── train/          # 对应标签
    └── val/
```

---

### 3. 数据集配置文件

创建 YAML 配置文件用于训练，例如 `uav_custom.yaml`：

```yaml
train: D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\images\train
val: D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\images\val
nc: 1
names: ['UAV']
```

---

## 🎓 模型训练

### YOLOv12 训练

**脚本**：`BoT-SORT/yolov12/train.py`

**功能说明**：
- 基于预训练权重进行迁移学习
- 支持数据增强（Mosaic、Mixup）
- 自动下载预训练模型
- 灵活的超参数配置

**训练命令**：

```bash
cd BoT-SORT/yolov12

# 基础训练
python train.py --model_name ./weights/MOT_yolov12n.pt \
                --yaml_path uav_custom.yaml \
                --n_epoch 100 \
                --bs 64 \
                --imgsz 640

# 完整参数示例
python train.py \
    --model_name ./weights/MOT_yolov12n.pt \
    --yaml_path uav_custom.yaml \
    --n_epoch 100 \
    --n_patience 50 \
    --bs 64 \
    --imgsz 640 \
    --single_cls True \
    --n_worker 8 \
    --save_path ./runs/uav \
    --lr0 0.01 \
    --lrf 0.01 \
    --mosaic 1.0 \
    --mixup 0.0 \
    --augment True
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--model_name` | 预训练模型路径 | `./weights/MOT_yolov12n.pt` |
| `--yaml_path` | 数据集配置文件 | `uav.yaml` |
| `--n_epoch` | 训练轮数 | 100 |
| `--n_patience` | 早停耐心值 | 100 |
| `--bs` | 批次大小 | 64 |
| `--imgsz` | 输入图像尺寸 | 640 |
| `--single_cls` | 单类别模式 | True |
| `--lr0` | 初始学习率 | 0.01 |
| `--lrf` | 最终学习率 | 0.01 |
| `--mosaic` | Mosaic 增强比例 | 1.0 |
| `--mixup` | Mixup 增强比例 | 0.0 |

**训练输出**：
- 训练日志：`runs/uav/train*/`
- 最佳权重：`runs/uav/train*/weights/best.pt`
- 最后权重：`runs/uav/train*/weights/last.pt`

---

## 🔍 模型推理与检测

### 1. 图像序列检测

**脚本**：`test_uavswarm.py`

**功能说明**：
- 对图像序列进行批量检测
- 生成 MOT 格式的检测结果
- 保存可视化结果（可选）

**使用方法**：

```bash
python test_uavswarm.py \
    --model_path BoT-SORT/yolov12/runs/uav/train/weights/best.pt \
    --image_folder data/UAVSwarm-dataset-master/test/UAVSwarm-02/img1 \
    --output_folder test_results/UAVSwarm-02 \
    --conf_threshold 0.3
```

**输出文件**：
- `detections.txt`：MOT 格式检测结果
- `vis/`：可视化结果图像（如果启用）

---

### 2. 视频跟踪推理

**脚本**：`BoT-SORT/tools/inference.py`

**功能说明**：
- 完整的检测+跟踪流程
- 支持视频文件和图像序列
- 集成 Fast-ReID 重识别
- 支持全局运动补偿（GMC）

**基础使用**：

```bash
cd BoT-SORT

python tools/inference.py \
    --weights ./yolov12/weights/v1/MOT_yolov12n.pt \
    --source ../data/MultiUAV_Train/TrainVideos/video001.mp4 \
    --img-size 1600 \
    --device 0 \
    --track_buffer 60 \
    --agnostic-nms \
    --save_path_answer ../test_results/video001
```

**高级配置（含 ReID）**：

```bash
python tools/inference.py \
    --weights ./yolov12/weights/v1/MOT_yolov12n.pt \
    --source ../data/MultiUAV_Train/TrainVideos/video001.mp4 \
    --img-size 1600 \
    --device 0 \
    --track_buffer 60 \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --agnostic-nms \
    --hide-labels-name \
    --nosave
```

**关键参数**：

| 参数 | 说明 |
|-----|------|
| `--weights` | YOLOv12 模型权重路径 |
| `--source` | 输入视频/图像序列路径 |
| `--img-size` | 推理图像尺寸（1600 适合高分辨率视频）|
| `--device` | GPU 设备 ID（0, 1, ...）或 'cpu' |
| `--track_buffer` | 跟踪缓冲区大小（帧数）|
| `--with-reid` | 启用 Fast-ReID 重识别 |
| `--fast-reid-config` | ReID 模型配置文件 |
| `--fast-reid-weights` | ReID 模型权重 |
| `--agnostic-nms` | 类别无关的 NMS |
| `--hide-labels-name` | 隐藏标签名称 |
| `--nosave` | 不保存可视化视频（仅保存跟踪结果）|

---

### 3. 批量视频处理

**脚本**：`BoT-SORT/batch_process_videos.py`

**功能说明**：
- 自动遍历视频文件夹
- 批量处理所有视频
- 支持断点续传（跳过已处理视频）
- 显示处理进度和统计信息

**使用方法**：

1. 修改脚本中的配置：
```python
video_folder = Path("D:/UAV/YOLOv12-BoT-SORT-ReID/data/MultiUAV_Train/TrainVideos")
output_dir = Path("D:/UAV/YOLOv12-BoT-SORT-ReID/TrackEval/data/trackers/mot_challenge/UAV-train/my_botsort/data")
```

2. 运行批处理：
```bash
cd BoT-SORT
python batch_process_videos.py
```

**输出示例**：
```
Found 50 video files to process
============================================================
[1/50] Processing: video001.mp4
============================================================
✅ video001.mp4 completed successfully in 45.3s
============================================================
[2/50] Processing: video002.mp4
============================================================
⏭️ video002.mp4 already processed, skipping...
...
```

---

## 📊 结果评估

### 1. 检测精度评估

**脚本**：`evaluate_detections.py`

**功能说明**：
- 计算检测指标：Precision、Recall、F1-Score、mAP@50
- 支持 MOT 格式的真值和预测比较
- 可设置不同的 IoU 阈值

**使用方法**：

```bash
python evaluate_detections.py \
    --pred_file test_results/UAVSwarm-02/detections.txt \
    --gt_file data/UAVSwarm-dataset-master/test/UAVSwarm-02/gt/gt.txt \
    --iou_threshold 0.5
```

**输出示例**：
```
📊 Detection Evaluation Results
================================
Total Frames: 1500
Total GT Objects: 12500
Total Predictions: 12800

Precision: 0.8765
Recall: 0.8543
F1-Score: 0.8653
mAP@50: 0.8721
```

---

### 2. MOT 指标评估

**工具**：TrackEval

**功能说明**：
- 计算标准 MOT 指标：HOTA、MOTA、IDF1
- 支持多场景批量评估
- 生成详细的评估报告

**TrackEval 评估指标**：

| 指标 | 说明 |
|-----|------|
| **HOTA** | Higher Order Tracking Accuracy（高阶跟踪精度）|
| **MOTA** | Multiple Object Tracking Accuracy（多目标跟踪精度）|
| **IDF1** | Identification F1 Score（身份识别 F1 分数）|
| **DetA** | Detection Accuracy（检测精度）|
| **AssA** | Association Accuracy（关联精度）|
| **MT** | Mostly Tracked（主要跟踪目标数）|
| **ML** | Mostly Lost（主要丢失目标数）|
| **FP** | False Positives（误报）|
| **FN** | False Negatives（漏报）|
| **ID Sw.** | Identity Switches（ID 切换次数）|

**使用TrackEval**：

```bash
cd TrackEval

python scripts/run_mot_challenge.py \
    --GT_FOLDER data/gt/mot_challenge/ \
    --TRACKERS_FOLDER data/trackers/mot_challenge/ \
    --TRACKER_SUB_FOLDER my_botsort \
    --BENCHMARK UAV-train \
    --SPLIT_TO_EVAL train \
    --METRICS HOTA CLEAR Identity
```

---

### 3. MOTA 快速评估

**脚本**：`BoT-SORT/tools/mota.py`

**功能说明**：
- 快速计算 MOTA、MOTP 等基础指标
- 适合调试和快速验证

**使用方法**：

修改脚本中的路径后运行：
```bash
cd BoT-SORT
python tools/mota.py
```

---

## 🔧 高级功能

### 1. 轨迹插值优化

**脚本**：`BoT-SORT/tools/interpolation.py`

**功能说明**：
- 对跟踪结果进行轨迹插值
- 填补短暂的跟踪间断
- 提高跟踪连续性和 MOTA 指标

**使用方法**：

```bash
cd BoT-SORT

python tools/interpolation.py \
    --txt_path ../test_results/UAVSwarm-02 \
    --save_path ../test_results/UAVSwarm-02_interpolated \
    --n_min 5 \
    --n_dti 20
```

**参数说明**：
- `--txt_path`：原始跟踪结果目录
- `--save_path`：插值后结果保存路径（None 则覆盖原文件）
- `--n_min`：最小轨迹长度（小于此值的轨迹不进行插值）
- `--n_dti`：最大插值间隔（帧数）

**插值效果**：
- 填补 1-20 帧之间的跟踪空白
- 减少 ID 切换次数
- 提高整体跟踪稳定性

---

### 2. 数据集统计分析

**脚本**：`BoT-SORT/getInfo.py`

**功能说明**：
- 统计数据集的基本信息
- 分析目标尺寸分布
- 计算数据集统计指标

**使用方法**：

```python
# 在脚本中调用相应函数
from getInfo import sot_train, mot_train

# SOT 数据集分析
sot_train("data/SOT/train")

# MOT 数据集分析
mot_train("data/MOT/train")
```

**输出信息**：
- 序列数量、帧数统计
- 目标数量和密度
- 边界框尺寸分布
- 图像分辨率统计

---

### 3. 全局运动补偿（GMC）

**模块**：`BoT-SORT/tracker/gmc.py`

**功能说明**：
- 补偿摄像机运动造成的位置偏移
- 提高无人机航拍场景的跟踪精度
- 支持多种方法：ORB、ECC、OpticalFlow

**GMC 方法**：

| 方法 | 说明 | 适用场景 |
|-----|------|---------|
| `file` | 从文件读取相机运动参数 | 已知相机运动 |
| `orb` | ORB 特征匹配 | 一般场景 |
| `ecc` | 增强相关系数 | 纹理丰富场景 |
| `sparseOptFlow` | 稀疏光流 | 快速运动 |
| `none` | 不使用 GMC | 静态相机 |

**在 inference.py 中使用**：
```bash
python tools/inference.py \
    --cmc-method orb \
    ... # 其他参数
```

---

### 4. 多摄像头跟踪

**模块**：`BoT-SORT/tracker/mc_bot_sort.py`

**功能说明**：
- 支持多摄像头场景的目标跟踪
- 跨摄像头的目标重识别
- 全局 ID 管理

**使用场景**：
- 多无人机协同监控
- 大范围区域覆盖
- 目标跨视野跟踪

---

## 🎯 完整工作流程示例

### 场景：从数据标注到模型部署

```bash
# 1. 数据准备：转换 LabelMe 标注
python convert_labelme_to_yolo.py

# 2. 模型训练
cd BoT-SORT/yolov12
python train.py --model_name ./weights/MOT_yolov12n.pt \
                --yaml_path uav_custom.yaml \
                --n_epoch 100 \
                --bs 64

# 3. 单序列检测测试
cd ../..
python test_uavswarm.py \
    --model_path BoT-SORT/yolov12/runs/uav/train/weights/best.pt \
    --image_folder data/test/sequence_01 \
    --output_folder test_results/sequence_01

# 4. 检测结果评估
python evaluate_detections.py \
    --pred_file test_results/sequence_01/detections.txt \
    --gt_file data/test/sequence_01/gt/gt.txt

# 5. 视频跟踪（含 ReID）
cd BoT-SORT
python tools/inference.py \
    --weights ./yolov12/runs/uav/train/weights/best.pt \
    --source ../data/test_video.mp4 \
    --img-size 1600 \
    --device 0 \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth

# 6. 轨迹插值优化
python tools/interpolation.py \
    --txt_path ../test_results/track_output \
    --n_min 5 \
    --n_dti 20

# 7. MOT 指标评估
cd ../TrackEval
python scripts/run_mot_challenge.py \
    --GT_FOLDER data/gt/mot_challenge/ \
    --TRACKERS_FOLDER data/trackers/mot_challenge/ \
    --TRACKER_SUB_FOLDER my_botsort \
    --METRICS HOTA CLEAR Identity
```

---

## 📈 性能优化建议

### 1. 训练优化

- **数据增强**：启用 Mosaic (1.0) 和适量 Mixup (0.1-0.2)
- **图像尺寸**：高分辨率视频使用 1280 或 1600
- **批次大小**：根据 GPU 显存调整（RTX 3090: 64-128）
- **学习率**：使用余弦退火，初始 0.01，最终 0.001

### 2. 推理优化

- **置信度阈值**：0.3-0.5（根据场景调整）
- **NMS 阈值**：0.4-0.5
- **跟踪缓冲区**：30-60 帧（根据视频帧率）
- **图像尺寸**：推理时可以大于训练尺寸（如 1600）

### 3. 跟踪优化

- **启用 ReID**：提高遮挡后的重识别能力
- **启用 GMC**：补偿摄像机运动
- **调整匹配阈值**：match_thresh (0.7-0.9)
- **轨迹插值**：n_dti 设置为 10-30 帧

---

## ❓ 常见问题

### Q1: 训练时显存不足？
**A:** 减小批次大小 `--bs 32` 或图像尺寸 `--imgsz 320`

### Q2: 推理速度慢？
**A:** 
- 使用较小的模型（yolov12n 而非 yolov12x）
- 降低输入图像尺寸
- 使用 GPU 加速 `--device 0`
- 禁用可视化 `--nosave`

### Q3: 跟踪效果不佳？
**A:**
- 降低检测置信度阈值
- 增加跟踪缓冲区 `--track_buffer 60`
- 启用 ReID `--with-reid`
- 使用 GMC 补偿相机运动

### Q4: ID 切换频繁？
**A:**
- 启用 Fast-ReID 重识别
- 增大匹配阈值 `--match_thresh 0.9`
- 使用轨迹插值后处理
- 调整外观相似度阈值

### Q5: 如何处理小目标？
**A:**
- 增大输入图像尺寸 `--img-size 1600`
- 调低 `--min_box_area 5`
- 使用多尺度训练
- 增强数据集中的小目标样本

---

### 相关项目

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [BoT-SORT Official](https://github.com/NirAharon/BoT-SORT)
- [Fast-ReID](https://github.com/JDAI-CV/fast-reid)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目基于 MIT 许可证发布。详见 [LICENSE](LICENSE) 文件。

---

## 👥 作者与致谢

**项目维护者**：[LiangXianguang]

**特别感谢**：
- Ultralytics 团队提供的 YOLOv8/YOLOv12 框架
- BoT-SORT 作者的开源贡献
- Fast-ReID 团队的重识别模型
- TrackEval 工具的开发者

---

## 📧 联系方式

- 邮箱：2811306715@qq.com

---

## 🔄 更新日志

### v1.0.0 (2025-11-25)
- ✅ 完整的数据转换工具（LabelMe、UAVSwarm）
- ✅ YOLOv12 训练和推理流程
- ✅ BoT-SORT 多目标跟踪
- ✅ Fast-ReID 重识别集成
- ✅ TrackEval 评估工具
- ✅ 批量视频处理
- ✅ 轨迹插值优化
- ✅ 完整文档和示例

---

**🎉 祝您使用愉快！如有问题，欢迎提 Issue！**
