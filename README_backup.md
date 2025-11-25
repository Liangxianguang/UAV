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

## 📚 参考资料

### 论文

1. **YOLOv12**: [待发布]
2. **BoT-SORT**: [Robust Multi-Object Tracking by Marginal Inference](https://arxiv.org/abs/2206.14651)
3. **Fast-ReID**: [FastReID: A Pytorch Toolbox for General Instance Re-identification](https://arxiv.org/abs/2006.02631)
4. **HOTA Metrics**: [HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://link.springer.com/article/10.1007/s11263-020-01375-2)

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

**项目维护者**：[您的名字]

**特别感谢**：
- Ultralytics 团队提供的 YOLOv8/YOLOv12 框架
- BoT-SORT 作者的开源贡献
- Fast-ReID 团队的重识别模型
- TrackEval 工具的开发者

---

## 📧 联系方式

- 邮箱：your.email@example.com
- GitHub Issues：[项目 Issues 页面](https://github.com/your-repo/issues)

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

### 使用方法

```- `ultralytics` - YOLOv12 模型库

编辑脚本中的参数：

```python

model_path = r'D:\path\to\best.pt'              # 模型权重

image_folder = r'D:\path\to\images'             # 输入图像文件夹**使用方法**：编辑脚本中的路径配置后运行：- `torch` / `torchvision` - 深度学习框架### 2.1 LabelMe 格式转换```bash

output_folder = r'D:\path\to\output'            # 输出结果目录

conf_threshold = 0.3                             # 置信度阈值```bash

```

python convert_labelme_to_yolo.py- `opencv-python` - 图像处理

运行推理：

```bash```

python test_uavswarm.py

```- `numpy` - 数值计算如果你使用 LabelMe 工具进行数据标注，生成的 JSON 文件可以通过此脚本转换为 YOLO 格式的 TXT 标签。



### 输出说明**输出结构**：



- **detections.txt**：MOT 格式的检测结果文件```- `tqdm` - 进度条

- **vis_*.jpg**：可视化图像

uav_custom/

**detections.txt 格式**：

```├── images/pip install -r BoT-SORT/requirements.txt

1,-1,123.5,98.2,45.0,52.1,0.95,-1,-1,-1

1,-1,345.8,210.3,38.5,48.9,0.87,-1,-1,-1│   ├── train/  # 训练集图片（70%）

2,-1,125.3,100.1,44.5,51.8,0.92,-1,-1,-1

```│   └── val/    # 验证集图片（30%）## 📂 数据准备



---└── labels/



## 📊 结果评估    ├── train/  # 训练集标签- **脚本**: `convert_labelme_to_yolo.py`



**脚本**：`evaluate_detections.py`    └── val/    # 验证集标签



### 功能说明```### 2.1 LabelMe 标注数据转换



对比预测结果和真实标签，计算检测精度指标：



1. **解析文件**：读取预测结果和真实标签（都是 MOT 格式）---- **功能**: ```

2. **IOU 匹配**：计算预测框和真实框的交并比

3. **计算指标**：TP（真正例）、FP（假正例）、FN（假负例）

4. **生成报告**：输出精度、召回率、F1 分数等指标

5. **逐帧对比**：随机采样显示部分帧的详细对比### 2.2 UAVSwarm 数据集转换**场景**：你使用 LabelMe 工具对图像进行了矩形标注，生成了 JSON 格式的标签文件。



### 计算的指标



| 指标 | 公式 | 说明 |**场景**：有 UAVSwarm 数据集（MOT Challenge 格式），包含 `train/` 和 `test/` 目录。  - 遍历指定目录下的 JSON 文件。

|-----|------|------|

| **Precision** | TP / (TP + FP) | 检测准确率 |

| **Recall** | TP / (TP + FN) | 检测召回率 |

| **F1-Score** | 2 × P × R / (P + R) | 精度和召回的调和平均数 |**脚本**：`convert_uavswarm_to_yolo.py`**脚本**：`convert_labelme_to_yolo.py`

| **Detection Rate** | TP / Total_GT | 目标检测率 |

| **False Alarm Rate** | FP / Total_Pred | 误检率 |



### 使用方法**功能**：  - 将矩形标注转换为 YOLO 归一化坐标 (class x_center y_center w h)。



编辑脚本中的文件路径：- 读取 MOT 格式的 `gt.txt` 文件

```python

pred_file = r'D:\path\to\detections.txt'     # 预测结果- 过滤机制：只保留可见性 > 0.3 和置信度 > 0 的目标**功能**：

gt_file = r'D:\path\to\gt.txt'               # 真实标注

```- 按帧号分组，为每帧生成 YOLO 格式的标签文件



运行评估：- 自动扫描所有序列，随机分割为训练集（70%）和验证集（30%）- 解析 LabelMe 生成的 JSON 标注文件  - 自动划分训练集和验证集，并移动图片。

```bash

python evaluate_detections.py- 生成 YOLO 训练所需的结构

```

- 提取矩形标注的坐标，转换为 YOLO 格式（归一化的类别和中心坐标）

---

**MOT 格式说明**：

## 🚀 核心模块说明

```- 自动扫描目录下的所有子文件夹，收集所有 JSON 文件和对应的图片- **运行**:## 2. 数据准备[![arXiv](https://img.shields.io/badge/arXiv-2503.17237-b31b1b.svg)](https://arxiv.org/abs/2503.17237)

### BoT-SORT 跟踪器

frame_id, track_id, x, y, w, h, conf, class_id, visibility

位置：`BoT-SORT/tracker/`

1,1,100,50,30,50,1,-1,0.9- 将数据随机分割为训练集（70%）和验证集（30%）

**主要组件**：

- **bot_sort.py**：核心跟踪算法```

  - 使用卡尔曼滤波预测轨迹

  - 提取目标外观特征（ReID）- 创建标准的 YOLO 目录结构：`images/train`, `images/val`, `labels/train`, `labels/val`  ```bash

  - 进行轨迹匹配

  **使用方法**：

- **kalman_filter.py**：卡尔曼滤波器

  - 预测目标位置和速度```bash

  

- **matching.py**：轨迹匹配算法python convert_uavswarm_to_yolo.py

  - Hungarian 算法进行二部图匹配

  - IOU 相似度计算```**YOLO 标签格式**：  python convert_labelme_to_yolo.py[![PyPI - Python Version](https://img.shields.io/badge/python-3.11-blue.svg?logo=python&logoColor=gold)](https://www.python.org/downloads/release/python-3110/)

  - 特征距离计算



- **gmc.py**：全局运动补偿

  - 处理摄像机运动**输出结构**：```



### YOLOv12 检测模型```



位置：`BoT-SORT/yolov12/`uavswarm_yolo/class_id x_center y_center width height  ```



**主要功能**：├── images/

- **train.py**：模型训练脚本

- **配置文件**（YAML）：│   ├── train/  # 所有序列的训练图片0 0.5 0.5 0.3 0.4  # 归一化坐标（0-1）

  - `uav.yaml` - 默认 UAV 数据集配置

  - `uav_custom.yaml` - 自定义 UAV 数据集配置│   └── val/    # 所有序列的验证图片

  - `uavswarm.yaml` - UAVSwarm 数据集配置

└── labels/```  *注意：需在脚本中修改 `base_dir` 和输出路径。*项目数据存放在 `data/` 目录下。提供了以下脚本用于数据格式转换：[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wish44165/YOLOv12-BoT-SORT-ReID) 

**训练命令示例**：

```bash    ├── train/  # 对应的训练标签

cd BoT-SORT/yolov12/

python train.py --model_name weights/MOT_yolov12n.pt \    └── val/    # 对应的验证标签

                 --yaml_path uav.yaml \

                 --n_epoch 100 \```

                 --bs 32

```**使用方法**：



### FastReID 特征提取---



位置：`BoT-SORT/fast_reid/`



**功能**：## 🔍 模型推理与检测

- 提取目标外观特征

- 用于轨迹关联的特征匹配编辑脚本中的路径配置：### 2.2 UAVSwarm 数据集转换[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x5T6woUdV6dD_T6qdYcKG04Q2iVVHGoD?usp=sharing)

- 支持特征距离计算

**脚本**：`test_uavswarm.py`

---

```python

## 📁 项目文件结构

### 功能说明

```

YOLOv12-BoT-SORT-ReID/base_dir = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\images"  # 输入：包含 JSON 的目录针对 UAVSwarm 数据集（MOT 格式），将其转换为 YOLO 训练所需的图片和标签格式。

│

├── BoT-SORT/使用已训练的 YOLOv12 模型对图像序列进行目标检测：

│   ├── yolov12/                    # YOLOv12 检测模型

│   │   ├── train.pyoutput_images_train = r"..."  # 输出训练集图片

│   │   ├── weights/

│   │   ├── models/1. **加载模型**：从 `.pt` 权重文件加载 YOLOv12 模型

│   │   └── ultralytics/

│   │2. **批量推理**：遍历指定文件夹中的所有图像（支持 .jpg, .jpeg, .png, .bmp）output_images_val = r"..."    # 输出验证集图片- **LabelMe 转 YOLO**:[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/yuhsi44165/yolov12-bot-sort/)

│   ├── tracker/                    # BoT-SORT 跟踪器

│   │   ├── bot_sort.py3. **生成检测结果**：

│   │   ├── kalman_filter.py

│   │   ├── matching.py   - 保存为 **MOT 格式** 的 TXT 文件 (`detections.txt`)output_labels_train = r"..."  # 输出训练集标签

│   │   ├── gmc.py

│   │   └── tracking_utils/   - 每行一个检测框：`frame_id, -1, x, y, w, h, conf, -1, -1, -1`

│   │

│   ├── fast_reid/                  # FastReID 特征提取4. **可视化**：在图像上绘制检测框和置信度，保存为 `vis_*.jpg`output_labels_val = r"..."    # 输出验证集标签- **脚本**: `convert_uavswarm_to_yolo.py`

│   │   ├── fast_reid_interfece.py

│   │   ├── fastreid/

│   │   ├── projects/

│   │   └── logs/### 使用方法```

│   │

│   ├── tools/                      # 工具脚本

│   │   ├── track.py

│   │   ├── predict_track1/2/3.py编辑脚本中的参数：- **功能**:  如果你使用 LabelMe 进行标注，可以使用 `convert_labelme_to_yolo.py` 将 JSON 文件转换为 YOLO 格式的 TXT 标签。

│   │   ├── inference.py

│   │   ├── mota.py```python

│   │   └── demo.py

│   │model_path = r'D:\path\to\best.pt'              # 模型权重然后运行：

│   ├── cocoapi/                    # COCO API

│   ├── datasets/                   # 数据集配置image_folder = r'D:\path\to\images'             # 输入图像文件夹

│   ├── logs/                       # 训练日志

│   ├── runs/                       # 推理输出output_folder = r'D:\path\to\output'            # 输出结果目录```bash  - 读取 MOT 格式的 `gt.txt`。

│   ├── submit/                     # 提交结果

│   ├── VideoCameraCorrection/      # 摄像机矫正conf_threshold = 0.3                             # 置信度阈值

│   └── requirements.txt

│```python convert_labelme_to_yolo.py

├── data/

│   ├── images/                     # 原始图像

│   ├── labels/                     # 标注标签

│   ├── demo/                       # 演示数据运行推理：```  - 过滤低可见度和低置信度的目标。  ```bash<a href="https://doi.org/10.5281/zenodo.15203123"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15203123.svg" alt="DOI"></a>

│   ├── uav_custom/                 # LabelMe 转换数据

│   ├── uavswarm_yolo/              # UAVSwarm 转换数据```bash

│   ├── MOT/ / SOT/                 # 标准数据集

│   ├── MultiUAV_Test/ / MultiUAV_Train/python test_uavswarm.py

│   └── UAVSwarm-dataset-master/    # 原始 UAVSwarm 数据

│```

├── test_results/

│   ├── UAVSwarm-02/ / UAVSwarm-12/ / UAVSwarm-44/**输出示例**：  - 按序列处理并划分为训练集和验证集。

│   ├── UAVSwarm-*_bytetrack/       # ByteTrack 结果

│   └── UAVSwarm-*_tracking/        # BoT-SORT 结果### 输出说明

│

├── TrackEval/```

│   ├── trackeval/

│   │   ├── metrics/                # 评估指标- **detections.txt**：MOT 格式的检测结果文件

│   │   │   ├── hota.py             # HOTA（推荐）

│   │   │   ├── clear.py            # MOTA/MOTP- **vis_*.jpg**：可视化图像，便于查看检测效果uav_custom/- **运行**:  python convert_labelme_to_yolo.py<a href="https://github.com/wish44165/wish44165/tree/main/assets"><img src="https://github.com/wish44165/wish44165/blob/main/assets/msi_Cyborg_15_A12VE_badge.svg" alt="MSI"></a> 

│   │   │   ├── identity.py         # IDF1

│   │   │   └── track_map.py        # Track mAP

│   │   ├── datasets/               # 数据集加载器

│   │   ├── eval.py                 # 评估主程序**detections.txt 格式**：├── images/

│   │   └── plotting.py             # 结果绘图

│   │```

│   ├── scripts/

│   │   └── run_mot_challenge.py    # MOT 评估脚本1,-1,123.5,98.2,45.0,52.1,0.95,-1,-1,-1│   ├── train/  # 训练集图片（70%）  ```bash

│   │

│   ├── data/1,-1,345.8,210.3,38.5,48.9,0.87,-1,-1,-1

│   │   ├── gt/                     # 真实标注

│   │   └── trackers/               # 跟踪结果2,-1,125.3,100.1,44.5,51.8,0.92,-1,-1,-1│   └── val/    # 验证集图片（30%）

│   │

│   ├── docs/                       # 文档```

│   └── requirements.txt

│└── labels/  python convert_uavswarm_to_yolo.py  ```<a href="https://dashboard.hpc.unimelb.edu.au/"><img src="https://github.com/wish44165/wish44165/blob/main/assets/unimelb_spartan.svg" alt="Spartan"></a> 

├── convert_labelme_to_yolo.py      # LabelMe 转换工具

├── convert_uavswarm_to_yolo.py     # UAVSwarm 转换工具---

├── test_uavswarm.py                # 推理检测工具

├── evaluate_detections.py          # 评估工具    ├── train/  # 训练集标签

└── README.md                       # 本文档

```## 📊 结果评估



---    └── val/    # 验证集标签  ```



## 💻 使用流程示例**脚本**：`evaluate_detections.py`



### 流程 1：使用自定义 LabelMe 标注数据```



```bash### 功能说明

# 1. 转换标注格式

python convert_labelme_to_yolo.py  *注意：需在脚本中修改 `base_dir` 为数据集根目录。*



# 2. 使用生成的数据训练模型对比预测结果和真实标签，计算检测精度指标：

cd BoT-SORT/yolov12/

python train.py --yaml_path ../../../uav_custom.yaml --n_epoch 100---



# 3. 用训练好的模型推理1. **解析文件**：读取预测结果和真实标签（都是 MOT 格式）

cd ../../..

python test_uavswarm.py2. **IOU 匹配**：计算预测框和真实框的交并比



# 4. 评估检测结果3. **计算指标**：

python evaluate_detections.py

```   - **TP（真正例）**：IOU ≥ 阈值的正确检测### 2.2 UAVSwarm 数据集转换



### 流程 2：使用 UAVSwarm 数据集   - **FP（假正例）**：错误的检测



```bash   - **FN（假负例）**：未检测到的目标## 3. 模型推理与检测- **UAVSwarm 数据集转换**:[![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://medium.com/@scofield44165/ubuntu-24-04-1-getting-started-with-yolov12-bot-sort-reid-on-linux-20826ffc8224)

# 1. 转换 MOT 格式

python convert_uavswarm_to_yolo.py4. **生成报告**：输出精度、召回率、F1 分数等指标



# 2. 训练模型5. **逐帧对比**：随机采样显示部分帧的详细对比**场景**：你有 UAVSwarm 数据集（MOT Challenge 格式），包含 `train/` 和 `test/` 目录，每个序列中有 `gt/gt.txt` 和 `img1/` 文件夹。

cd BoT-SORT/yolov12/

python train.py --yaml_path ../../../uavswarm.yaml



# 3. 推理### 计算的指标

cd ../../..

python test_uavswarm.py



# 4. 评估| 指标 | 公式 | 说明 |**脚本**：`convert_uavswarm_to_yolo.py`

python evaluate_detections.py

```|-----|------|------|



---| **Precision** | TP / (TP + FP) | 检测准确率 |使用训练好的 YOLOv12 模型对图像序列进行目标检测。  针对 UAVSwarm 数据集，使用 `convert_uavswarm_to_yolo.py` 将其转换为 YOLO 训练所需的格式。[![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)](https://medium.com/@scofield44165/macos-tahoe-26-0-1-getting-started-with-yolov12-bot-sort-reid-on-mac-f87400d5b096)



## 🔧 TrackEval 评估工具| **Recall** | TP / (TP + FN) | 检测召回率 |



位置：`TrackEval/`| **F1-Score** | 2 × P × R / (P + R) | 精度和召回的调和平均数 |**功能**：



### 支持的指标| **Detection Rate** | TP / Total_GT | 目标检测率 |



- **HOTA**（推荐）：更高阶的跟踪精度| **False Alarm Rate** | FP / Total_Pred | 误检率 |- 读取 MOT 格式的 `gt.txt` 文件（包含所有帧的目标标注）

- **MOTA/MOTP**：多目标跟踪精度

- **IDF1**：身份保留度

- **Track mAP**：跟踪平均精度

### 使用方法- **过滤机制**：

### 使用示例



```bash

cd TrackEval/编辑脚本中的文件路径：  - 只保留可见性 (visibility) > 0.3 的目标- **脚本**: `test_uavswarm.py`  ```bash[![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://medium.com/@scofield44165/windows-11-getting-started-with-yolov12-bot-sort-reid-on-windows-11-24ee1f1cd513)

python scripts/run_mot_challenge.py \

    --benchmark_name MOT17 \```python

    --split_to_eval test \

    --tracker_folder ../submit/track3/pred_file = r'D:\path\to\detections.txt'     # 预测结果  - 只保留置信度 (conf) > 0 的目标

```

gt_file = r'D:\path\to\gt.txt'               # 真实标注

---

```- 按帧号 (frame_id) 分组，为每帧生成 YOLO 格式的标签文件- **功能**:

## 📝 常见问题



### Q1：推理时提示模型文件不存在？

**A**：检查 `test_uavswarm.py` 中的 `model_path` 是否正确指向 `.pt` 权重文件。运行评估：- 自动扫描 `train/` 和 `test/` 目录，收集所有序列



### Q2：转换数据时报错 "File not found"？```bash

**A**：确保输入路径（`base_dir`）存在，且包含相应格式的文件（JSON 或 gt.txt）。

python evaluate_detections.py- 将所有序列随机分割为训练集（70%）和验证集（30%）  - 加载 YOLOv12 模型权重。  python convert_uavswarm_to_yolo.py[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/publication/390114692_Strong_Baseline_Multi-UAV_Tracking_via_YOLOv12_with_BoT-SORT-ReID)

### Q3：评估结果精度很低？

**A**：可能原因：```

- 模型训练不足

- 置信度阈值设置过高- 生成 YOLO 训练所需的结构

- 真实标注与检测框的 IOU 不匹配

---

### Q4：如何调整检测敏感度？

**A**：修改 `test_uavswarm.py` 中的 `conf_threshold`，值越低越敏感。  - 对指定文件夹下的所有图片进行推理。



### Q5：如何使用 ReID 进行跟踪？## 🚀 核心模块说明

**A**：在 `BoT-SORT/tools/predict_track3.py` 中启用 `--with-reid` 选项。

**MOT 格式说明**：

---

### BoT-SORT 跟踪器

## 📚 参考资源

```  - 生成 MOT 格式的检测结果文件 `detections.txt`。  ```[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@scofield44165/strong-baseline-multi-uav-tracking-via-yolov12-with-bot-sort-reid-5d6b71230e39)

- [YOLOv12](https://github.com/sunsmarterjie/yolov12) - 检测模型

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 跟踪算法位置：`BoT-SORT/tracker/`

- [TrackEval](https://github.com/JonathonLuiten/TrackEval) - 评估工具

- [MOT Challenge](https://motchallenge.net/) - 多目标跟踪基准frame_id, track_id, x, y, w, h, conf, class_id, visibility

- [LabelMe](http://labelme.csail.mit.edu/) - 标注工具

**主要组件**：

---

- **bot_sort.py**：核心跟踪算法1,1,100,50,30,50,1,-1,0.9  - 保存带有检测框的可视化图片。

## 📄 License

  - 使用卡尔曼滤波预测轨迹

此项目代码遵循相关开源项目的许可证。

  - 提取目标外观特征（ReID）```

---

  - 进行轨迹匹配

**最后更新**：2025 年 11 月 25 日

  - **运行**:[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/playlist?list=PLfr5E6mAx5EUpqP41CPSm5Nwfqe35iwtl)

- **kalman_filter.py**：卡尔曼滤波器

  - 预测目标位置和速度**YOLO 转换**：坐标转换为归一化的中心点和宽高

  - 处理目标运动模型

    ```bash

- **matching.py**：轨迹匹配算法

  - Hungarian 算法进行二部图匹配**使用方法**：

  - IOU 相似度计算

  - 特征距离计算  python test_uavswarm.py## 3. 模型推理与测试



- **gmc.py**：全局运动补偿编辑脚本中的路径：

  - 处理摄像机运动

  - 改进轨迹的稳定性```python  ```



### YOLOv12 检测模型base_dir = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\UAVSwarm-dataset-master"  # 数据集根目录



位置：`BoT-SORT/yolov12/`output_train_imgs = r"..."    # 输出训练集图片  *参数配置（在脚本中修改）*:



**主要功能**：output_train_labels = r"..."  # 输出训练集标签

- **train.py**：模型训练脚本

  - 从预训练权重开始训练output_val_imgs = r"..."      # 输出验证集图片  - `model_path`: 模型权重路径 (e.g., `best.pt`)

  - 支持数据增强

  - 输出模型权重和日志output_val_labels = r"..."    # 输出验证集标签



- **配置文件**（YAML）：```  - `image_folder`: 待检测图片文件夹使用 `test_uavswarm.py` 对图像序列进行推理测试。该脚本会加载训练好的 YOLOv12 模型，对指定文件夹下的图片进行检测，并保存检测结果和可视化图像。

  - `uav.yaml`：默认 UAV 数据集配置

  - `uav_custom.yaml`：自定义 UAV 数据集配置

  - `uavswarm.yaml`：UAVSwarm 数据集配置

然后运行：  - `output_folder`: 结果保存路径

**训练命令示例**：

```bash```bash

cd BoT-SORT/yolov12/

python train.py --model_name weights/MOT_yolov12n.pt \python convert_uavswarm_to_yolo.py  - `conf_threshold`: 置信度阈值

                 --yaml_path uav.yaml \

                 --n_epoch 100 \```

                 --bs 32 \

                 --imgsz 640

```

**输出示例**：

### FastReID 特征提取

```## 4. 结果评估```bash<details><summary>Preface</summary>

位置：`BoT-SORT/fast_reid/`

uavswarm_yolo/

**功能**：

- 提取行人/无人机外观特征├── images/

- 用于轨迹关联的特征匹配

- 支持特征距离计算│   ├── train/  # 所有序列的训练图片



**接口**：`fast_reid_interfece.py`│   └── val/    # 所有序列的验证图片对检测结果进行定量评估，计算精度指标。python test_uavswarm.py

- 加载预训练模型

- 提取输入图像的特征向量└── labels/

- 计算特征之间的相似度

    ├── train/  # 对应的训练标签

---

    └── val/    # 对应的验证标签

## 📁 详细文件结构

```- **脚本**: `evaluate_detections.py````The combination of YOLOv12 and BoT-SORT demonstrates strong object detection and tracking potential yet remains underexplored in current literature and implementations.

```

YOLOv12-BoT-SORT-ReID/

│

├── BoT-SORT/---- **功能**:

│   ├── yolov12/

│   │   ├── train.py                   # 训练脚本

│   │   ├── weights/                   # 权重文件

│   │   ├── models/                    # 模型定义## 🔍 模型推理与检测  - 读取预测结果 (`detections.txt`) 和真实标签 (`gt.txt`)。*注意：请在脚本中修改 `model_path` 和 `image_folder` 为你的实际路径。*

│   │   ├── ultralytics/               # Ultralytics 框架

│   │   ├── uav.yaml / uav_custom.yaml / uavswarm.yaml  # 数据集配置

│   │   └── runs/                      # 训练输出

│   │**脚本**：`test_uavswarm.py`  - 计算 **Precision**, **Recall**, **F1-Score**。

│   ├── tracker/

│   │   ├── bot_sort.py                # 核心跟踪算法

│   │   ├── kalman_filter.py           # 卡尔曼滤波器

│   │   ├── matching.py                # 轨迹匹配### 功能说明  - 计算检测率 (Detection Rate) 和误检率 (False Alarm Rate)。<img src="https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/assets/existing_methods_overview.png" width="100%">

│   │   ├── gmc.py                     # 全局运动补偿

│   │   ├── basetrack.py               # 轨迹基类

│   │   └── tracking_utils/            # 工具函数

│   │该脚本使用已训练的 YOLOv12 模型对图像序列进行目标检测：  - 支持随机采样进行逐帧结果对比，方便排查问题。

│   ├── fast_reid/

│   │   ├── fast_reid_interfece.py     # ReID 接口

│   │   ├── fastreid/                  # 模型实现

│   │   ├── projects/                  # 配置文件1. **加载模型**：从 `.pt` 权重文件加载 YOLOv12 模型- **运行**:## 4. 结果评估

│   │   └── logs/                      # 预训练模型

│   │2. **批量推理**：遍历指定文件夹中的所有图像（支持 .jpg, .jpeg, .png, .bmp）

│   ├── tools/

│   │   ├── track.py                   # 跟踪主程序3. **生成检测结果**：  ```bash

│   │   ├── predict_track1/2/3.py      # 各阶段推理脚本

│   │   ├── inference.py               # 通用推理脚本   - 保存为 **MOT 格式** 的 TXT 文件 (`detections.txt`)

│   │   ├── mota.py / interpolation.py # 指标和插值

│   │   └── demo.py                    # 演示脚本   - 每行一个检测框：`frame_id, -1, x, y, w, h, conf, -1, -1, -1`  python evaluate_detections.py```

│   │

│   ├── tools/ → datasets/4. **可视化**：在图像上绘制检测框和置信度，保存为 `vis_*.jpg`

│   ├── cocoapi/                       # COCO API

│   ├── datasets/                      # 数据集配置  ```

│   ├── logs/                          # 训练日志

│   ├── runs/                          # 推理输出### 使用方法

│   ├── submit/                        # 提交结果

│   ├── VideoCameraCorrection/         # 摄像机矫正  *参数配置（在脚本中修改）*:使用 `evaluate_detections.py` 对检测结果进行精度评估。该脚本会计算 Precision, Recall, F1-Score 等指标，并支持逐帧对比查看。[1] Jocher, Glenn, et al. "ultralytics/yolov5: v6. 0-YOLOv5n'Nano'models, Roboflow integration, TensorFlow export, OpenCV DNN support." Zenodo (2021).

│   ├── batch_process_videos.py        # 批处理脚本

│   ├── getInfo.py                     # 数据统计#### 配置参数

│   ├── requirements.txt               # 依赖

│   └── run_track3.sh                  # 运行脚本  - `pred_file`: 预测生成的检测文件路径

│

├── data/编辑脚本中的参数：

│   ├── images/                        # 原始图像

│   ├── labels/                        # 标注标签```python  - `gt_file`: 真实标注文件路径[2] Tian, Yunjie, Qixiang Ye, and David Doermann. "Yolov12: Attention-centric real-time object detectors." arXiv preprint arXiv:2502.12524 (2025).

│   ├── demo/                          # 演示数据

│   ├── uav_custom/                    # LabelMe 转换数据# 模型权重路径（你训练好的模型）

│   ├── uavswarm_yolo/                 # UAVSwarm 转换数据

│   ├── MOT/ / SOT/                    # 标准数据集model_path = r'D:\UAV\YOLOv12-BoT-SORT-ReID\BoT-SORT\yolov12\runs\uav\train15\weights\best.pt'

│   ├── MultiUAV_Test/ / MultiUAV_Train/

│   └── UAVSwarm-dataset-master/       # 原始 UAVSwarm 数据

│

├── test_results/# 输入：待检测的图像文件夹## 5. 文件结构说明```bash[3] Zhang, Guangdong, et al. "Multi-object Tracking Based on YOLOX and DeepSORT Algorithm." International Conference on 5G for Future Wireless Networks. Cham: Springer Nature Switzerland, 2022.

│   ├── UAVSwarm-02/ / UAVSwarm-12/ / UAVSwarm-44/

│   ├── UAVSwarm-*_bytetrack/          # ByteTrack 结果image_folder = r'D:\UAV\YOLOv12-BoT-SORT-ReID\data\UAVSwarm-dataset-master\test\UAVSwarm-44\img1'

│   └── UAVSwarm-*_tracking/           # BoT-SORT 结果

│

├── TrackEval/

│   ├── trackeval/# 输出：结果保存目录

│   │   ├── metrics/                   # 评估指标

│   │   │   ├── hota.py                # HOTA（推荐）output_folder = r'D:\UAV\YOLOv12-BoT-SORT-ReID\test_results\UAVSwarm-44'```python evaluate_detections.py[4] Aharon, Nir, Roy Orfaig, and Ben-Zion Bobrovsky. "Bot-sort: Robust associations multi-pedestrian tracking." arXiv preprint arXiv:2206.14651 (2022).

│   │   │   ├── clear.py               # MOTA/MOTP

│   │   │   ├── identity.py            # IDF1

│   │   │   └── track_map.py           # Track mAP

│   │   ├── datasets/                  # 数据集加载器# 置信度阈值（0-1），越低越敏感YOLOv12-BoT-SORT-ReID/

│   │   ├── eval.py                    # 评估主程序

│   │   └── plotting.py                # 结果绘图conf_threshold = 0.3

│   │

│   ├── scripts/```├── BoT-SORT/                   # BoT-SORT 算法及 YOLOv12 子模块``````

│   │   └── run_mot_challenge.py       # MOT 评估脚本

│   │

│   ├── data/

│   │   ├── gt/                        # 真实标注#### 运行推理├── data/                       # 数据集存放目录

│   │   └── trackers/                  # 跟踪结果

│   │

│   ├── docs/                          # 文档

│   │   └── MOTChallenge-format.txt    # MOT 格式```bash├── test_results/               # 推理结果保存目录*注意：请在脚本中修改 `pred_file` (预测结果) 和 `gt_file` (真实标签) 的路径。*

│   │

│   └── requirements.txt               # 依赖python test_uavswarm.py

│

├── convert_labelme_to_yolo.py         # LabelMe 转换工具```├── TrackEval/                  # 跟踪评测工具库

├── convert_uavswarm_to_yolo.py        # UAVSwarm 转换工具

├── test_uavswarm.py                   # 推理检测工具

├── evaluate_detections.py             # 评估工具

└── README.md                          # 本文档#### 输出说明├── convert_labelme_to_yolo.py  # [工具] LabelMe -> YOLO 转换</details>

```



---

脚本会输出：├── convert_uavswarm_to_yolo.py # [工具] UAVSwarm -> YOLO 转换

## 💻 使用流程示例

- **detections.txt**：MOT 格式的检测结果文件，可用于后续评估

### 流程 1：使用自定义 LabelMe 标注数据

- **vis_*.jpg**：可视化图像，便于查看检测效果├── evaluate_detections.py      # [工具] 检测结果评估## 5. 文件结构说明

```bash

# 1. 转换标注格式

python convert_labelme_to_yolo.py

**示例输出**：├── test_uavswarm.py            # [工具] 模型推理测试

# 2. 使用生成的数据训练模型

cd BoT-SORT/yolov12/```

python train.py --yaml_path ../../../uav_custom.yaml --n_epoch 100

test_results/UAVSwarm-44/└── README.md                   # 项目说明文档

# 3. 用训练好的模型推理

cd ../../..├── detections.txt         # 检测结果

python test_uavswarm.py

├── vis_000001.jpg         # 可视化结果```

# 4. 评估检测结果

python evaluate_detections.py├── vis_000002.jpg

```

├── ...```

### 流程 2：使用 UAVSwarm 数据集

└── vis_000100.jpg

```bash

# 1. 转换 MOT 格式```## 6. 参考引用

python convert_uavswarm_to_yolo.py



# 2. 训练模型

cd BoT-SORT/yolov12/**detections.txt 格式**：YOLOv12-BoT-SORT-ReID/

python train.py --yaml_path ../../../uavswarm.yaml

```

# 3. 推理

cd ../../..1,-1,123.5,98.2,45.0,52.1,0.95,-1,-1,-1本项目参考了以下开源项目：

python test_uavswarm.py

1,-1,345.8,210.3,38.5,48.9,0.87,-1,-1,-1

# 4. 评估

python evaluate_detections.py2,-1,125.3,100.1,44.5,51.8,0.92,-1,-1,-1- [YOLOv12](https://github.com/sunsmarterjie/yolov12)├── BoT-SORT/               # BoT-SORT 跟踪算法核心代码This repository provides a strong baseline for multi-UAV tracking in thermal infrared videos by leveraging YOLOv12 and BoT-SORT with ReID. Our approach significantly outperforms the widely adopted YOLOv5 with the DeepSORT pipeline, offering a high-performance foundation for UAV swarm tracking. Importantly, the established workflow in this repository can be easily integrated with any custom-trained model, extending its applicability beyond UAV scenarios. Refer to [this](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID#-quickstart-installation-and-demonstration) section for practical usage examples.

```

...

---

```- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)

## 🔧 TrackEval 评估工具



位置：`TrackEval/`

---├── data/                   # 数据集目录

### 支持的指标



- **HOTA**（推荐）：更高阶的跟踪精度

- **MOTA/MOTP**：多目标跟踪精度## 📊 结果评估├── test_results/           # 测试结果保存目录

- **IDF1**：身份保留度

- **Track mAP**：跟踪平均精度



### 使用示例**脚本**：`evaluate_detections.py`├── TrackEval/              # 评测工具



```bash

cd TrackEval/

python scripts/run_mot_challenge.py \### 功能说明├── convert_labelme_to_yolo.py  # LabelMe 格式转换脚本

    --benchmark_name MOT17 \

    --split_to_eval test \

    --tracker_folder ../submit/track3/

```该脚本对比预测结果和真实标签，计算检测精度指标：├── convert_uavswarm_to_yolo.py # UAVSwarm 数据集转换脚本<details><summary>📹 Preview - Strong Baseline</summary>



### 输出



自动生成：1. **解析文件**：读取预测结果和真实标签（都是 MOT 格式）├── evaluate_detections.py      # 检测结果评估脚本

- 详细的 CSV 结果文件

- 可视化图表2. **IOU 匹配**：计算预测框和真实框的交并比 (Intersection over Union)

- 汇总报告

3. **计算指标**：├── test_uavswarm.py            # 推理测试脚本[strong_baseline.webm](https://github.com/user-attachments/assets/702b3e80-fd3c-48f0-8032-a2a97563c19f)

---

   - **TP（真正例）**：IOU ≥ 阈值的正确检测

## 📝 常见问题

   - **FP（假正例）**：错误的检测或低于阈值的检测└── README.md                   # 项目说明文档

### Q1：推理时提示模型文件不存在？

**A**：检查 `test_uavswarm.py` 中的 `model_path` 是否正确指向 `.pt` 权重文件。   - **FN（假负例）**：未检测到的目标



### Q2：转换数据时报错 "File not found"？4. **生成报告**：输出精度、召回率、F1 分数等指标```🔗 Full video available at: [Track 3](https://youtu.be/_IiUISzCeU8?si=19JnHdwS9GLoYdtL)

**A**：确保输入路径（`base_dir`）存在，且包含相应格式的文件（JSON 或 gt.txt）。

5. **逐帧对比**：随机采样显示部分帧的详细对比

### Q3：评估结果精度很低？

**A**：可能原因：

- 模型训练不足

- 置信度阈值设置过高### 计算的指标

- 真实标注与检测框的 IOU 不匹配

## 6. 参考引用🔍 See also SOT inferences: [Track 1](https://youtu.be/HOwMRm1l124?si=ewlZ5wr1_CUDFWk_) and [Track 2](https://youtu.be/M7lSrqYkpEQ?si=EyVhfOPNRLPVzYI2)

### Q4：如何调整检测敏感度？

**A**：修改 `test_uavswarm.py` 中的 `conf_threshold`，值越低越敏感。| 指标 | 公式 | 说明 |



### Q5：如何使用 ReID 进行跟踪？|-----|------|------|

**A**：在 `BoT-SORT/tools/predict_track3.py` 中启用 `--with-reid` 选项。

| **Precision** | TP / (TP + FP) | 检测准确率：正确检测的比例 |

---

| **Recall** | TP / (TP + FN) | 检测召回率：检测到的目标比例 |本项目参考了以下开源项目：🌐 [CVPR2025](https://cvpr.thecvf.com/) | [Workshops](https://cvpr.thecvf.com/Conferences/2025/workshop-list) | [4th Anti-UAV Workshop](https://anti-uav.github.io/) | [Track-1](https://codalab.lisn.upsaclay.fr/competitions/21688) | [Track-2](https://codalab.lisn.upsaclay.fr/competitions/21690) | [Track-3](https://codalab.lisn.upsaclay.fr/competitions/21806)

## 📚 参考资源

| **F1-Score** | 2 × P × R / (P + R) | 精度和召回的调和平均数 |

- [YOLOv12](https://github.com/sunsmarterjie/yolov12) - 检测模型

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 跟踪算法| **Detection Rate** | TP / Total_GT | 目标检测率 |- [YOLOv12](https://github.com/sunsmarterjie/yolov12)

- [TrackEval](https://github.com/JonathonLuiten/TrackEval) - 评估工具

- [MOT Challenge](https://motchallenge.net/) - 多目标跟踪基准| **False Alarm Rate** | FP / Total_Pred | 误检率 |

- [LabelMe](http://labelme.csail.mit.edu/) - 标注工具

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)</details>

---

### 使用方法

## 📄 License

- [TrackEval](https://github.com/JonathonLuiten/TrackEval)

此项目代码遵循相关开源项目的许可证。

#### 配置参数

---



**最后更新**：2025 年 11 月 25 日

编辑脚本中的文件路径：

```python

# 预测结果（从 test_uavswarm.py 生成）<details><summary>📹 Preview - Single-Frame Enhancements</summary>

pred_file = r'D:\UAV\YOLOv12-BoT-SORT-ReID\test_results\UAVSwarm-44\detections.txt'

[enhancements_MultiUAV-261.webm](https://github.com/user-attachments/assets/f1dd3877-d898-45c2-93c9-26f677020e07)

# 真实标注（数据集自带）

gt_file = r'D:\UAV\YOLOv12-BoT-SORT-ReID\data\UAVSwarm-dataset-master\test\UAVSwarm-44\det\det.txt'🔗 Full video available at: [Enhancements](https://youtu.be/lkIlYCjz8r4?si=7jpgs5OAEeABNVGo)

```

</details>

#### 运行评估



```bash

python evaluate_detections.py

```<details><summary>📹 Preview - Custom Model Inference</summary>



#### 输出示例This section showcases example videos processed using a custom-trained model. The scenes are not limited to UAV footage or single-class detection. See [🚀 Quickstart: Installation and Demonstration](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID?tab=readme-ov-file#-quickstart-installation-and-demonstration) → `Run Inference Using a Custom-Trained Model` for more details.



```<details><summary>1. Multi-Class on a Walkway Scene</summary>

============================================================

📊 DETECTION EVALUATION[palace.webm](https://github.com/user-attachments/assets/cc32bda1-f461-4813-9639-eab2adfc178e)

============================================================

🔗 Original video: [palace.mp4](https://github.com/FoundationVision/ByteTrack/blob/main/videos/palace.mp4)

📁 Prediction file: ...

   Total frames: 1200</details>

   Total detections: 3456

<details><summary>2. Common Objects Underwater</summary>

📁 Ground truth file: ...

   Total frames: 1200[cou.webm](https://github.com/user-attachments/assets/59a81337-245a-49a7-817e-422536199b19)

   Total ground truth boxes: 3520

🔗 Full video available at: [COU.mp4](https://youtu.be/dZAQnpDq7NQ?si=ovF637bp4D-HZ04_)

📈 Total frames to evaluate: 1200

</details>

------------------------------------------------------------

📊 METRICS (IOU threshold: 0.5)<details><summary>3. UAVDB</summary>

------------------------------------------------------------

True Positives (TP):   3250[uavdb.webm](https://github.com/user-attachments/assets/3eff3e71-4111-4792-b4f6-4f1371843978)

False Positives (FP):  206

False Negatives (FN):  270🔗 Full video available at: [UAVDB.mp4](https://youtu.be/NOZ4yhgXF7Q?si=bPM0N3SjR6tcHH3z)



Precision: 0.9406 (3250/3456)</details>

Recall:    0.9232 (3250/3520)

F1-Score:  0.9318<details open><summary>4. NPS-Drones dataset</summary>



Detection Rate: 92.32% (3250/3520)[nps.webm](https://github.com/user-attachments/assets/78209701-f61d-480b-9bb4-c0e8697d6148)

False Alarm Rate: 5.96% (206/3456)

🔗 Full video available at: [NPS.mp4](https://youtu.be/a5jTaHiARkE?si=mIBWeIPpI1IMGF6O)

============================================================

🔍 FRAME-BY-FRAME COMPARISON</details>

============================================================

</details>

显示 15 个样本帧的对比:



Frame 000001:

  Predictions: 3 boxes

  Ground Truth: 3 boxes

  ✅ Count matches



Frame 000002:

  Predictions: 2 boxes## 🏁 Beyond Strong Baseline: Multi-UAV Tracking Competition ₊˚⊹

  Ground Truth: 3 boxes

  ⚠️  Count mismatch: 2 vs 3

  

...

```<details><summary>📹 Preview - Vision in Action: Overview of All Videos</summary>



---A complete visual overview of all training and test videos.



## 📁 项目文件结构[vision_in_action.webm](https://github.com/user-attachments/assets/f50d8e90-63b8-4b62-84ca-7e71c0750c67)



```🔗 Full video available at: [Overview](https://youtu.be/0-Sn_mxRPJw?si=xfFXvBNoQz8zxnbK)

YOLOv12-BoT-SORT-ReID/

│Scenarios are categorized to evaluate tracking performance under diverse conditions:

├── BoT-SORT/                          # BoT-SORT 算法和 YOLOv12 子模块

│   ├── yolov12/                       # YOLOv12 检测模型- **Takeoff** - UAV launch phase: 2 videos.

│   │   ├── train.py                   # 模型训练脚本- **L** - Larger UAV target: 15 videos.

│   │   ├── weights/                   # 预训练权重- **C** - Cloud background: 39 videos.

│   │   └── requirements.txt           # 依赖- **CF** - Cloud (Fewer UAVs): 18 videos.

│   ├── tracker/                       # BoT-SORT 跟踪器实现- **T** - Tree background: 68 videos.

│   ├── fast_reid/                     # ReID 特征提取模块- **TF** - Tree (Fewer UAVs): 14 videos.

│   └── requirements.txt               # 环境依赖- **B** - Scene with buildings: 11 videos.

│- **BB1** - Building Background 1: 4 videos.

├── data/                              # 数据存放目录- **BB2** - Building Background 2: 17 videos.

│   ├── images/                        # 原始图像（按日期分类）- **BB2P** - Building Background 2 (UAV partially out of view): 8 videos.

│   ├── labels/                        # 标注标签- **Landing** - UAV landing phase: 4 videos.

│   ├── uav_custom/                    # LabelMe 转换后的数据（YOLO格式）

│   ├── uavswarm_yolo/                 # UAVSwarm 转换后的数据（YOLO格式）**TOTAL: 200 videos (151,384 frames)**

│   ├── MOT/                           # MOT 数据集

│   ├── SOT/                           # SOT 数据集</details>

│   └── UAVSwarm-dataset-master/       # UAVSwarm 原始数据集

│

├── test_results/                      # 推理结果保存目录

│   ├── UAVSwarm-02/

│   ├── UAVSwarm-12/<details><summary>📹 Preview - Vision in Action: Training Videos</summary>

│   └── UAVSwarm-44/

│[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15853476.svg)](https://doi.org/10.5281/zenodo.15853476)

├── TrackEval/                         # 跟踪评估工具库

├── convert_labelme_to_yolo.py         # [工具] LabelMe → YOLO[demo_train.webm](https://github.com/user-attachments/assets/e01c0bb5-f08e-4a76-829f-7d2ea717184e)

├── convert_uavswarm_to_yolo.py        # [工具] UAVSwarm MOT → YOLO

├── test_uavswarm.py                   # [工具] 模型推理检测🔗 Full video available at: [Training Videos](https://youtu.be/rny0-nyFBk0?si=jxCPlCcHgU4zcUwU)

├── evaluate_detections.py             # [工具] 检测结果评估

└── README.md                          # 项目文档- **Takeoff** - UAV launch phase: 1 videos.

```- **L** - Larger UAV target: 8 videos.

- **C** - Cloud background: 20 videos.

---- **CF** - Cloud (Fewer UAVs): 9 videos.

- **T** - Tree background: 34 videos.

## 🚀 使用流程示例- **TF** - Tree (Fewer UAVs): 7 videos.

- **B** - Scene with buildings: 6 videos.

### 场景 1：使用 LabelMe 标注的自定义数据- **BB1** - Building Background 1: 2 videos.

- **BB2** - Building Background 2: 9 videos.

```bash- **BB2P** - Building Background 2 (UAV partially out of view): 4 videos.

# 1. 转换标注格式- **Landing** - UAV landing phase: 2 videos.

python convert_labelme_to_yolo.py

**TOTAL: 102 videos (77,293 frames)**

# 2. 使用生成的 uav_custom 数据集训练模型

cd BoT-SORT/yolov12/</details>

python train.py  # 需自行配置训练参数



# 3. 用训练好的模型进行推理

cd ../../

python test_uavswarm.py<details><summary>📹 Preview - Vision in Action: Test Videos</summary>



# 4. 评估检测结果[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16299533.svg)](https://doi.org/10.5281/zenodo.16299533)

python evaluate_detections.py

```[demo_test.webm](https://github.com/user-attachments/assets/15e9143e-303f-4ef1-849d-735f8763e112)



### 场景 2：使用 UAVSwarm 数据集🔗 Full video available at: [Test Videos](https://youtu.be/G_8fE9njTRs?si=xUJjaYYC3D81m3Na)



```bash- **Takeoff** - UAV launch phase: 1 videos.

# 1. 转换 MOT 格式到 YOLO 格式- **L** - Larger UAV target: 7 videos.

python convert_uavswarm_to_yolo.py- **C** - Cloud background: 19 videos.

- **CF** - Cloud (Fewer UAVs): 9 videos.

# 2. 使用生成的 uavswarm_yolo 数据集训练模型- **T** - Tree background: 34 videos.

cd BoT-SORT/yolov12/- **TF** - Tree (Fewer UAVs): 7 videos.

python train.py- **B** - Scene with buildings: 5 videos.

- **BB1** - Building Background 1: 2 videos.

# 3. 对测试集进行推理- **BB2** - Building Background 2: 8 videos.

cd ../../- **BB2P** - Building Background 2 (UAV partially out of view): 4 videos.

python test_uavswarm.py- **Landing** - UAV landing phase: 2 videos.



# 4. 评估推理结果**TOTAL: 98 videos (74,538 frames)**

python evaluate_detections.py

```</details>



---



## 📝 关键代码说明

<details open><summary>📹 Preview - Vision in Action: Beyond Strong Baseline</summary>

### MOT 格式

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16458805.svg)](https://doi.org/10.5281/zenodo.16458805)

MOT Challenge 格式是多目标跟踪的标准格式：

[<img src="https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/assets/beyond_strong_baseline.png" width="100%">](https://www.codabench.org/competitions/9888/)

```

frame_id, track_id, x, y, w, h, conf, class_id, visibility, [occlusion][<img src="https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/assets/beyond_strong_baseline_strong_baseline.png" width="100%">](https://www.codabench.org/competitions/9888/#/results-tab)

1, 1, 123.5, 98.2, 45.0, 52.1, 1, 1, 0.9, 0

2, 1, 125.3, 100.1, 44.5, 51.8, 1, 1, 0.95, 0🔗 View the competition on [Codabench](https://www.codabench.org/competitions/9888/)

```

</details>

| 字段 | 说明 | 范围 |

|-----|------|------|

| frame_id | 帧号 | 正整数 |

| track_id | 目标 ID（跟踪时使用） | 正整数 |

| x, y | 边界框左上角坐标 | 像素坐标 |### Participation

| w, h | 边界框宽高 | 像素坐标 |

| conf | 置信度 | -1 (忽略) 或 0-1 |<details><summary>Performance</summary>

| class_id | 类别 ID | 通常为 1（UAV类） |

| visibility | 可见度 | 0-1 |[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17089400.svg)](https://doi.org/10.5281/zenodo.17089400)



### YOLO 格式[ViA_MultiUAV-261.webm](https://github.com/user-attachments/assets/dda89d21-7c25-4e33-b0cf-ab8fa126ac00)



YOLO 使用归一化的相对坐标：🔗 Full video available at: [Performance.mp4](https://youtu.be/uj-eFWOG9RU?si=BGWluZ9q2K1f0wwG)



```#### Public Leaderboard Phase

class_id x_center y_center width height

0 0.5 0.5 0.3 0.4| Methods                          | HOTA     | MOTA     | IDF1     |

```| :------------------------------: | :------: | :------: | :------: |

| Strong Baseline (SB)             | 0.873908 | 0.628351 | 0.717146 |

所有坐标归一化到 [0, 1] 范围内，便于模型训练。| SB + CLAHE                       | 0.836414 | 0.626376 | 0.686967 |

| SB + Sobel-based Image Gradients | 0.823678 | 0.634651 | 0.680124 |

---| SB + Sobel-based Edge Sharpening | 0.831300 | 0.609124 | 0.680843 |

| [TransVisDrone](https://github.com/tusharsangam/TransVisDrone) | 0.818562 | 0.602384 | 0.683446 |

## 🔧 常见问题

</details>

### Q1：推理时提示模型文件不存在？

**A**：检查 `test_uavswarm.py` 中的 `model_path` 是否正确指向 `.pt` 权重文件。<details><summary>Interpolation</summary>



### Q2：转换数据时报错 "File not found"？Interpolation commands for this competition. Example usage:

**A**：确保输入路径（`base_dir`）存在，且包含相应格式的文件（JSON 或 gt.txt）。

```bash

### Q3：评估结果精度很低？# input and output are both folders containing .txt files

**A**：可能原因包括：$ python tools/pre_interpolation.py --input ./submission --output ./pre_submission

- 模型训练不足$ python tools/interpolation.py --txt_path ./pre_submission --save_path ./mid_submission

- 置信度阈值设置过高$ python tools/post_interpolation.py --input ./mid_submission --output ./post_submission

- 真实标注与检测框的 IOU 计算方式不匹配```



### Q4：如何调整检测敏感度？#### Public Leaderboard Phase

**A**：修改 `test_uavswarm.py` 中的 `conf_threshold` 参数，值越低越敏感。

| Methods             | HOTA     | MOTA     | IDF1     |

---| :-----------------: | :------: | :------: | :------: |

| TransVisDrone (TVD) | 0.818562 | 0.602384 | 0.683446 |

## 📚 参考资源| TVD + Interpolation | 0.832675 | 0.611150 | 0.689753 |



- [YOLOv12](https://github.com/sunsmarterjie/yolov12) - 检测模型</details>

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 跟踪算法

- [MOT Challenge](https://motchallenge.net/) - 多目标跟踪基准

- [LabelMe](http://labelme.csail.mit.edu/) - 标注工具



---



## 📄 License



此项目代码遵循相关开源项目的许可证。## 🗞️ News


## 🚀 Quickstart: Installation and Demonstration

[![Colab Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x5T6woUdV6dD_T6qdYcKG04Q2iVVHGoD?usp=sharing)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/yuhsi44165/yolov12-bot-sort/)

[![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://medium.com/@scofield44165/ubuntu-24-04-1-getting-started-with-yolov12-bot-sort-reid-on-linux-20826ffc8224)
[![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)](https://medium.com/@scofield44165/macos-tahoe-26-0-1-getting-started-with-yolov12-bot-sort-reid-on-mac-f87400d5b096)
[![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://medium.com/@scofield44165/windows-11-getting-started-with-yolov12-bot-sort-reid-on-windows-11-24ee1f1cd513)

<details><summary>Installation</summary>

```bash
$ conda create -n yolov12_botsort python=3.11 -y
$ conda activate yolov12_botsort
$ git clone https://github.com/wish44165/YOLOv12-BoT-SORT-ReID.git
$ cd YOLOv12-BoT-SORT-ReID/BoT-SORT/yolov12/
$ wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# Install dependencies (choose one):
$ pip install -r requirements.txt        # Linux
$ pip install -r requirements_mac.txt    # macOS
$ pip install -r requirements_win.txt    # Windows
$ cd ../
$ pip install torch torchvision torchaudio
$ pip install -r requirements.txt
$ pip install ultralytics
$ pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
$ pip install cython_bbox
$ pip install faiss-cpu
$ pip install seaborn
```

</details>


<details><summary>Folder Structure</summary>

The following folder structure will be created upon cloning this repository.

```
YOLOv12-BoT-SORT-ReID/
├── data/
│   └── demo/
│       ├── MOT/
│       │   ├── MultiUAV-003.mp4
│       │   ├── Test_imgs/
│       │   │   ├── MultiUAV-003/
│       │   │   ├── MultiUAV-135/
│       │   │   ├── MultiUAV-173/
│       │   │   └── MultiUAV-261/
│       │   └── TestLabels_FirstFrameOnly/
│       │       ├── MultiUAV-003.txt
│       │       ├── MultiUAV-135.txt
│       │       ├── MultiUAV-173.txt
│       │       └── MultiUAV-261.txt
│       └── SOT/
│           ├── Track1/
│           │   ├── 20190926_111509_1_8/
│           │   ├── 41_1/
│           │   ├── new30_train-new/
│           │   └── wg2022_ir_050_split_01/
│           └── Track2/
│               ├── 02_6319_0000-1499/
│               ├── 3700000000002_110743_1/
│               ├── DJI_0057_1/
│               └── wg2022_ir_032_split_04/
└── BoT-SORT/
```

</details>


<details><summary>Demonstration</summary>

Toy example with three tracks, including SOT and MOT.

```bash
$ cd BoT-SORT/

# Track 1
$ python tools/predict_track1.py --weights ./yolov12/weights/v1/SOT_yolov12l.pt --source ../data/demo/SOT/Track1/ --img-size 640 --device "0" --conf-thres 0.01 --iou-thres 0.01 --track_high_thresh 0.1 --track_low_thresh 0.01 --fuse-score --agnostic-nms --min_box_area 4 --save_path_answer ./submit/track1/demo --hide-labels-name
# output: ./runs/detect/, ./submit/track1/demo/

# Track 2
$ python tools/predict_track2.py --weights ./yolov12/weights/v1/SOT_yolov12l.pt --source ../data/demo/SOT/Track2/ --img-size 640 --device "0" --conf-thres 0.01 --iou-thres 0.01 --track_high_thresh 0.1 --track_low_thresh 0.01 --fuse-score --agnostic-nms --min_box_area 1 --save_path_answer ./submit/track2/demo --hide-labels-name
# output: ./runs/detect/, ./submit/track2/demo/

# Track 3
$ python tools/predict_track3.py --weights ./yolov12/weights/v1/MOT_yolov12n.pt --source ../data/demo/MOT/ --img-size 1600 --device "0" --track_buffer 60 --save_path_answer ./submit/track3/demo --hide-labels-name
$ python tools/predict_track3.py --weights ./yolov12/weights/v1/MOT_yolov12n.pt --source ../data/demo/MOT/ --img-size 1600 --device "0" --track_buffer 60 --save_path_answer ./submit/track3/demo --with-reid --fast-reid-config logs/sbs_S50/config.yaml --fast-reid-weights logs/sbs_S50/model_0016.pth --hide-labels-name
# output: ./runs/detect/, ./submit/track3/demo/

# Heatmap
$ cd yolov12/
$ python heatmap.py
# output: ./outputs/
```

</details>


<details><summary>Run Inference on Custom Data</summary>

This project supports flexible inference on image folders and video files, with or without initial object positions, specifically for MOT task.

```bash
python tools/inference.py \
    --weights ./yolov12/weights/ViA_yolov12n.pt \
    --source <path to folder or video> \
    --with-initial-positions \
    --initial-position-config <path to initial positions file (optional)> \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name
```

Below are examples of supported inference settings:

```bash
# 1. Inference on Image Folder (without initial position)
python tools/inference.py \
    --weights ./yolov12/weights/ViA_yolov12n.pt \
    --source ../data/demo/MOT/Test_imgs/MultiUAV-003/ \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name

# 2. Inference on Image Folder (with initial position)
python tools/inference.py \
    --weights ./yolov12/weights/ViA_yolov12n.pt \
    --source ../data/demo/MOT/Test_imgs/MultiUAV-003/ \
    --with-initial-positions \
    --initial-position-config ../data/demo/MOT/TestLabels_FirstFrameOnly/MultiUAV-003.txt \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name

# 3. Inference on Video (without initial position)
python tools/inference.py \
    --weights ./yolov12/weights/ViA_yolov12n.pt \
    --source ../data/demo/MOT/MultiUAV-003.mp4 \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name

# 4. Inference on Video (with initial position)
python tools/inference.py \
    --weights ./yolov12/weights/ViA_yolov12n.pt \
    --source ../data/demo/MOT/MultiUAV-003.mp4 \
    --with-initial-positions \
    --initial-position-config ../data/demo/MOT/TestLabels_FirstFrameOnly/MultiUAV-003.txt \
    --img-size 1600 \
    --track_buffer 60 \
    --device "0" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/ \
    --with-reid \
    --fast-reid-config logs/sbs_S50/config.yaml \
    --fast-reid-weights logs/sbs_S50/model_0016.pth \
    --hide-labels-name
```

</details>


<details><summary>Run Inference Using a Custom Trained Model</summary>

This project also supports flexible inference using a custom-trained model for any MOT task. Below are the instructions for reproducing the preview section.

```bash
$ cd BoT-SORT/
```

### 1. Multi-Class on a Walkway Scene

```bash
$ wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt
$ wget https://github.com/FoundationVision/ByteTrack/raw/main/videos/palace.mp4
$ python tools/inference.py \
    --weights yolov12x.pt \
    --source palace.mp4 \
    --img-size 640 \
    --device "0" \
    --save_path_answer ./submit/palace/
```

### 2. Common Objects Underwater

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15828323.svg)](https://doi.org/10.5281/zenodo.15828323)

```bash
for f in ./videos/COU/*.mp4; do
    python tools/inference.py \
        --weights ./yolov12/runs/det/train/weights/best.pt \
        --source "$f" \
        --img-size 1600 \
        --device "0" \
        --save_path_answer ./submit/COU/
done
```

### 3. UAVDB

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16342697.svg)](https://doi.org/10.5281/zenodo.16342697)

```bash
for f in ./videos/UAVDB/*.mp4; do
    python tools/inference.py \
        --weights ./yolov12/runs/det/train/weights/best.pt \
        --source "$f" \
        --img-size 1600 \
        --device "0" \
        --save_path_answer ./submit/UAVDB/
done
```

### 4. NPS-Drones dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16891919.svg)](https://doi.org/10.5281/zenodo.16891919)

```bash
for f in ./videos/NPS/*.mp4; do
    python tools/inference.py \
        --weights ./yolov12/runs/det/train/weights/best.pt \
        --source "$f" \
        --img-size 1600 \
        --device "0" \
        --save_path_answer ./submit/NPS/
done
```

</details>


<details><summary>Run Inference on macOS</summary>

This project also supports running inference on macOS. However, for efficiency reasons, performing both training and inference on a GPU is still recommended.

When running on macOS, the following limitations apply:

1. No GPU or MPS acceleration (CPU only).
2. ReID is not supported.
3. Initial position is not supported.

Below are two examples of running inference on macOS.

```bash
# 1. Inference on Multi-Class on a Walkway Scene
$ wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt
$ wget https://github.com/FoundationVision/ByteTrack/raw/main/videos/palace.mp4
$ python tools/inference.py \
    --weights yolov12x.pt \
    --source palace.mp4 \
    --img-size 640 \
    --device "cpu" \
    --save_path_answer ./submit/palace/

# 2. Inference on MultiUAV Video
python tools/inference.py \
    --weights ./yolov12/weights/ViA_yolov12n.pt \
    --source ../data/demo/MOT/MultiUAV-003.mp4 \
    --img-size 1600 \
    --track_buffer 60 \
    --device "cpu" \
    --agnostic-nms \
    --save_path_answer ./submit/inference/
```

Inference time comparison for the two examples on GPU (Ubuntu) and CPU (macOS).

<img src="https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/assets/inference_ubuntu_mac.png" width="100%">

</details>








## 🛠️ Implementation Details


<details><summary>Hardware Information</summary>

Experiments were conducted on two platforms: (1) a local system with an Intel Core i7-12650H CPU, NVIDIA RTX 4050 GPU, and 16 GB RAM for data processing and inference, and (2) an HPC system with an NVIDIA H100 GPU and 80 GB memory for model training.

### Laptop

<a href="https://github.com/wish44165/wish44165/tree/main/assets"><img src="https://github.com/wish44165/wish44165/blob/main/assets/msi_Cyborg_15_A12VE_badge.svg" alt="Spartan"></a> 

- CPU: Intel® Core™ i7-12650H
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6GB)
- RAM: 23734MiB

### HPC

<a href="https://dashboard.hpc.unimelb.edu.au/"><img src="https://github.com/wish44165/wish44165/blob/main/assets/unimelb_spartan.svg" alt="Spartan"></a> 

- GPU: Spartan gpu-h100 (80GB), gpu-a100 (80GB)
  
</details>




### 🖻 Data Preparation


<details><summary>Officially Released</summary>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15103888.svg)](https://doi.org/10.5281/zenodo.15103888)

```
4th_Anti-UAV_Challenge/
├── baseline/
│   ├── Baseline_code.zip
│   └── MultiUAV_Baseline_code_and_submissi.zip
├── test/
│   ├── MultiUAV_Test.zip
│   ├── track1_test.zip
│   └── track2_test.zip
└── train/
    ├── MultiUAV_Train.zip
    └── train.zip
```

- Train
    - Track 1 & Track 2: [Google Drive](https://drive.google.com/drive/folders/1hEGq14WnfPstYrI_9OgscR1VsWc5_XDl) | [Baidu](https://pan.baidu.com/s/1rtZ_PkYX__Bt2O5MgTj1tg?pwd=CVPR)
    - Track 3: [Google Drive](https://drive.google.com/drive/folders/1JvGdAJjGzjOIGMG82Qiz5YJKzjy8VOd-?usp=drive_link) | [Baidu](https://pan.baidu.com/s/19iVwI1MW9OdXyPIc0xBSjQ?from=init&pwd=CVPR)
- Test
    - Track 1: [Google Drive](https://drive.google.com/drive/folders/1qkUeglLk9-OXniIUVh1r7OljDLwDNhBs?usp=sharing) | [Baidu](https://pan.baidu.com/s/13HFq5P0gWrdlBerFZBKbuA?pwd=cvpr)
    - Track 2: [Google Drive](https://drive.google.com/drive/folders/1qkUeglLk9-OXniIUVh1r7OljDLwDNhBs?usp=sharing) | [Baidu](https://pan.baidu.com/s/1s7KkyjgXP1v495EULqwoew?pwd=cvpr)
    - Track 3: [Google Drive](https://drive.google.com/drive/folders/1cfF00w_3ewUMELSSnmaYOKLTZoIWlxbF?usp=sharing) | [Baidu](https://pan.baidu.com/s/1rhB24tksTw1JW6ZltOSvOg?pwd=CVPR)

</details>


<details><summary>Strong Baseline</summary>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15203123.svg)](https://doi.org/10.5281/zenodo.15203123)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/wish44165/StrongBaseline_YOLOv12-BoT-SORT-ReID) 

```
train/
├── MOT/
│   └── AntiUAV_train_val.zip
├── ReID/
│   ├── MOT20_subset.zip
│   └── MOT20.zip
└── SOT/
    ├── AntiUAV_train_val_test.zip
    └── AntiUAV_train_val.zip
```

</details>


<details><summary>Enhancements</summary>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15276582.svg)](https://doi.org/10.5281/zenodo.15276582)

```
enhancements/
├── MOT/
│   ├── CLAHE_train_val.zip
│   ├── Sobel-based_Edge_Sharpening_train_val.zip
│   └── Sobel-based_Image_Gradients_train_val.zip
└── ReID/
    ├── CLAHE_subset.zip
    ├── Sobel-based_Edge_Sharpening_subset.zip
    └── Sobel-based_Image_Gradients_subset.zip
```

</details>




### 📂 Folder Structure

<details><summary>Project Layout</summary>

Follow the folder structure below to ensure smooth execution and easy navigation.

```
YOLOv12-BoT-SORT-ReID/
├── BoT-SORT/
│   ├── getInfo.py
│   ├── datasets/
│   │   └── README.md
│   ├── fast_reid/
│   │   └── datasets/
│   │       ├── generate_mot_patches.py
│   │       └── README.md
│   ├── logs/
│   │   ├── sbs_S50/
│   │   │   ├── config.yaml
│   │   │   └── model_0016.pth
│   │   └── README.md
│   ├── requirements.txt
│   ├── runs/
│   │   └── README.md
│   ├── submit/
│   │   └── README.md
│   ├── tools/
│   │   ├── predict_track1.py
│   │   ├── predict_track2.py
│   │   └── predict_track3.py
│   └── yolov12/
│       ├── heatmap.py
│       ├── imgs_dir/
│       │   ├── 00096.jpg
│       │   ├── 00379.jpg
│       │   ├── 00589.jpg
│       │   └── 00643.jpg
│       ├── requirements.txt
│       └── weights/
│           ├── v1/
│           │   ├── MOT_yolov12n.pt
│           │   └── SOT_yolov12l.pt
│           └── ViA_yolov12n.pt
├── data/
│   ├── demo/
│   ├── MOT/
│   │   └── README.md
│   └── SOT/
│       └── README.md
├── LICENSE
└── README.md
```

</details>




### 🔨 Reproduction

<details><summary>Run Commands</summary>

Executing the following commands can reproduce the leaderboard results.

<details><summary>Data Analysis</summary>

```bash
$ cd BoT-SORT/

# Table 1
$ python getInfo.py
```

</details>

<details><summary>Train YOLOv12</summary>

Refer to the [README](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/tree/main/data/MOT#readme) for more information.

```bash
$ cd BoT-SORT/yolov12/

# Run training with default settings
$ python train.py
```

</details>

<details><summary>Train BoT-SORT-ReID</summary>

Refer to the [README](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/tree/main/BoT-SORT/fast_reid/datasets#readme) for more information.

```bash
$ cd BoT-SORT/

# Train with final config
$ python fast_reid/tools/train_net.py --config-file ./logs/sbs_S50/config.yaml MODEL.DEVICE "cuda:0"
```

</details>

<details><summary>Inference</summary>

```bash
$ cd BoT-SORT/

# Track 1
$ python tools/predict_track1.py --weights ./yolov12/weights/v1/SOT_yolov12l.pt --source ../data/SOT/track1_test/ --img-size 640 --device "0" --conf-thres 0.01 --iou-thres 0.01 --track_high_thresh 0.1 --track_low_thresh 0.01 --fuse-score --agnostic-nms --min_box_area 4 --save_path_answer ./submit/track1/test --hide-labels-name
# output: ./runs/detect/, ./submit/track1/test/

# Track 2
$ python tools/predict_track2.py --weights ./yolov12/weights/v1/SOT_yolov12l.pt --source ../data/SOT/track2_test/ --img-size 640 --device "0" --conf-thres 0.01 --iou-thres 0.01 --track_high_thresh 0.1 --track_low_thresh 0.01 --fuse-score --agnostic-nms --min_box_area 1 --save_path_answer ./submit/track2/test --hide-labels-name
# output: ./runs/detect/, ./submit/track2/test/

# Track 3
$ chmod +x run_track3.sh
$ ./run_track3.sh
# output: ./runs/detect/, ./submit/track3/test/
```

</details>

</details>








## ✨ Models

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/wish44165/YOLOv12-BoT-SORT-ReID) 

| Model                                                                                | size<br><sup>(pixels) | AP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(G) | Note |
| :----------------------------------------------------------------------------------- | :-------------------: | :-------------------:| :-----------------:| :---------------:| :----: |
| [SOT_yolov12l.pt](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/BoT-SORT/yolov12/weights/v1/SOT_yolov12l.pt) | 640                   | 67.2                 | 26.3                | 88.5               |
| [MOT_yolov12n.pt](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/blob/main/BoT-SORT/yolov12/weights/v1/MOT_yolov12n.pt) ([ReID](https://huggingface.co/wish44165/YOLOv12-BoT-SORT-ReID/tree/main)) | 1600                   | 68.5                 | 2.6                | 6.3              | [#4 (Comment)](https://github.com/wish44165/YOLOv12-BoT-SORT-ReID/issues/4#issuecomment-2959336418) |








## 📜 Citation

If you find this project helpful for your research or applications, we would appreciate it if you could cite the paper and give it a star.

```
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Yu-Hsi},
    title     = {Strong Baseline: Multi-UAV Tracking via YOLOv12 with BoT-SORT-ReID},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {6573-6582}
}
```

<a href="https://www.star-history.com/#wish44165/YOLOv12-BoT-SORT-ReID&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=wish44165/YOLOv12-BoT-SORT-ReID&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=wish44165/YOLOv12-BoT-SORT-ReID&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=wish44165/YOLOv12-BoT-SORT-ReID&type=Date" />
 </picture>
</a>








## 🙏 Acknowledgments

Much of the code builds upon [YOLOv12](https://github.com/sunsmarterjie/yolov12), [BoT-SORT](https://github.com/NirAharon/BoT-SORT), and [TrackEval](https://github.com/JonathonLuiten/TrackEval). We also sincerely thank the organizers of the [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) benchmark for providing the valuable dataset. We greatly appreciate their contributions!