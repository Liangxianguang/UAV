# TrackEval 评估指南

## 概述

这个脚本使用 **TrackEval** 框架来评估 BoT-SORT 追踪结果与地面真值（GT）之间的性能差异，计算 HOTA、MOTA、IDF1 等指标。

## 文件结构

```
YOLOv12-BoT-SORT-ReID/
├── test_results/
│   └── inference_answers/          # 追踪结果（由 run_batch_inference.bat 生成）
│       ├── UAVSwarm-02/
│       │   └── UAVSwarm-02.txt     # 追踪结果
│       ├── UAVSwarm-04/
│       │   └── UAVSwarm-04.txt
│       └── ...
├── data/
│   └── UAVSwarm-dataset-master/
│       └── test/                   # 测试集
│           ├── UAVSwarm-02/
│           │   └── gt/
│           │       └── gt.txt      # 地面真值
│           ├── UAVSwarm-04/
│           │   └── gt/
│           │       └── gt.txt
│           └── ...
├── evaluate_tracking_results.py     # 评估脚本
└── run_evaluation.bat               # Windows 批处理脚本（推荐）
```

## 使用方法

### 方法 1：Windows 批处理脚本（最简单）

```batch
cd /d d:\UAV\YOLOv12-BoT-SORT-ReID
run_evaluation.bat
```

**优点**：
- 一键运行，无需参数
- 自动设置所有参数
- 自动复制 GT 文件和追踪结果到 TrackEval 格式

### 方法 2：Python 脚本（更灵活）

```bash
# 使用默认参数评估 BoTSORT
python evaluate_tracking_results.py

# 指定自定义追踪器名称
python evaluate_tracking_results.py --tracker-name ByteTrack

# 自定义数据路径
python evaluate_tracking_results.py \
  --tracking-dir custom/results \
  --gt-dir custom/gt_folder

# 指定要计算的指标
python evaluate_tracking_results.py \
  --metrics HOTA CLEAR Identity VACE

# 跳过目录设置（使用已存在的 TrackEval 结构）
python evaluate_tracking_results.py --no-setup
```

## 可用参数

```
--dataset-name NAME          数据集名称（默认：UAVSwarm）
--tracker-name NAME          追踪器名称（默认：BoTSORT）
--tracking-dir PATH          追踪结果目录（默认：test_results/inference_answers）
--gt-dir PATH               GT 文件目录（默认：data/UAVSwarm-dataset-master/test）
--metrics NAME1 NAME2 ...   要计算的指标（默认：HOTA CLEAR Identity）
--split SPLIT               数据集划分：train/test/all（默认：test）
--no-setup                  跳过目录设置步骤
```

## 输出结果

评估完成后，会生成以下输出：

### 1. 控制台输出
```
================================================================================
TRACKING EVALUATION RESULTS
================================================================================
UAVSwarm-02          | HOTA: 0.5234 | MOTA: 0.6123 | IDF1: 0.7145
UAVSwarm-04          | HOTA: 0.5891 | MOTA: 0.6845 | IDF1: 0.7523
...
--------------------------------------------------------------------------------
AVERAGE              | HOTA: 0.5542 | MOTA: 0.6484 | IDF1: 0.7334
================================================================================
```

### 2. JSON 文件输出
```
evaluation_results_BoTSORT.json
{
  "tracker": "BoTSORT",
  "dataset": "UAVSwarm",
  "hota": 0.5542,
  "mota": 0.6484,
  "idf1": 0.7334,
  "num_sequences": 35
}
```

## 指标说明

| 指标 | 全称 | 说明 |
|------|------|------|
| **HOTA** | Higher Order Tracking Accuracy | 综合追踪准确度，考虑检测与追踪性能 |
| **MOTA** | Multiple Object Tracking Accuracy | 多目标追踪准确度 |
| **IDF1** | ID F1 Score | 身份一致性指标 |
| **CLEAR** | CLEAR MOT 指标组 | 包含精度（MOTP）、召回率等 |

## 工作流程

### 步骤 1：生成追踪结果
```bash
cd BoT-SORT
run_batch_inference.bat
```
这会在 `test_results/inference_answers/` 中生成 `UAVSwarm-XX/UAVSwarm-XX.txt` 文件。

### 步骤 2：评估结果
```bash
cd ..
run_evaluation.bat
```
或
```bash
python evaluate_tracking_results.py
```

### 步骤 3：查看结果
- 控制台输出：实时显示每个序列的指标
- JSON 文件：`evaluation_results_BoTSORT.json`

## 常见问题

### Q: 如何评估多个追踪器的结果？

```bash
# 评估 BoTSORT
python evaluate_tracking_results.py --tracker-name BoTSORT

# 评估 ByteTrack
python evaluate_tracking_results.py --tracker-name ByteTrack

# 两个结果会并行显示在 TrackEval 输出中
```

### Q: 如何只评估特定序列？

需要在 TrackEval 目录中手动删除其他序列的文件，或者修改 `evaluate_tracking_results.py` 中的 `setup_trackeval_structure` 函数。

### Q: 结果文件夹在哪里？

TrackEval 评估数据位置：
- GT: `TrackEval/data/gt/mot_challenge/UAVSwarm/gt/`
- 追踪结果: `TrackEval/data/trackers/mot_challenge/UAVSwarm/BoTSORT/data/`

### Q: 如何修改评估指标？

编辑 `run_evaluation.bat` 或使用 Python 命令的 `--metrics` 参数：
```bash
python evaluate_tracking_results.py --metrics HOTA CLEAR Identity VACE
```

## TrackEval 格式说明

### 追踪结果格式（MOT Challenge）
```
<frame>, <id>, <x1>, <y1>, <w>, <h>, <conf>, <class>, <visibility>, <feature_ratio>
```

示例：
```
1, 1, 100.5, 200.3, 50, 80, 1, 1, 1, 0
2, 1, 101.2, 201.5, 50, 80, 1, 1, 1, 0
2, 2, 300.0, 150.0, 45, 75, 1, 1, 1, 0
```

### GT 格式
与追踪结果格式相同（由 `inference.py` 保证）。

## 调试技巧

### 1. 检查文件是否正确复制
```bash
# 检查 GT 文件
dir TrackEval\data\gt\mot_challenge\UAVSwarm\gt\

# 检查追踪结果
dir TrackEval\data\trackers\mot_challenge\UAVSwarm\BoTSORT\data\
```

### 2. 查看详细日志
修改 `evaluate_tracking_results.py` 中的：
```python
default_eval_config['DISPLAY_LESS_PROGRESS'] = True  # 显示更多进度信息
```

### 3. 验证数据格式
使用 Python 检查文件格式：
```python
with open('test_results/inference_answers/UAVSwarm-02/UAVSwarm-02.txt') as f:
    for line in f:
        parts = line.strip().split(',')
        print(f"Frame: {parts[0]}, ID: {parts[1]}, Bbox: {parts[2:6]}")
```

## 下一步

- 比较不同追踪器的性能
- 分析哪些序列评分较低
- 优化模型参数以提高追踪性能
