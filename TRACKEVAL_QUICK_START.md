# 🎯 TrackEval 评估完整指南（已修复）

## 📋 准备步骤

### 步骤 1: 清理并准备数据

运行准备脚本（一次性）：
```batch
cd /d d:\UAV\YOLOv12-BoT-SORT-ReID
prepare_only.bat
```

这个脚本会：
- ✅ 清理旧的 TrackEval 结构
- ✅ 生成所有序列的 `seqinfo.ini` 文件
- ✅ 生成 `MOT17-test.txt` seqmap 文件

**输出应该显示**：
```
✓ UAVSwarm-02        : 1200 frames @ 1920x1080
✓ UAVSwarm-04        : 1200 frames @ 1920x1080
...
✓ Created seqmap: ...
  Total sequences: 35
```

### 步骤 2: 运行完整评估

```batch
run_evaluation.bat
```

这会：
1. 再次运行准备脚本（确保所有文件就位）
2. 复制追踪结果和 GT 文件到 TrackEval 结构
3. 运行评估并输出 HOTA/MOTA/IDF1 指标

## 📁 预期的目录结构

成功后应该是这样：
```
TrackEval/data/
├── gt/mot_challenge/UAVSwarm/
│   ├── UAVSwarm-02/
│   │   └── seqinfo.ini
│   ├── UAVSwarm-04/
│   │   └── seqinfo.ini
│   └── gt/
│       ├── UAVSwarm-02.txt
│       ├── UAVSwarm-04.txt
│       └── seqmaps/
│           └── MOT17-test.txt
└── trackers/mot_challenge/UAVSwarm/BoTSORT/data/
    ├── UAVSwarm-02.txt
    ├── UAVSwarm-04.txt
    └── ...
```

## 🚀 快速开始（完整流程）

```batch
cd /d d:\UAV\YOLOv12-BoT-SORT-ReID

REM 第一次使用：先准备数据
prepare_only.bat

REM 然后运行评估
run_evaluation.bat
```

## 📊 输出示例

```
✓ UAVSwarm-02        :  850 frames @ 1920x1080
✓ UAVSwarm-04        :  900 frames @ 1920x1080
...
✓ Created seqmap: TrackEval\data\gt\mot_challenge\UAVSwarm\seqmaps\MOT17-test.txt
  Total sequences: 35

================================================================================
TRACKING EVALUATION RESULTS
================================================================================
UAVSwarm-02          | HOTA: 0.5234 | MOTA: 0.6123 | IDF1: 0.7145
UAVSwarm-04          | HOTA: 0.5891 | MOTA: 0.6845 | IDF1: 0.7523
...
AVERAGE              | HOTA: 0.5542 | MOTA: 0.6484 | IDF1: 0.7334
================================================================================
```

## ⚙️ 故障排除

### 问题：`seqinfo.ini not found`

**解决方案**：
1. 运行 `prepare_only.bat` 
2. 检查 `TrackEval\data\gt\mot_challenge\UAVSwarm\` 目录是否包含所有序列子目录

### 问题：`seqmap not found`

**解决方案**：
确保 `MOT17-test.txt` 在：
```
TrackEval\data\gt\mot_challenge\UAVSwarm\seqmaps\MOT17-test.txt
```

### 问题：追踪结果找不到

**解决方案**：
确保 `test_results/inference_answers/` 中有以下结构：
```
test_results/inference_answers/
├── UAVSwarm-02/
│   └── UAVSwarm-02.txt
├── UAVSwarm-04/
│   └── UAVSwarm-04.txt
└── ...
```

## 🔍 验证文件结构

运行这些命令检查是否正确设置：

```batch
REM 检查 seqinfo.ini
dir /s TrackEval\data\gt\mot_challenge\UAVSwarm\UAVSwarm-02\

REM 检查 GT 文件
dir TrackEval\data\gt\mot_challenge\UAVSwarm\gt\

REM 检查 seqmap
type TrackEval\data\gt\mot_challenge\UAVSwarm\seqmaps\MOT17-test.txt

REM 检查追踪结果
dir /s test_results\inference_answers\UAVSwarm-02\
```

## 📈 后续步骤

1. **比较不同追踪器**：
   ```bash
   python evaluate_tracking_results.py --tracker-name ByteTrack
   ```

2. **分析低分序列**：找出哪些序列评分低，优化模型参数

3. **导出详细报告**：查看 `evaluation_results_BoTSORT.json` 获取完整数据

## 相关文件

- `prepare_trackeval.py` - 生成元数据
- `evaluate_tracking_results.py` - 运行评估
- `prepare_only.bat` - 仅准备数据
- `run_evaluation.bat` - 完整评估流程
