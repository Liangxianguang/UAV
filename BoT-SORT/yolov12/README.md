# YOLOv12 + Trackers 快速使用（ByteTrack / OC-SORT）

本目录提供两个便捷运行脚本：
- `run_bytetrack.py`：使用 ByteTrack 进行多目标跟踪
- `run_ocsort.py`：使用 OC-SORT 进行多目标跟踪

两者均基于本仓库内的 Ultralytics 集成（无需额外安装外部跟踪库）。

---

## 1. 准备

- Python 环境已安装本仓库依赖，GPU 环境可选（`--device 0` 使用 CUDA）。
- 权重文件（示例）：`.\nweights\ViA_yolov12n.pt`
- 测试视频（示例）：`D:\UAV\YOLOv12-BoT-SORT-ReID\data\demo\MOT\MultiUAV-003.mp4`

可选：在命令行运行前进入目录
```cmd
cd D:\UAV\YOLOv12-BoT-SORT-ReID\BoT-SORT\yolov12
```

---

## 2. 运行 ByteTrack

默认使用 `ultralytics\cfg\trackers\bytetrack.yaml`。

```cmd
python run_bytetrack.py ^
  --weights .\weights\ViA_yolov12n.pt ^
  --source D:\UAV\YOLOv12-BoT-SORT-ReID\data\demo\MOT\MultiUAV-003.mp4 ^
  --imgsz 1600 --conf 0.25 --iou 0.70 --device 0 --save ^
  --line_width 1 --font_size 0.35 --hide_conf
```

如需显式指定 tracker 配置：
```cmd
python run_bytetrack.py ... --tracker .\ultralytics\cfg\trackers\bytetrack.yaml
```

---

## 3. 运行 OC-SORT

默认使用 `ultralytics\cfg\trackers\ocsort.yaml`。

```cmd
python run_ocsort.py ^
  --weights .\weights\ViA_yolov12n.pt ^
  --source D:\UAV\YOLOv12-BoT-SORT-ReID\data\demo\MOT\MultiUAV-003.mp4 ^
  --imgsz 1600 --conf 0.25 --iou 0.70 --device 0 --save ^
  --line_width 1 --font_size 0.35 --hide_conf
```

如需显式指定 tracker 配置：
```cmd
python run_ocsort.py ... --tracker .\ultralytics\cfg\trackers\ocsort.yaml
```

---

## 4. 常用可视化与输出参数

- `--line_width`：框线宽（像素）。示例：`--line_width 1`
- `--font_size`：标签字号缩放。示例：`--font_size 0.35`
- `--hide_labels`：隐藏文字标签（类别/ID）
- `--hide_conf`：隐藏置信度
- `--project`、`--name`：自定义输出目录，默认保存到 `..\runs\track\<name>`

示例：
```cmd
python run_bytetrack.py ... --project ..\runs\track --name my_exp
```

---

## 5. Tracker 配置参数（YAML）

- ByteTrack：`ultralytics\cfg\trackers\bytetrack.yaml`
- OC-SORT：`ultralytics\cfg\trackers\ocsort.yaml`

关键参数（可按需要微调）：
- `track_high_thresh`：用于关联的检测分数阈值（提高更稳，降低更易召回）
- `new_track_thresh`：新轨迹初始化阈值
- `match_thresh`：IoU 匹配阈值（提高更严格，降低更易匹配）
- `track_buffer`：丢失保留帧数（更大更不易丢轨但更易误保）

当目标较小或检测分数偏低时，可尝试：
- 将 `track_high_thresh` 调低（如 0.25）
- 或将运行时检测 `--conf` 调低至 0.2～0.3

---

## 6. 输出

- 视频/图像结果：在 `--project/--name` 指定目录下保存（默认 `..\runs\track\exp_*`）
- 可选保存选项：
  - `--save`：保存可视化图像/视频
  - `--save_txt`：保存 txt 结果（在底层 Ultralytics Predictor 中支持）

---

## 7. 故障排查

- 报错 `'font_size' is not a valid YOLO argument`：本仓库已在 `ultralytics/cfg/default.yaml` 注册了 `font_size`，如你同步更新后仍报错，请确认该文件中存在 `font_size:` 字段。
- GPU 未被使用：确认 `--device 0` 且驱动/CUDA 正常。
- 运行很慢：尝试降低 `--imgsz`（如 1280 或 960）。

---
