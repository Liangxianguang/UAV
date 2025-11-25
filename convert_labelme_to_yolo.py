import json
import os
from pathlib import Path
import shutil
import random

def convert_labelme_to_yolo(json_path, output_dir, image_width, image_height):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    txt_path = os.path.join(output_dir, Path(json_path).stem + '.txt')
    with open(txt_path, 'w') as f:
        for shape in data['shapes']:
            if shape['shape_type'] == 'rectangle':
                label = 0  # UAV class
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[2]
                x_center = (x1 + x2) / 2 / image_width
                y_center = (y1 + y2) / 2 / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 通用脚本：遍历base_dir下的所有子目录，收集所有JSON和图片
base_dir = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\images"
output_images_train = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\images\train"
output_images_val = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\images\val"
output_labels_train = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\labels\train"
output_labels_val = r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\uav_custom\labels\val"

for d in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
    os.makedirs(d, exist_ok=True)

all_files = []
# 遍历base_dir下的所有子目录
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            all_files.append(os.path.join(root, file))

# 随机分割（70% train, 30% val）避免角度偏置
random.shuffle(all_files)
total = len(all_files)
train_count = int(total * 0.7)
train_files = all_files[:train_count]
val_files = all_files[train_count:]

for json_path in train_files:
    base_name = Path(json_path).stem
    img_path = json_path.replace('.json', '.png')  # 假设图片是.png
    if os.path.exists(img_path):
        shutil.copy(img_path, os.path.join(output_images_train, f"{base_name}.png"))
    convert_labelme_to_yolo(json_path, output_labels_train, 1280, 720)

for json_path in val_files:
    base_name = Path(json_path).stem
    img_path = json_path.replace('.json', '.png')
    if os.path.exists(img_path):
        shutil.copy(img_path, os.path.join(output_images_val, f"{base_name}.png"))
    convert_labelme_to_yolo(json_path, output_labels_val, 1280, 720)

print(f"Processed {total} files: {len(train_files)} train, {len(val_files)} val")