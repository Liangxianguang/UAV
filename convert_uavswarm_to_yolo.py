import os
import json
from pathlib import Path
import shutil
import random

def convert_mot_to_yolo(gt_file, img_dir, output_dir, image_width=1920, image_height=1080):
    """
    转换MOT格式的gt.txt到YOLO格式
    MOT格式: frame_id, track_id, x, y, w, h, conf, class_id, visibility
    YOLO格式: class x_center y_center width height (归一化)
    """
    if not os.path.exists(gt_file):
        return
    
    # 读取gt.txt
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    
    # 按帧分组
    frame_data = {}
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 7:
            frame_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            visibility = float(parts[8]) if len(parts) > 8 else 1.0
            
            # 只保留可见性 > 0.3 和置信度 > 0 的检测
            if visibility > 0.3 and conf > 0:
                if frame_id not in frame_data:
                    frame_data[frame_id] = []
                frame_data[frame_id].append((x, y, w, h))
    
    # 为每帧创建YOLO格式的txt文件
    for frame_id, detections in frame_data.items():
        img_path = os.path.join(img_dir, f"{frame_id:06d}.jpg")
        if not os.path.exists(img_path):
            continue
        
        txt_filename = f"{frame_id:06d}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for x, y, w, h in detections:
                # 转换为YOLO格式（归一化中心坐标）
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                width_norm = w / image_width
                height_norm = h / image_height
                
                # 确保值在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width_norm = max(0, min(1, width_norm))
                height_norm = max(0, min(1, height_norm))
                
                label = 0  # UAV class
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

def process_uavswarm_dataset(base_dir, output_train_imgs, output_train_labels, output_val_imgs, output_val_labels):
    """
    处理UAVSwarm数据集，将所有序列的gt.txt转换为YOLO格式
    处理train和test目录下的所有序列
    """
    # 创建输出目录
    for d in [output_train_imgs, output_train_labels, output_val_imgs, output_val_labels]:
        os.makedirs(d, exist_ok=True)
    
    all_sequences = []
    
    # 遍历train目录下的所有序列
    train_dir = os.path.join(base_dir, 'train')
    if os.path.exists(train_dir):
        for seq_name in os.listdir(train_dir):
            seq_path = os.path.join(train_dir, seq_name)
            if os.path.isdir(seq_path):
                gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
                img_dir = os.path.join(seq_path, 'img1')
                
                if os.path.exists(gt_file) and os.path.exists(img_dir):
                    all_sequences.append((seq_path, gt_file, img_dir, seq_name))
    
    # 遍历test目录下的所有序列
    test_dir = os.path.join(base_dir, 'test')
    if os.path.exists(test_dir):
        for seq_name in os.listdir(test_dir):
            seq_path = os.path.join(test_dir, seq_name)
            if os.path.isdir(seq_path):
                gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
                img_dir = os.path.join(seq_path, 'img1')
                
                if os.path.exists(gt_file) and os.path.exists(img_dir):
                    all_sequences.append((seq_path, gt_file, img_dir, seq_name))
    
    # 随机分割（70% train, 30% val）
    random.shuffle(all_sequences)
    total = len(all_sequences)
    train_count = int(total * 0.7)
    
    train_sequences = all_sequences[:train_count]
    val_sequences = all_sequences[train_count:]
    
    # 处理train序列
    for seq_path, gt_file, img_dir, seq_name in train_sequences:
        print(f"Processing train sequence: {seq_name}")
        
        # 获取图片分辨率（从第一张图片）
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        if img_files:
            import cv2
            first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
            if first_img is not None:
                height, width = first_img.shape[:2]
            else:
                width, height = 1920, 1080
        else:
            width, height = 1920, 1080
        
        # 创建临时标签目录
        temp_labels_dir = os.path.join(output_train_labels, seq_name)
        os.makedirs(temp_labels_dir, exist_ok=True)
        
        # 转换gt.txt到YOLO格式
        convert_mot_to_yolo(gt_file, img_dir, temp_labels_dir, width, height)
        
        # 复制图片和标签到train目录
        for img_file in img_files:
            src_img = os.path.join(img_dir, img_file)
            dst_img = os.path.join(output_train_imgs, f"{seq_name}_{img_file}")
            shutil.copy(src_img, dst_img)
            
            base_name = os.path.splitext(img_file)[0]
            src_txt = os.path.join(temp_labels_dir, f"{base_name}.txt")
            dst_txt = os.path.join(output_train_labels, f"{seq_name}_{base_name}.txt")
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
        
        # 删除临时目录
        shutil.rmtree(temp_labels_dir)
    
    # 处理val序列（类似train）
    for seq_path, gt_file, img_dir, seq_name in val_sequences:
        print(f"Processing val sequence: {seq_name}")
        
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        if img_files:
            import cv2
            first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
            if first_img is not None:
                height, width = first_img.shape[:2]
            else:
                width, height = 1920, 1080
        else:
            width, height = 1920, 1080
        
        temp_labels_dir = os.path.join(output_val_labels, seq_name)
        os.makedirs(temp_labels_dir, exist_ok=True)
        
        convert_mot_to_yolo(gt_file, img_dir, temp_labels_dir, width, height)
        
        for img_file in img_files:
            src_img = os.path.join(img_dir, img_file)
            dst_img = os.path.join(output_val_imgs, f"{seq_name}_{img_file}")
            shutil.copy(src_img, dst_img)
            
            base_name = os.path.splitext(img_file)[0]
            src_txt = os.path.join(temp_labels_dir, f"{base_name}.txt")
            dst_txt = os.path.join(output_val_labels, f"{seq_name}_{base_name}.txt")
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)
        
        shutil.rmtree(temp_labels_dir)
    
    print(f"Processed {total} sequences: {len(train_sequences)} train, {len(val_sequences)} val")

# 主程序
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert UAVSwarm dataset to YOLO format')
    parser.add_argument('--base_dir', type=str, 
                        default=r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\UAVSwarm-dataset-master",
                        help='Base directory of UAVSwarm dataset')
    parser.add_argument('--output_dir', type=str,
                        default=r"D:\UAV\YOLOv12-BoT-SORT-ReID\data\uavswarm_yolo",
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Train split ratio (default: 0.7)')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'all'], default='all',
                        help='Process train/test/all sequences')
    
    args = parser.parse_args()
    
    # 输出路径
    output_train_imgs = os.path.join(args.output_dir, 'images', 'train')
    output_train_labels = os.path.join(args.output_dir, 'labels', 'train')
    output_val_imgs = os.path.join(args.output_dir, 'images', 'val')
    output_val_labels = os.path.join(args.output_dir, 'labels', 'val')
    
    print("\n🚀 UAVSwarm to YOLO Conversion Script")
    print("="*70)
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Train split: {args.train_split:.1%}")
    print("="*70)
    
    # 验证输入目录
    if not os.path.exists(args.base_dir):
        print(f"\n❌ Base directory not found: {args.base_dir}")
        exit(1)
    
    # 获取所有序列
    all_sequences = []
    
    if args.mode in ['train', 'all']:
        train_dir = os.path.join(args.base_dir, 'train')
        if os.path.exists(train_dir):
            train_seqs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            print(f"\n📂 Found {len(train_seqs)} training sequences")
            for seq in sorted(train_seqs):
                seq_path = os.path.join(train_dir, seq)
                gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
                img_dir = os.path.join(seq_path, 'img1')
                if os.path.exists(gt_file) and os.path.exists(img_dir):
                    all_sequences.append((seq_path, gt_file, img_dir, seq))
                    print(f"   ✅ {seq}")
                else:
                    print(f"   ⚠️  {seq} (missing gt.txt or img1)")
    
    if args.mode in ['test', 'all']:
        test_dir = os.path.join(args.base_dir, 'test')
        if os.path.exists(test_dir):
            test_seqs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            print(f"\n📂 Found {len(test_seqs)} test sequences")
            for seq in sorted(test_seqs):
                seq_path = os.path.join(test_dir, seq)
                gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
                img_dir = os.path.join(seq_path, 'img1')
                if os.path.exists(gt_file) and os.path.exists(img_dir):
                    all_sequences.append((seq_path, gt_file, img_dir, seq))
                    print(f"   ✅ {seq}")
                else:
                    print(f"   ⚠️  {seq} (missing gt.txt or img1)")
    
    if not all_sequences:
        print("\n❌ No valid sequences found!")
        exit(1)
    
    print(f"\n📊 Total sequences to process: {len(all_sequences)}")
    
    # 处理数据集
    process_uavswarm_dataset(args.base_dir, output_train_imgs, output_train_labels, 
                             output_val_imgs, output_val_labels)
    
    # 统计输出
    train_img_count = len([f for f in os.listdir(output_train_imgs) if f.endswith(('.jpg', '.png'))]) if os.path.exists(output_train_imgs) else 0
    val_img_count = len([f for f in os.listdir(output_val_imgs) if f.endswith(('.jpg', '.png'))]) if os.path.exists(output_val_imgs) else 0
    
    print("\n" + "="*70)
    print("✅ UAVSwarm dataset conversion completed!")
    print("="*70)
    print(f"📁 Train images: {train_img_count}")
    print(f"📁 Val images: {val_img_count}")
    print(f"📁 Total: {train_img_count + val_img_count}")
    print("="*70)
