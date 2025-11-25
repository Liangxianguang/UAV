#!/usr/bin/env python3
"""
修复序列长度问题的脚本
1. 检查GT数据确定正确的seqLength
2. 更新seqinfo.ini文件
3. 清理跟踪结果中超出长度的帧
"""

import os
import csv
from pathlib import Path
import configparser

def get_max_frame_from_gt(gt_file):
    """从GT文件中获取最大帧号"""
    if not gt_file.exists():
        return 0
    
    max_frame = 0
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():
                    try:
                        frame_num = int(row[0])
                        max_frame = max(max_frame, frame_num)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading {gt_file}: {e}")
        return 0
    
    return max_frame

def update_seqinfo(seqinfo_file, seq_length):
    """更新seqinfo.ini文件中的seqLength"""
    if not seqinfo_file.exists():
        return False
    
    try:
        config = configparser.ConfigParser()
        config.read(seqinfo_file, encoding='utf-8')
        
        if 'Sequence' not in config:
            config.add_section('Sequence')
        
        config['Sequence']['seqLength'] = str(seq_length)
        
        with open(seqinfo_file, 'w', encoding='utf-8') as f:
            config.write(f)
        
        return True
    except Exception as e:
        print(f"Error updating {seqinfo_file}: {e}")
        return False

def clean_tracker_file(tracker_file, max_frame):
    """清理跟踪结果文件，删除超出最大帧号的行"""
    if not tracker_file.exists():
        return False
    
    try:
        valid_lines = []
        removed_count = 0
        
        with open(tracker_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():
                    try:
                        frame_num = int(row[0])
                        if frame_num <= max_frame:
                            valid_lines.append(','.join(row))
                        else:
                            removed_count += 1
                    except ValueError:
                        # 保留无法解析的行
                        valid_lines.append(','.join(row))
        
        # 重写文件
        with open(tracker_file, 'w', encoding='utf-8') as f:
            for line in valid_lines:
                f.write(line + '\n')
        
        if removed_count > 0:
            print(f"  Cleaned {tracker_file.name}: removed {removed_count} lines with frames > {max_frame}")
        
        return True
    except Exception as e:
        print(f"Error cleaning {tracker_file}: {e}")
        return False

def main():
    # 路径设置
    gt_root = Path(r'd:\UAV\YOLOv12-BoT-SORT-ReID\TrackEval\data\gt\mot_challenge\UAV-train')
    trackers_root = Path(r'd:\UAV\YOLOv12-BoT-SORT-ReID\TrackEval\data\trackers\mot_challenge\UAV-train')
    
    if not gt_root.exists():
        print(f"GT root not found: {gt_root}")
        return
    
    # 处理所有序列
    sequences = []
    for seq_dir in sorted(gt_root.iterdir()):
        if seq_dir.is_dir() and seq_dir.name.startswith('MultiUAV-'):
            sequences.append(seq_dir.name)
    
    print(f"Found {len(sequences)} sequences to process")
    
    updated_count = 0
    cleaned_count = 0
    
    for seq_name in sequences:
        print(f"\nProcessing {seq_name}...")
        
        # 1. 检查GT数据获取正确的序列长度
        gt_file = gt_root / seq_name / 'gt' / 'gt.txt'
        max_frame = get_max_frame_from_gt(gt_file)
        
        if max_frame == 0:
            print(f"  Warning: No valid frames found in {gt_file}")
            continue
        
        print(f"  GT data shows max frame: {max_frame}")
        
        # 2. 更新seqinfo.ini
        seqinfo_file = gt_root / seq_name / 'seqinfo.ini'
        if update_seqinfo(seqinfo_file, max_frame):
            print(f"  Updated seqinfo.ini: seqLength = {max_frame}")
            updated_count += 1
        
        # 3. 清理跟踪结果文件
        for tracker in ['bytetrack', 'botsort', 'ocsort', 'my_botsort']:
            tracker_file = trackers_root / tracker / 'data' / f'{seq_name}.txt'
            if tracker_file.exists():
                if clean_tracker_file(tracker_file, max_frame):
                    cleaned_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Updated {updated_count} seqinfo.ini files")
    print(f"Cleaned {cleaned_count} tracker files")
    print(f"All sequence length issues should now be fixed!")

if __name__ == "__main__":
    main()