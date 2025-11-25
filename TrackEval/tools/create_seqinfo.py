#!/usr/bin/env python3
"""
批量为所有GT序列创建seqinfo.ini文件
"""
from pathlib import Path
import os

def count_lines_in_gt(gt_file):
    """统计gt.txt文件的最大帧号"""
    max_frame = 0
    try:
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    frame_id = int(line.split(',')[0])
                    max_frame = max(max_frame, frame_id)
    except Exception as e:
        print(f"警告：读取 {gt_file} 时出错: {e}")
        return 750  # 默认值
    return max_frame

def create_seqinfo(seq_dir, seq_name):
    """为单个序列创建seqinfo.ini"""
    seqinfo_path = seq_dir / 'seqinfo.ini'
    gt_file = seq_dir / 'gt' / 'gt.txt'
    
    if seqinfo_path.exists():
        return f"跳过 {seq_name}: seqinfo.ini 已存在"
    
    if not gt_file.exists():
        return f"错误 {seq_name}: gt.txt 不存在"
    
    # 计算序列长度
    seq_length = count_lines_in_gt(gt_file)
    
    # 写入seqinfo.ini
    seqinfo_content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate=30
seqLength={seq_length}
imWidth=640
imHeight=512
imExt=.jpg
"""
    
    with open(seqinfo_path, 'w', encoding='utf-8') as f:
        f.write(seqinfo_content)
    
    return f"✅ {seq_name}: 创建seqinfo.ini (长度={seq_length})"

def main():
    gt_root = Path(r'd:\UAV\YOLOv12-BoT-SORT-ReID\TrackEval\data\gt\mot_challenge\UAV-train')
    
    print(f"扫描GT目录: {gt_root}")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 处理所有序列目录
    for seq_dir in sorted(gt_root.iterdir()):
        if seq_dir.is_dir() and seq_dir.name.startswith('MultiUAV-'):
            result = create_seqinfo(seq_dir, seq_dir.name)
            print(result)
            
            if result.startswith('✅'):
                success_count += 1
            elif result.startswith('跳过'):
                skip_count += 1
            else:
                error_count += 1
    
    print(f"\n📊 处理完成:")
    print(f"  ✅ 新创建: {success_count}")
    print(f"  ⏭️  跳过: {skip_count}")
    print(f"  ❌ 错误: {error_count}")

if __name__ == '__main__':
    main()