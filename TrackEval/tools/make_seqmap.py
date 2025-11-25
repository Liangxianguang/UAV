#!/usr/bin/env python3
"""
自动生成完整的seqmap文件，包含所有GT序列
"""
from pathlib import Path
import os

def main():
    gt_root = Path(r'd:\UAV\YOLOv12-BoT-SORT-ReID\TrackEval\data\gt\mot_challenge\UAV-train')
    seqmap_file = Path(r'd:\UAV\YOLOv12-BoT-SORT-ReID\TrackEval\data\gt\mot_challenge\seqmaps\UAV-train-train.txt')
    
    print(f"扫描GT目录: {gt_root}")
    
    # 收集所有有效序列（包含gt/gt.txt的目录）
    seqs = []
    for p in sorted(gt_root.iterdir()):
        if p.is_dir() and (p / 'gt' / 'gt.txt').is_file():
            seqs.append(p.name)
            print(f"发现序列: {p.name}")
    
    if not seqs:
        print("错误：未找到任何有效序列！")
        return
    
    # 确保输出目录存在
    seqmap_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入seqmap文件
    with seqmap_file.open('w', encoding='utf-8') as f:
        f.write('name\n')
        for seq in seqs:
            f.write(seq + '\n')
    
    print(f"\n✅ 成功生成seqmap: {seqmap_file}")
    print(f"📊 包含 {len(seqs)} 个序列")
    print("前10个序列:", seqs[:10])
    if len(seqs) > 10:
        print("...")

if __name__ == '__main__':
    main()