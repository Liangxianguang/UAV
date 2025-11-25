from pathlib import Path

gt_root = Path(r'd:\UAV\YOLOv12-BoT-SORT-ReID\TrackEval\data\gt\mot_challenge\UAV-train')
seqmap = Path(r'd:\UAV\YOLOv12-BoT-SORT-ReID\TrackEval\data\gt\mot_challenge\seqmaps\UAV-train-train.txt')

seqs = []
for p in sorted(gt_root.iterdir()):
    if p.is_dir() and (p / 'gt' / 'gt.txt').is_file():
        seqs.append(p.name)

seqmap.parent.mkdir(parents=True, exist_ok=True)
with seqmap.open('w', encoding='utf-8') as f:
    f.write('name\n')
    for s in seqs:
        f.write(s + '\n')

print(f'[OK] 写入 {seqmap}，共 {len(seqs)} 条序列')
for i, s in enumerate(seqs[:10]):  # 显示前10条
    print(f'  {i+1}: {s}')
if len(seqs) > 10:
    print(f'  ... 还有 {len(seqs)-10} 条')