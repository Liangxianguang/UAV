import argparse
import os
import shutil


def uav_to_multiuav(name: str) -> str | None:
    """Convert UAV123 -> MultiUAV-123. Return None if not convertible."""
    if name.startswith('UAV'):
        num = name[3:]
        if num.isdigit():
            return f'MultiUAV-{int(num):03d}'
    return None


def main():
    parser = argparse.ArgumentParser(description='Copy original GT (UAVxxx/gt/gt.txt) to TrackEval MOT format and generate a seqmap file.')
    parser.add_argument('--original-root', default=os.path.join('data', 'MultiUAV_Train', 'original_label_file'), help='Root of original GT folders (e.g., data/MultiUAV_Train/original_label_file)')
    parser.add_argument('--outeval-root', default=os.path.join('TrackEval', 'data', 'gt', 'mot_challenge'), help='TrackEval gt root (mot_challenge)')
    parser.add_argument('--benchmark', default='UAV-train', help='Benchmark name under mot_challenge (e.g., UAV-train)')
    parser.add_argument('--split', default='train', help='Split name for seqmaps file (e.g., train/val)')
    args = parser.parse_args()

    original_root = args.original_root
    out_root = os.path.join(args.outeval_root, args.benchmark)
    seqmap_dir = os.path.join(out_root, 'seqmaps')
    os.makedirs(seqmap_dir, exist_ok=True)

    if not os.path.isdir(original_root):
        print(f'[error] Original GT root not found: {original_root}')
        return 1

    sequences = []
    for d in sorted(os.listdir(original_root)):
        src_dir = os.path.join(original_root, d, 'gt')
        if not os.path.isdir(src_dir):
            continue
        seq_name = uav_to_multiuav(d)
        if not seq_name:
            continue
        src_gt = os.path.join(src_dir, 'gt.txt')
        if not os.path.isfile(src_gt):
            continue

        dst_gt_dir = os.path.join(out_root, seq_name, 'gt')
        os.makedirs(dst_gt_dir, exist_ok=True)
        dst_gt = os.path.join(dst_gt_dir, 'gt.txt')
        shutil.copy2(src_gt, dst_gt)
        sequences.append(seq_name)

    # write seqmap file with header 'name' then sequence names per line
    seqmap_path = os.path.join(seqmap_dir, f'{args.split}.txt')
    with open(seqmap_path, 'w', encoding='utf-8') as f:
        f.write('name\n')
        for s in sequences:
            f.write(f'{s}\n')

    print(f'[ok] Copied {len(sequences)} sequences to {out_root}')
    print(f'[ok] Seqmap written: {seqmap_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
