import argparse
import os
import sys
import subprocess
from pathlib import Path


def find_repo_root() -> Path:
    """Robustly locate repo root that contains 'TrackEval' and 'BoT-SORT'."""
    here = Path(__file__).resolve()
    p = here.parent
    for _ in range(6):
        if (p / 'TrackEval').is_dir() and (p / 'BoT-SORT').is_dir():
            return p
        p = p.parent
    # Fallback: typical layout .../YOLOv12-BoT-SORT-ReID
    return Path(__file__).resolve().parents[2]


ROOT = find_repo_root()


def run_py_module(module: str, args: list[str], cwd: Path | None = None):
    cmd = [sys.executable, '-m', module, *args]
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def run_py_file(file_path: Path, args: list[str]):
    if not file_path.exists():
        raise FileNotFoundError(f"Script not found: {file_path}")
    cmd = [sys.executable, str(file_path), *args]
    print('[run]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def prepare_gt(original_root: Path, benchmark: str, split: str):
    tool = ROOT / 'BoT-SORT' / 'yolov12' / 'tools' / 'prepare_gt_from_original.py'
    out_gt_root = ROOT / 'TrackEval' / 'data' / 'gt' / 'mot_challenge'
    args = [
        '--original-root', str(original_root),
        '--outeval-root', str(out_gt_root),
        '--benchmark', benchmark,
        '--split', split,
    ]
    run_py_file(tool, args)
    seqmap = out_gt_root / benchmark / 'seqmaps' / f'{split}.txt'
    return out_gt_root / benchmark, seqmap


def export_from_videos(weights: Path, videos_root: Path, tracker_yaml: Path, out_root: Path, imgsz: int, conf: float, iou: float, device: str):
    tool = ROOT / 'BoT-SORT' / 'yolov12' / 'tools' / 'export_mot_all_videos.py'
    ensure_dir(out_root)
    args = [
        '--weights', str(weights),
        '--videos_root', str(videos_root),
        '--tracker', str(tracker_yaml),
        '--out_root', str(out_root),
        '--imgsz', str(imgsz),
        '--conf', str(conf),
        '--iou', str(iou),
        '--device', device,
    ]
    run_py_file(tool, args)


def export_from_frames(weights: Path, frames_root: Path, tracker_yaml: Path, out_root: Path, imgsz: int, conf: float, iou: float, device: str, include: list[str] | None = None):
    tool = ROOT / 'BoT-SORT' / 'yolov12' / 'tools' / 'export_mot_all.py'
    ensure_dir(out_root)
    args = [
        '--weights', str(weights),
        '--sequences_root', str(frames_root),
        '--tracker', str(tracker_yaml),
        '--out_root', str(out_root),
        '--imgsz', str(imgsz),
        '--conf', str(conf),
        '--iou', str(iou),
        '--device', device,
    ]
    if include:
        args.extend(['--include', *include])
    run_py_file(tool, args)


def run_trackeval(benchmark: str, split: str, seqmap_file: Path, trackers: list[str]):
    # Use TrackEval module runner
    args = [
        '--GT_FOLDER', str(ROOT / 'TrackEval' / 'data' / 'gt' / 'mot_challenge'),
        '--TRACKERS_FOLDER', str(ROOT / 'TrackEval' / 'data' / 'trackers' / 'mot_challenge'),
        '--BENCHMARK', benchmark,
        '--SPLIT_TO_EVAL', split,
        '--SEQMAP_FILE', str(seqmap_file),
        '--TRACKERS_TO_EVAL', *trackers,
        '--METRICS', 'HOTA', 'CLEAR', 'Identity',
    ]
    run_py_module('trackeval.scripts.run_mot_challenge', args, cwd=ROOT / 'TrackEval')


def run_proxy_metrics(pred_dir: Path, out_json: Path, out_csv: Path):
    tool = ROOT / 'BoT-SORT' / 'yolov12' / 'tools' / 'track_quality_proxy.py'
    ensure_dir(out_json.parent)
    args = [
        '--input', str(pred_dir),
        '--out-json', str(out_json),
        '--out-csv', str(out_csv),
    ]
    run_py_file(tool, args)


def main():
    parser = argparse.ArgumentParser(description='One-click pipeline: prepare GT (train), export predictions (train/test), run TrackEval and proxy metrics.')
    parser.add_argument('--weights', required=True, help='YOLO .pt path')
    parser.add_argument('--trackers', nargs='+', default=['bytetrack', 'ocsort'], help='Trackers to eval: names in {bytetrack, ocsort}')
    parser.add_argument('--train_videos_root', default=str(ROOT / 'data' / 'MultiUAV_Train' / 'TrainVideos'))
    parser.add_argument('--original_gt_root', default=str(ROOT / 'data' / 'MultiUAV_Train' / 'original_label_file'))
    parser.add_argument('--benchmark', default='UAV-train')
    parser.add_argument('--split', default='train')
    parser.add_argument('--device', default='0')
    parser.add_argument('--imgsz', type=int, default=1600)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7)
    # test proxy options
    parser.add_argument('--run_test_proxy', action='store_true', help='Also export test set predictions and compute proxy metrics (no GT needed)')
    parser.add_argument('--test_videos_root', default=str(ROOT / 'data' / 'MultiUAV_Test' / 'TestVideos'))
    parser.add_argument('--test_frames_root', default=str(ROOT / 'data' / 'MultiUAV_Test' / 'Test_imgs'))
    parser.add_argument('--reports_root', default=str(ROOT / 'reports'))
    args = parser.parse_args()

    weights = Path(args.weights)
    train_videos_root = Path(args.train_videos_root)
    original_gt_root = Path(args.original_gt_root)
    reports_root = Path(args.reports_root)

    tracker_yaml_map = {
        'bytetrack': ROOT / 'ultralytics' / 'cfg' / 'trackers' / 'bytetrack.yaml',
        'ocsort': ROOT / 'ultralytics' / 'cfg' / 'trackers' / 'ocsort.yaml',
    }

    # 1) Prepare GT for training benchmark
    gt_benchmark_root, seqmap_file = prepare_gt(original_gt_root, args.benchmark, args.split)
    print('[info] GT ready at:', gt_benchmark_root)
    print('[info] Seqmap file:', seqmap_file)

    # 2) Export predictions from training videos for each tracker
    for tr in args.trackers:
        if tr not in tracker_yaml_map:
            raise ValueError(f'Unknown tracker: {tr}')
        out_root = ROOT / 'TrackEval' / 'data' / 'trackers' / 'mot_challenge' / args.benchmark / tr / 'data'
        export_from_videos(
            weights=weights,
            videos_root=train_videos_root,
            tracker_yaml=tracker_yaml_map[tr],
            out_root=out_root,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )

    # 3) Run TrackEval on training benchmark across trackers
    run_trackeval(args.benchmark, args.split, seqmap_file, args.trackers)

    # 4) Optional: test-set proxy metrics (no GT)
    if args.run_test_proxy:
        # Export predictions for test set under a separate benchmark name
        test_bench = 'UAV-test'
        for tr in args.trackers:
            out_root = ROOT / 'TrackEval' / 'data' / 'trackers' / 'mot_challenge' / test_bench / tr / 'data'
            # Prefer frames root if it exists (usually available for test set)
            test_frames_root = Path(args.test_frames_root)
            test_videos_root = Path(args.test_videos_root)
            if test_frames_root.is_dir() and any(test_frames_root.iterdir()):
                export_from_frames(
                    weights=weights,
                    frames_root=test_frames_root,
                    tracker_yaml=tracker_yaml_map[tr],
                    out_root=out_root,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                )
            elif test_videos_root.is_dir() and any(test_videos_root.iterdir()):
                export_from_videos(
                    weights=weights,
                    videos_root=test_videos_root,
                    tracker_yaml=tracker_yaml_map[tr],
                    out_root=out_root,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                )
            else:
                print('[warn] No test set found to export for proxy metrics')

            # Compute proxy metrics per tracker
            run_proxy_metrics(
                pred_dir=out_root,
                out_json=reports_root / f'{tr}_test_proxy.json',
                out_csv=reports_root / f'{tr}_test_proxy.csv',
            )

    print('[done] Evaluation pipeline completed.')


if __name__ == '__main__':
    main()
