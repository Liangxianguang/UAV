import argparse
from pathlib import Path
from ultralytics import YOLO
import sys, os

# Robust import for export_mot.py in the same folder
try:
    from export_mot import export_video_to_mot_txt, ensure_dir
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from export_mot import export_video_to_mot_txt, ensure_dir


def main():
    parser = argparse.ArgumentParser("Batch export tracking results (MOT format) for a folder of sequences")
    parser.add_argument("--weights", type=str, required=True, help="YOLO weights .pt")
    parser.add_argument(
        "--sequences_root",
        type=str,
        required=True,
        help="Root folder containing per-sequence subfolders with frames (e.g., .../MultiUAV_Test/Test_imgs)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        required=True,
        help="Tracker YAML path (e.g., ultralytics/cfg/trackers/bytetrack.yaml or ocsort.yaml)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help=(
            "TrackEval trackers root (e.g., D:/.../TrackEval/data/trackers/mot_challenge/UAV-val/<tracker_name>/data). "
            "This script will create <seq>.txt under the given folder."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=1600)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="Optional sequence names to include (defaults to all immediate subfolders)",
    )
    args = parser.parse_args()

    model = YOLO(args.weights)
    seq_root = Path(args.sequences_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    # Enumerate sequences
    subdirs = [p for p in seq_root.iterdir() if p.is_dir()]
    if args.include:
        names = set(args.include)
        subdirs = [p for p in subdirs if p.name in names]

    for seq_dir in sorted(subdirs):
        seq_name = seq_dir.name
        out_txt = out_root / f"{seq_name}.txt"
        print(f"[export] {seq_name} -> {out_txt}")
        export_video_to_mot_txt(
            model=model,
            source=str(seq_dir),  # folder of images is supported by Ultralytics loader
            out_txt=out_txt,
            tracker_cfg=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )


if __name__ == "__main__":
    main()
