import argparse
from pathlib import Path
from ultralytics import YOLO
import sys, os, subprocess, time

# Robust import for export_mot.py in the same folder
try:
    from export_mot import export_video_to_mot_txt, ensure_dir
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from export_mot import export_video_to_mot_txt, ensure_dir

def to_seq_name_from_stem(stem: str) -> str:
    """Map file stem to expected sequence name.
    - UAV123 -> MultiUAV-123
    - MultiUAV-123 -> MultiUAV-123
    - otherwise return stem
    """
    s = stem
    if s.startswith("UAV") and s[3:].isdigit():
        num = int(s[3:])
        return f"MultiUAV-{num:03d}"
    if s.startswith("MultiUAV-"):
        return s
    return s

def _resolve_tracker_path(args) -> str:
    """Resolve tracker config path from --tracker or --tracker_name.
    Priority: --tracker (absolute/relative path) > --tracker_name (builtin mapping).
    """
    # Built-in mapping to local YAMLs
    base_dir = Path(__file__).parent / "ultralytics" / "cfg" / "trackers"
    name_map = {
        "bytetrack": base_dir / "bytetrack.yaml",
        "botsort": base_dir / "botsort.yaml",
        "ocsort": base_dir / "ocsort.yaml",
    }
    if getattr(args, "tracker", None):
        return str(args.tracker)
    if getattr(args, "tracker_name", None):
        p = name_map.get(args.tracker_name.lower())
        if p and p.exists():
            return str(p)
        raise FileNotFoundError(f"tracker_name='{args.tracker_name}' 未找到对应yaml: {p}")
    raise ValueError("必须通过 --tracker 或 --tracker_name 指定跟踪器配置")


def main():
    parser = argparse.ArgumentParser("Batch export tracking results (MOT format) for a folder of video files")
    parser.add_argument("--weights", type=str, required=True, help="YOLO weights .pt")
    parser.add_argument("--videos_root", type=str, required=True, help="Root folder containing video files")
    parser.add_argument("--tracker", type=str, required=False, help="Tracker YAML path (e.g., ultralytics/cfg/trackers/bytetrack.yaml)")
    parser.add_argument("--tracker_name", type=str, choices=["bytetrack", "botsort", "ocsort"], help="Use builtin tracker yaml by name")
    parser.add_argument("--out_root", type=str, required=True, help="TrackEval trackers/<bench>/<tracker>/data output folder")
    parser.add_argument("--imgsz", type=int, default=1600)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--exts", type=str, nargs="*", default=[".mp4", ".avi", ".mov", ".mkv"], help="Video extensions")
    parser.add_argument("--include", type=str, nargs="*", default=None, help="Optional file stems to include")
    args = parser.parse_args()

    model = YOLO(args.weights)
    vid_root = Path(args.videos_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    if not vid_root.is_dir():
        print(f"[error] Videos root not found: {vid_root}")
        return 1

    files = [p for p in sorted(vid_root.iterdir()) if p.is_file() and p.suffix.lower() in {e.lower() for e in args.exts}]
    if args.include:
        inc = set(args.include)
        files = [p for p in files if p.stem in inc]

    if not files:
        print(f"[warn] No video files found under: {vid_root}")
        return 0

    # 解析跟踪器配置
    try:
        tracker_cfg_path = _resolve_tracker_path(args)
    except Exception as e:
        print(f"[error] {e}")
        return 2

    # 顺序处理每个视频，不并行
    for video in files:
        seq_name = to_seq_name_from_stem(video.stem)
        out_txt = out_root / f"{seq_name}.txt"
        print(f"[export] {video.name} -> {out_txt} (seq={seq_name})")
        # If user selected OC-SORT (custom), call run_ocsort.py and copy its .txt output
        if str(tracker_cfg_path).lower().endswith('ocsort.yaml'):
            # run_ocsort.py is one level up from tools/
            run_script = Path(__file__).parent.parent / 'run_ocsort.py'
            yolodir = Path(__file__).parent.parent
            run_base = yolodir / 'runs' / 'track'
            name = f"ocsort_{video.stem}"
            cmd = [
                sys.executable, str(run_script),
                '--weights', str(args.weights),
                '--source', str(video),
                '--imgsz', str(args.imgsz),
                '--conf', str(args.conf),
                '--iou', str(args.iou),
                '--device', str(args.device),
                '--save',
                '--name', name,
                '--project', str(run_base.parent),
                '--output_txt', str(out_txt)
            ]
            try:
                subprocess.run(cmd, cwd=str(yolodir), check=True)
            except subprocess.CalledProcessError as e:
                print(f"[error] run_ocsort failed for {video.name}: {e}")
                continue

            # allow FS to settle
            time.sleep(0.2)

            # locate most likely .txt produced by the run
            chosen_txt = None
            if run_base.exists():
                # search subfolders recent first
                candidates = sorted([p for p in run_base.iterdir() if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                for c in candidates:
                    # prefer folder that contains our run name
                    if name in c.name:
                        search_dirs = [c]
                        break
                else:
                    search_dirs = candidates

                for d in search_dirs:
                    for txt in d.rglob('*.txt'):
                        try:
                            text = txt.read_text(encoding='utf-8', errors='ignore')
                        except Exception:
                            continue
                        # heuristic: a valid track file has lines starting with frame numbers
                        if any(line.strip() and line.strip().split()[0].lstrip('0').isdigit() for line in text.splitlines()):
                            chosen_txt = txt
                            break
                    if chosen_txt:
                        break

            if chosen_txt is None:
                print(f"[warn] No track .txt found for {video.name} in runs/track")
                continue

            # copy relevant lines to out_txt (no reformatting here)
            out_txt.parent.mkdir(parents=True, exist_ok=True)
            try:
                txt_text = chosen_txt.read_text(encoding='utf-8', errors='ignore')
                # keep only non-empty lines
                lines = [l for l in txt_text.splitlines() if l.strip()]
                out_txt.write_text('\n'.join(lines), encoding='utf-8')
            except Exception as e:
                print(f"[error] Failed to write {out_txt}: {e}")
                continue
        else:
            export_video_to_mot_txt(
                model=model,
                source=str(video),
                out_txt=out_txt,
                tracker_cfg=tracker_cfg_path,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
            )

if __name__ == "__main__":
    main()