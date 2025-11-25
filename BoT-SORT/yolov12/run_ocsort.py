import argparse
from pathlib import Path

from ultralytics import YOLO

# Local tracker cfg path for OC-SORT
DEFAULT_TRACKER_CFG = Path(__file__).parent / "ultralytics" / "cfg" / "trackers" / "ocsort.yaml"

def main():
    parser = argparse.ArgumentParser("YOLOv12 + OC-SORT runner")
    parser.add_argument("--weights", type=str, default=str(Path(__file__).parent / "weights" / "MOT_yolov12n.pt"), help="Path to YOLOv12 weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Source: video file, image folder, or stream URL")
    parser.add_argument("--imgsz", type=int, default=1600, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="Detector NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--save", action="store_true", help="Save annotated results")
    parser.add_argument("--project", type=str, default=str(Path(__file__).parent.parent / "runs" / "track"), help="Project dir for results")
    parser.add_argument("--name", type=str, default="exp_ocsort", help="Run name under project directory")
    parser.add_argument("--tracker", type=str, default=str(DEFAULT_TRACKER_CFG), help="Path to ocsort.yaml")
    parser.add_argument("--output_txt", type=str, default=None, help="Path to save MOT results txt (optional)")
    parser.add_argument("--persist", action="store_true", help="Persist tracker state across videos in a stream batch")
    # Visualization controls
    parser.add_argument("--line_width", type=int, default=1, help="Bounding-box line width (pixels)")
    parser.add_argument("--font_size", type=float, default=0.4, help="Label font scale (smaller -> less occlusion)")
    parser.add_argument("--hide_labels", action="store_true", help="Hide text labels (IDs/classes)")
    parser.add_argument("--hide_conf", action="store_true", help="Hide confidence values")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # Use built-in Ultralytics .track API with our OC-SORT cfg
    results = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        project=args.project,
        name=args.name,
        tracker=args.tracker,
        persist=args.persist,
        verbose=True,
        # Visualization passthrough
        line_width=args.line_width,
        font_size=args.font_size,
        show_labels=not args.hide_labels,
        show_conf=not args.hide_conf,
    )

    # 如果指定了 --output_txt，则保存 MOT 格式结果（每帧每ID只保留一条，帧号递增）
    if args.output_txt:
        mot_lines = []
        # results 是每帧的结果，按顺序遍历，帧号从 1 开始递增
        for frame_idx, r in enumerate(results, start=1):
            if hasattr(r, 'boxes') and r.boxes is not None:
                ids = r.boxes.id if hasattr(r.boxes, 'id') else None
                xywh = r.boxes.xywh.cpu().numpy() if hasattr(r.boxes, 'xywh') else None
                confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else None
                if ids is not None and xywh is not None:
                    seen = set()
                    for i in range(len(ids)):
                        tid = int(ids[i])
                        if tid in seen:
                            continue  # 跳过同一帧重复ID
                        seen.add(tid)
                        line = f"{frame_idx},{tid},{xywh[i][0]:.2f},{xywh[i][1]:.2f},{xywh[i][2]:.2f},{xywh[i][3]:.2f},{confs[i]:.4f},-1,-1,-1"
                        mot_lines.append(line)
        out_path = Path(args.output_txt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('\n'.join(mot_lines), encoding='utf-8')


if __name__ == "__main__":
    main()
