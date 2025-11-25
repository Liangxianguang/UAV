import argparse
from pathlib import Path

from ultralytics import YOLO

# Local tracker cfg path
DEFAULT_TRACKER_CFG = Path(__file__).parent / "ultralytics" / "cfg" / "trackers" / "bytetrack.yaml"

def main():
    parser = argparse.ArgumentParser("YOLOv12 + ByteTrack runner")
    parser.add_argument("--weights", type=str, default=str(Path(__file__).parent / "weights" / "MOT_yolov12n.pt"), help="Path to YOLOv12 weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Source: video file, image folder, or stream URL")
    parser.add_argument("--imgsz", type=int, default=1600, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="Detector NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--save", action="store_true", help="Save annotated results")
    parser.add_argument("--project", type=str, default=str(Path(__file__).parent.parent / "runs" / "track"), help="Project dir for results")
    parser.add_argument("--name", type=str, default="exp_bytetrack", help="Run name under project directory")
    parser.add_argument("--tracker", type=str, default=str(DEFAULT_TRACKER_CFG), help="Path to bytetrack.yaml")
    parser.add_argument("--persist", action="store_true", help="Persist tracker state across videos in a stream batch")
    # Visualization controls
    parser.add_argument("--line_width", type=int, default=1, help="Bounding-box line width (pixels)")
    parser.add_argument("--font_size", type=float, default=0.4, help="Label font scale (smaller -> less occlusion)")
    parser.add_argument("--hide_labels", action="store_true", help="Hide text labels (IDs/classes)")
    parser.add_argument("--hide_conf", action="store_true", help="Hide confidence values")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # Use built-in Ultralytics .track API with our local ByteTrack cfg
    model.track(
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


if __name__ == "__main__":
    main()
