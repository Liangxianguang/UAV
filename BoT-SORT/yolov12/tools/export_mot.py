import os
from pathlib import Path
import torch

def ensure_dir(path):
	if isinstance(path, (str, Path)):
		os.makedirs(str(path), exist_ok=True)

def export_video_to_mot_txt(model, source, out_txt, tracker_cfg, imgsz, conf, iou, device):
	"""
	用Ultralytics YOLO模型对视频/帧目录进行跟踪，导出MOTChallenge格式TXT：
	每行格式：frame,id,left,top,width,height,conf,-1,-1,-1
	"""
	ensure_dir(Path(out_txt).parent)
	with open(out_txt, "w", encoding="utf-8") as f:
		frame_idx = 0
		# 调试：打印模型当前设备，并在 CUDA 下设置 benchmark 以提速
		try:
			pdev = next(model.model.parameters()).device  # type: ignore[attr-defined]
			print(f"[DEBUG] export_mot: 模型参数设备: {pdev}")
		except Exception as e:
			print(f"[WARN] export_mot: 无法读取模型参数设备: {e}")
		if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
			try:
				cuda_index = 0 if device in ("cuda", "cuda:0") else int(device.split(":")[1]) if ":" in device else int(device)
				torch.cuda.set_device(cuda_index)
				print(f"[DEBUG] export_mot: 已设置当前 CUDA 设备为 {torch.cuda.current_device()}")
				torch.backends.cudnn.benchmark = True
			except Exception as e:
				print(f"[WARN] export_mot: 设置 CUDA 设备失败: {e}")
		gen = model.track(
			source=source,
			stream=True,
			imgsz=imgsz,
			conf=conf,
			iou=iou,
			device=device,
			tracker=tracker_cfg,
			verbose=False,
			persist=False,
		)
		for r in gen:
			frame_idx += 1
			# 仅在首帧打印一次速度与显存情况，帮助确认 GPU 是否参与推理
			if frame_idx == 1:
				try:
					speed = getattr(r, 'speed', None)
					if speed:
						print(f"[DEBUG] speed(ms): preprocess={speed.get('preprocess', 'NA')}, inference={speed.get('inference', 'NA')}, postprocess={speed.get('postprocess', 'NA')}")
					a = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
					rsv = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
					print(f"[DEBUG] CUDA 显存: allocated={a/1024/1024:.1f}MB, reserved={rsv/1024/1024:.1f}MB")
				except Exception as e:
					print(f"[WARN] 获取速度/显存信息失败: {e}")
			boxes = getattr(r, "boxes", None)
			if boxes is None or boxes.data is None or len(boxes) == 0:
				continue
			ids = boxes.id
			xyxy = boxes.xyxy.cpu().numpy()
			confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
			ids_np = ids.cpu().numpy().astype(int) if ids is not None else None
			for i in range(xyxy.shape[0]):
				if ids_np is None:
					continue
				tid = int(ids_np[i])
				x1, y1, x2, y2 = xyxy[i].tolist()
				w, h = x2 - x1, y2 - y1
				sc = float(confs[i]) if confs is not None else 1.0
				# MOT格式：frame,id,left,top,width,height,conf,-1,-1,-1
				f.write(f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{sc:.4f},-1,-1,-1\n")
