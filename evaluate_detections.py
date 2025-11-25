"""
比较模型推理结果和真实标注数据
计算检测精度指标: Precision, Recall, mAP50等
"""
import os
import numpy as np
from collections import defaultdict

def parse_detections(txt_file):
    """
    解析检测结果文件 (MOT格式)
    格式: frame_id, track_id, x, y, w, h, conf, -1, -1, -1
    返回: {frame_id: [(x, y, w, h, conf), ...]}
    """
    detections = defaultdict(list)
    
    if not os.path.exists(txt_file):
        print(f"⚠️  File not found: {txt_file}")
        return detections
    
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame_id = int(parts[0])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                
                detections[frame_id].append({
                    'x': x, 'y': y, 'w': w, 'h': h, 'conf': conf
                })
    
    return detections


def iou(box1, box2):
    """
    计算两个边界框的IOU (Intersection over Union)
    box format: {x, y, w, h}
    """
    # 转换为 (x1, y1, x2, y2) 格式
    x1_min, y1_min = box1['x'], box1['y']
    x1_max, y1_max = x1_min + box1['w'], y1_min + box1['h']
    
    x2_min, y2_min = box2['x'], box2['y']
    x2_max, y2_max = x2_min + box2['w'], y2_min + box2['h']
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    box1_area = box1['w'] * box1['h']
    box2_area = box2['w'] * box2['h']
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_detections(pred_file, gt_file, iou_threshold=0.5):
    """
    评估检测结果
    """
    print("\n" + "="*60)
    print("📊 DETECTION EVALUATION")
    print("="*60)
    
    # 解析文件
    predictions = parse_detections(pred_file)
    ground_truth = parse_detections(gt_file)
    
    print(f"\n📁 Prediction file: {pred_file}")
    print(f"   Total frames: {len(predictions)}")
    total_pred = sum(len(boxes) for boxes in predictions.values())
    print(f"   Total detections: {total_pred}")
    
    print(f"\n📁 Ground truth file: {gt_file}")
    print(f"   Total frames: {len(ground_truth)}")
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    print(f"   Total ground truth boxes: {total_gt}")
    
    # 获取所有帧
    all_frames = set(predictions.keys()) | set(ground_truth.keys())
    print(f"\n📈 Total frames to evaluate: {len(all_frames)}")
    
    # 计算TP, FP, FN
    tp = 0
    fp = 0
    fn = 0
    
    matched_pred = defaultdict(set)  # 记录哪些预测被匹配
    matched_gt = defaultdict(set)    # 记录哪些GT被匹配
    
    for frame_id in sorted(all_frames):
        preds = predictions.get(frame_id, [])
        gts = ground_truth.get(frame_id, [])
        
        # 排序（按置信度降序）
        preds_sorted = sorted(preds, key=lambda x: x['conf'], reverse=True)
        
        # 为每个预测找最好的匹配GT
        for pred_idx, pred in enumerate(preds_sorted):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt[frame_id]:
                    continue  # 这个GT已经被匹配过
                
                curr_iou = iou(pred, gt)
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_gt_idx = gt_idx
            
            # 判断是否为TP或FP
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_pred[frame_id].add(pred_idx)
                matched_gt[frame_id].add(best_gt_idx)
            else:
                fp += 1
        
        # 未匹配的GT为FN
        fn += len(gts) - len(matched_gt[frame_id])
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n" + "-"*60)
    print(f"📊 METRICS (IOU threshold: {iou_threshold})")
    print("-"*60)
    print(f"True Positives (TP):   {tp}")
    print(f"False Positives (FP):  {fp}")
    print(f"False Negatives (FN):  {fn}")
    print(f"\nPrecision: {precision:.4f} ({tp}/{tp+fp})")
    print(f"Recall:    {recall:.4f} ({tp}/{tp+fn})")
    print(f"F1-Score:  {f1:.4f}")
    
    # 检测率
    detection_rate = tp / total_gt if total_gt > 0 else 0
    print(f"\nDetection Rate: {detection_rate:.2%} ({tp}/{total_gt})")
    
    # 误检率
    false_alarm_rate = fp / total_pred if total_pred > 0 else 0
    print(f"False Alarm Rate: {false_alarm_rate:.2%} ({fp}/{total_pred})")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'detection_rate': detection_rate
    }


def compare_frame_by_frame(pred_file, gt_file, sample_frames=10):
    """
    逐帧对比
    """
    print("\n" + "="*60)
    print("🔍 FRAME-BY-FRAME COMPARISON")
    print("="*60)
    
    predictions = parse_detections(pred_file)
    ground_truth = parse_detections(gt_file)
    
    all_frames = sorted(set(predictions.keys()) | set(ground_truth.keys()))
    
    # 随机选择样本帧
    if len(all_frames) > sample_frames:
        sample_indices = np.random.choice(len(all_frames), sample_frames, replace=False)
        sample_frames_list = [all_frames[i] for i in sorted(sample_indices)]
    else:
        sample_frames_list = all_frames[:sample_frames]
    
    print(f"\n显示 {len(sample_frames_list)} 个样本帧的对比:\n")
    
    for frame_id in sample_frames_list:
        preds = predictions.get(frame_id, [])
        gts = ground_truth.get(frame_id, [])
        
        print(f"Frame {frame_id:06d}:")
        print(f"  Predictions: {len(preds):3d} boxes")
        print(f"  Ground Truth: {len(gts):3d} boxes")
        
        if len(preds) == 0 and len(gts) > 0:
            print(f"  ⚠️  MISS! Expected {len(gts)} detections")
        elif len(preds) > 0 and len(gts) == 0:
            print(f"  ⚠️  FALSE ALARM! Detected {len(preds)} boxes but GT is empty")
        elif len(preds) == len(gts):
            print(f"  ✅ Count matches")
        else:
            print(f"  ⚠️  Count mismatch: {len(preds)} vs {len(gts)}")
    
    return sample_frames_list


if __name__ == '__main__':
    # 文件路径
    pred_file = r'D:\UAV\YOLOv12-BoT-SORT-ReID\test_results\UAVSwarm-44\detections.txt'
    gt_file = r'D:\UAV\YOLOv12-BoT-SORT-ReID\data\UAVSwarm-dataset-master\test\UAVSwarm-44\det\det.txt'
    
    print("\n🚀 Detection Evaluation Script")
    print(f"Comparing prediction vs ground truth")
    
    # 验证文件存在
    if not os.path.exists(pred_file):
        print(f"❌ Prediction file not found: {pred_file}")
        exit(1)
    
    if not os.path.exists(gt_file):
        print(f"❌ Ground truth file not found: {gt_file}")
        exit(1)
    
    # 评估
    metrics = evaluate_detections(pred_file, gt_file, iou_threshold=0.5)
    
    # 逐帧对比
    compare_frame_by_frame(pred_file, gt_file, sample_frames=15)
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)
