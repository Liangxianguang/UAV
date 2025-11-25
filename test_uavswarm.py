"""
用训练好的模型测试UAVSwarm-02图像序列
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加yolov12路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BoT-SORT/yolov12'))

from ultralytics import YOLO

def test_images_in_folder(model_path, image_folder, output_folder, conf_threshold=0.3):
    """
    对文件夹中的所有图像进行检测
    
    Args:
        model_path: 训练好的模型权重路径
        image_folder: 输入图像文件夹
        output_folder: 输出结果文件夹
        conf_threshold: 置信度阈值
    """
    
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    results_txt = os.path.join(output_folder, 'detections.txt')
    
    # 加载模型
    print(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"✅ Model loaded successfully")
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = sorted([f for f in os.listdir(image_folder) 
                         if any(f.lower().endswith(ext) for ext in image_extensions)])
    
    print(f"\n📁 Found {len(image_files)} images in {image_folder}")
    
    # 检测结果列表
    detections = []
    
    # 处理每张图像
    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(image_folder, img_file)
            
            # 进行检测
            results = model.predict(img_path, conf=conf_threshold, verbose=False)
            
            # 读取图像用于绘制
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            # 处理检测结果
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                # 绘制检测框并保存结果
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().item()
                        cls = int(box.cls[0].cpu().item())
                        
                        # 保存为MOT格式: frame_id, -1, x, y, w, h, conf, -1, -1, -1
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1
                        detections.append(f"{idx},-1,{x1:.1f},{y1:.1f},{bbox_w:.1f},{bbox_h:.1f},{conf:.3f},-1,-1,-1")
                        
                        # 绘制框
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(img, f'conf: {conf:.2f}', (int(x1), int(y1) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 保存可视化结果
            output_img = os.path.join(output_folder, f'vis_{img_file}')
            cv2.imwrite(output_img, img)
            
            pbar.update(1)
    
    # 保存检测结果为txt
    print(f"\n💾 Saving detection results to: {results_txt}")
    with open(results_txt, 'w') as f:
        for det in detections:
            f.write(det + '\n')
    
    print(f"✅ Detection complete!")
    print(f"   - Total detections: {len(detections)}")
    print(f"   - Output folder: {output_folder}")
    print(f"   - Detection file: {results_txt}")
    
    return detections


if __name__ == '__main__':
    # 配置参数
    model_path = r'D:\UAV\YOLOv12-BoT-SORT-ReID\BoT-SORT\yolov12\runs\uav\train15\weights\best.pt'
    image_folder = r'D:\UAV\YOLOv12-BoT-SORT-ReID\data\UAVSwarm-dataset-master\test\UAVSwarm-44\img1'
    output_folder = r'D:\UAV\YOLOv12-BoT-SORT-ReID\test_results\UAVSwarm-44'
    
    # 检查输入路径
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(image_folder):
        print(f"❌ Image folder not found: {image_folder}")
        sys.exit(1)
    
    # 运行检测
    test_images_in_folder(
        model_path=model_path,
        image_folder=image_folder,
        output_folder=output_folder,
        conf_threshold=0.3  # 置信度阈值
    )
