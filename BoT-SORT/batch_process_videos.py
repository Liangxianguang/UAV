#!/usr/bin/env python3
"""
批量处理视频文件的脚本
自动遍历视频文件夹，逐个处理所有视频
"""

import subprocess
import os
from pathlib import Path
import time

def batch_process_videos():
    # 视频文件夹路径
    video_folder = Path("D:/UAV/YOLOv12-BoT-SORT-ReID/data/MultiUAV_Train/TrainVideos")
    
    # 获取所有视频文件
    video_files = sorted(list(video_folder.glob("*.mp4")))
    
    print(f"Found {len(video_files)} video files to process")
    
    # 基础命令
    base_cmd = [
        "python", "tools/inference.py",
        "--weights", "./yolov12/weights/v1/MOT_yolov12n.pt",
        "--img-size", "1600",
        "--track_buffer", "60",
        "--device", "0",
        "--agnostic-nms",
        "--save_path_answer", "D:/UAV/YOLOv12-BoT-SORT-ReID/TrackEval/data/trackers/mot_challenge/UAV-train/my_botsort/data",
        "--with-reid",
        "--fast-reid-config", "logs/sbs_S50/config.yaml",
        "--fast-reid-weights", "logs/sbs_S50/model_0016.pth",
        "--hide-labels-name",
        "--nosave"
    ]
    
    # 确保输出目录存在
    output_dir = Path("D:/UAV/YOLOv12-BoT-SORT-ReID/TrackEval/data/trackers/mot_challenge/UAV-train/my_botsort/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(video_files)}] Processing: {video_file.name}")
        print(f"{'='*60}")
        
        # 检查输出文件是否已存在（跳过已处理的）
        output_file = output_dir / f"{video_file.stem}.txt"
        if output_file.exists():
            print(f"⏭️ {video_file.name} already processed, skipping...")
            successful += 1
            continue
        
        # 构建完整命令
        cmd = base_cmd + ["--source", str(video_file)]
        
        # 显示正在运行的命令（用于调试）
        print(f"Running: {' '.join(cmd)}")
        
        try:
            # 运行命令
            video_start_time = time.time()
            result = subprocess.run(cmd, capture_output=False, text=True, cwd="D:/UAV/YOLOv12-BoT-SORT-ReID/BoT-SORT")
            video_duration = time.time() - video_start_time
            
            if result.returncode == 0:
                print(f"✅ {video_file.name} completed successfully in {video_duration:.1f}s")
                successful += 1
            else:
                print(f"❌ {video_file.name} failed with return code {result.returncode}")
                failed += 1
                # 不显示stderr以避免输出混乱，如果需要可以取消注释
                # if result.stderr:
                #     print(f"Error: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error processing {video_file.name}: {e}")
            failed += 1
        
        # 显示总体进度统计
        elapsed = time.time() - start_time
        avg_time = elapsed / i if i > 0 else 0
        remaining = (len(video_files) - i) * avg_time
        
        print(f"\n📊 Progress Statistics:")
        print(f"   Completed: {i}/{len(video_files)} ({i/len(video_files)*100:.1f}%)")
        print(f"   Successful: {successful}, Failed: {failed}")
        print(f"   Elapsed: {elapsed/60:.1f} min, ETA: {remaining/60:.1f} min")
        print(f"   Avg time per video: {avg_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\n🎉 Batch processing completed!")
    print(f"📈 Final Results:")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Successful: {successful}/{len(video_files)}")
    print(f"   Failed: {failed}/{len(video_files)}")
    print(f"   Success rate: {successful/len(video_files)*100:.1f}%")

if __name__ == "__main__":
    print("🚀 Starting batch video processing...")
    print("Press Ctrl+C to interrupt if needed")
    
    try:
        batch_process_videos()
    except KeyboardInterrupt:
        print("\n⛔ Processing interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")