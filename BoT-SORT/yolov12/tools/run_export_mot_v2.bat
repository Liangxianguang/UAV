@echo off
REM 清除旧的追踪结果
echo [清理] 删除旧的追踪结果...
if exist "..\..\test_results\mot_results\BoT-SORT\data" rmdir /s /q "..\..\test_results\mot_results\BoT-SORT\data"
if exist "..\..\test_results\mot_results\Bytetrack\data" rmdir /s /q "..\..\test_results\mot_results\Bytetrack\data"
mkdir "..\..\test_results\mot_results\BoT-SORT\data"
mkdir "..\..\test_results\mot_results\Bytetrack\data"
echo [完成] 清理完毕

REM 第一次运行: BoT-SORT (使用改进版本，每个序列创建新模型实例)
echo.
echo ============================================================
echo 第一次运行: BoT-SORT 追踪器 (v2 - 每序列新模型)
echo ============================================================
python export_mot_batch_v2.py ^
  --weights "runs/uav/train17/weights/best.pt" ^
  --sequences_root "..\..\data\UAVSwarm-dataset-master\test" ^
  --tracker "ultralytics/cfg/trackers/botsort.yaml" ^
  --out_root "..\..\test_results\mot_results\BoT-SORT\data" ^
  --imgsz 1600 ^
  --conf 0.25 ^
  --iou 0.7 ^
  --device 0

echo.
echo ============================================================
echo 第二次运行: ByteTrack 追踪器 (v2 - 每序列新模型)
echo ============================================================
python export_mot_batch_v2.py ^
  --weights "runs/uav/train17/weights/best.pt" ^
  --sequences_root "..\..\data\UAVSwarm-dataset-master\test" ^
  --tracker "ultralytics/cfg/trackers/bytetrack.yaml" ^
  --out_root "..\..\test_results\mot_results\Bytetrack\data" ^
  --imgsz 1600 ^
  --conf 0.25 ^
  --iou 0.7 ^
  --device 0

echo.
echo ============================================================
echo 完成! 追踪结果已保存
echo ============================================================
pause
