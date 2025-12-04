@echo off
REM Batch script to run BoT-SORT inference on multiple UAVSwarm sequences
REM This script processes UAVSwarm-04, UAVSwarm-06, and UAVSwarm-08

setlocal enabledelayedexpansion

REM Configuration
set PYTHON_EXE=D:\download\anaconda3\envs\yolov12_botsort\python.exe
set WEIGHTS=yolov12\runs\uav\train17\weights\best.pt
set IMG_SIZE=1600
set TRACK_BUFFER=60
set DEVICE=0
set SAVE_PATH=..\test_results\inference_answers
set REID_CONFIG=logs\sbs_S50\config.yaml
set REID_WEIGHTS=logs\sbs_S50\model_0016.pth

echo.
echo ===== BoT-SORT Batch Inference Runner =====
echo Weights: %WEIGHTS%
echo Image Size: %IMG_SIZE%
echo Track Buffer: %TRACK_BUFFER%
echo ReID Config: %REID_CONFIG%
echo.

set COUNT=0
set BASE_PATH=..\data\UAVSwarm-dataset-master\test

REM Count total sequences first
for /d %%D in ("%BASE_PATH%\UAVSwarm-*") do (
    if exist "%%D\img1" set /a COUNT+=1
)
set TOTAL=%COUNT%
set COUNT=0

echo Total sequences found: %TOTAL%
echo.

REM Process each sequence directory
for /d %%D in ("%BASE_PATH%\UAVSwarm-*") do (
    set SEQUENCE=%%D\img1
    
    if exist "!SEQUENCE!" (
        set /a COUNT+=1
        
        REM Extract sequence name from path
        for %%F in ("%%D") do set SEQNAME=%%~nF
        
        REM Create sequence-specific output directory
        set SEQ_SAVE_PATH=%SAVE_PATH%\!SEQNAME!
        
        echo [!COUNT!/%TOTAL%] Processing: !SEQNAME!
        echo ===================================================================
        
        %PYTHON_EXE% tools\inference.py ^
          --weights "%WEIGHTS%" ^
          --source "!SEQUENCE!" ^
          --img-size %IMG_SIZE% ^
          --track_buffer %TRACK_BUFFER% ^
          --device %DEVICE% ^
          --agnostic-nms ^
          --save_path_answer "!SEQ_SAVE_PATH!" ^
          --with-reid ^
          --fast-reid-config "%REID_CONFIG%" ^
          --fast-reid-weights "%REID_WEIGHTS%" ^
          --hide-labels-name ^
          --save-txt
        
        if !ERRORLEVEL! equ 0 (
            echo ✓ !SEQNAME! completed successfully
            echo   Output: !SEQ_SAVE_PATH!\img1.txt
        ) else (
            echo ✗ !SEQNAME! failed with error code !ERRORLEVEL!
        )
        echo.
    )
)

echo.
echo ===================================================================
echo Batch processing complete. Results saved to: %SAVE_PATH%
echo ===================================================================
echo.

endlocal
