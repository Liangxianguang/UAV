#!/usr/bin/env python3
"""
批量导出 MOT 格式 - 用于多个图像序列
模仿 export_mot_all_videos.py 的结构
"""

import sys
import os
from pathlib import Path

# 确保使用本地的 ultralytics 模块而不是已安装的包
yolov12_dir = Path(__file__).parent.parent
sys.path.insert(0, str(yolov12_dir))

import argparse
import torch
import torch.nn as nn


def _patch_aattn_module():
    """
    修补已安装的 ultralytics 中的 AAttn 模块以支持两种格式
    """
    try:
        from ultralytics.nn.modules import block as installed_block
        
        # 检查是否存在 AAttn 类
        if not hasattr(installed_block, 'AAttn'):
            print("[信息] 找不到已安装的 AAttn 模块")
            return
        
        AAttn = installed_block.AAttn
        original_forward = AAttn.forward
        
        # 缓存：用于标记模块是否已检查过格式
        aattn_format_cache = {}
        
        def patched_forward(self, x):
            """修补的 forward 方法，支持两种 AAttn 格式"""
            module_id = id(self)
            
            # 检查是否已确定该模块的格式
            if module_id not in aattn_format_cache:
                # 首次调用时检查格式
                try:
                    # 尝试使用原始的 forward 方法
                    result = original_forward(self, x)
                    aattn_format_cache[module_id] = 'original'
                    return result
                except AttributeError as e:
                    if 'qkv' in str(e):
                        # qkv 不存在，标记为使用新格式 (qk+v)
                        if not hasattr(self, 'qk') or not hasattr(self, 'v'):
                            raise
                        aattn_format_cache[module_id] = 'alternate'
                        print(f"[信息] 检测到 AAttn 模块使用新格式 (qk+v)")
                    else:
                        raise
            
            # 根据缓存的格式选择执行路径
            if aattn_format_cache[module_id] == 'original':
                return original_forward(self, x)
            else:
                # 备用格式: 使用 qk 和 v
                # USE_FLASH_ATTN 和其他变量
                USE_FLASH_ATTN = getattr(installed_block, 'USE_FLASH_ATTN', False)
                flash_attn_func = getattr(installed_block, 'flash_attn_func', None) if USE_FLASH_ATTN else None
                
                B, C, H, W = x.shape
                N = H * W
                
                # 新格式: 使用 qk 和 v
                qk = self.qk(x).flatten(2).transpose(1, 2)
                v = self.v(x)
                pp = self.pe(v) if hasattr(self, 'pe') else None
                v = v.flatten(2).transpose(1, 2)
                
                if self.area > 1:
                    qk = qk.reshape(B * self.area, N // self.area, C * 2)
                    v = v.reshape(B * self.area, N // self.area, C)
                    B, N, _ = qk.shape
                q, k = qk.split([C, C], dim=2)
                
                if x.is_cuda and USE_FLASH_ATTN and flash_attn_func:
                    q = q.view(B, N, self.num_heads, self.head_dim)
                    k = k.view(B, N, self.num_heads, self.head_dim)
                    v = v.view(B, N, self.num_heads, self.head_dim)
                    
                    x = flash_attn_func(
                        q.contiguous().half(),
                        k.contiguous().half(),
                        v.contiguous().half()
                    ).to(q.dtype)
                else:
                    q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
                    k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
                    v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
                    
                    attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
                    max_attn = attn.max(dim=-1, keepdim=True).values
                    exp_attn = torch.exp(attn - max_attn)
                    attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
                    x = (v @ attn.transpose(-2, -1))
                    
                    x = x.permute(0, 3, 1, 2)
                
                if self.area > 1:
                    x = x.reshape(B // self.area, N * self.area, C)
                    B, N, _ = x.shape
                x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                
                if pp is not None:
                    return self.proj(x + pp)
                else:
                    return self.proj(x)
        
        # 替换 forward 方法
        AAttn.forward = patched_forward
        print("[信息] 成功修补 AAttn 模块")
        
    except Exception as e:
        print(f"[警告] 修补 AAttn 模块失败: {e}")


# 在导入 YOLO 之前修补模块
_patch_aattn_module()

# 现在导入 YOLO
from ultralytics import YOLO

# 导入 MOT 导出函数
try:
    from export_mot import export_video_to_mot_txt, ensure_dir
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from export_mot import export_video_to_mot_txt, ensure_dir


def to_seq_name_from_stem(stem: str) -> str:
    """
    从文件夹名映射到序列名
    - UAV123 -> UAVSwarm-123
    - UAVSwarm-123 -> UAVSwarm-123
    - 其他 -> 返回原名
    """
    s = stem
    if s.startswith("UAV") and s[3:].isdigit():
        num = int(s[3:])
        return f"UAVSwarm-{num:02d}"
    if s.startswith("UAVSwarm-"):
        return s
    return s


def main():
    parser = argparse.ArgumentParser(
        description="批量导出追踪结果（MOT 格式）用于多个图像序列"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="YOLO 权重文件路径 (.pt)"
    )
    parser.add_argument(
        "--sequences_root",
        type=str,
        required=True,
        help="包含所有序列子目录的根目录 (例如: .../test/)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        required=True,
        help="追踪器 YAML 路径 (例如: ultralytics/cfg/trackers/bytetrack.yaml 或 botsort.yaml)"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="MOT 格式输出文件的根目录 (每个序列生成 <seq_name>.txt)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1600,
        help="推理时的图像大小"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="检测置信度阈值"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="跟踪时的 IOU 阈值"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备 (例如: 0, 1, cuda:0, cpu)"
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="可选：仅处理指定的序列名 (如果不指定则处理所有序列)"
    )

    args = parser.parse_args()

    # 加载模型
    print(f"[加载] 权重文件: {args.weights}")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"[错误] 加载模型失败: {e}")
        print(f"[提示] 请检查:")
        print(f"  1. 权重文件是否存在: {args.weights}")
        print(f"  2. 权重文件是否损坏")
        print(f"  3. 本地 ultralytics 模块是否与模型兼容")
        return 1
    
    seq_root = Path(args.sequences_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    # 检查源目录是否存在
    if not seq_root.is_dir():
        print(f"[错误] 序列根目录不存在: {seq_root}")
        return 1

    # 枚举所有序列子目录
    subdirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    
    # 过滤：如果指定了 --include，只处理这些序列
    if args.include:
        include_names = set(args.include)
        subdirs = [p for p in subdirs if p.name in include_names]

    if not subdirs:
        print(f"[警告] 未找到任何序列目录")
        return 0

    print(f"[信息] 找到 {len(subdirs)} 个序列")
    print(f"[信息] 推理参数: imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, device={args.device}")
    print(f"[信息] 追踪器配置: {args.tracker}")
    print(f"[信息] 追踪器配置完整路径: {Path(args.tracker).absolute()}")

    # 顺序处理每个序列，不并行
    success_count = 0
    failed_count = 0
    
    for idx, seq_dir in enumerate(subdirs, 1):
        seq_name = to_seq_name_from_stem(seq_dir.name)
        out_txt = out_root / f"{seq_name}.txt"
        
        # 检查 img1 子目录（MOT Challenge 格式）
        img_dir = seq_dir / "img1"
        if not img_dir.exists():
            img_dir = seq_dir
        
        print(f"\n[{idx}/{len(subdirs)}] 处理 {seq_dir.name} -> {out_txt.name}")
        
        # 调用导出函数
        try:
            export_video_to_mot_txt(
                model=model,
                source=str(img_dir),
                out_txt=str(out_txt),
                tracker_cfg=args.tracker,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device
            )
            print(f"[✓] {seq_name} 已完成")
            success_count += 1
        except Exception as e:
            print(f"[✗] {seq_name} 处理失败: {e}")
            failed_count += 1
            continue

    print(f"\n" + "="*60)
    print(f"[完成] 所有序列处理完毕")
    print(f"[统计] 成功: {success_count}/{len(subdirs)}, 失败: {failed_count}/{len(subdirs)}")
    print(f"[输出] 结果保存在: {out_root}")
    print(f"="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
