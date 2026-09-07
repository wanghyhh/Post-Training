#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练过程绘制脚本：从训练日志 JSONL 文件中提取训练/评估指标，
使用 matplotlib 库生成多子图训练曲线图（对齐 example/plot_Train_SFT.py 风格）。

用法:
    python plot_sft.py
"""

import os
import sys
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 添加当前目录到 sys.path 以导入 tool_sft
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import yaml

# 配置 matplotlib 字体（对齐 example 风格：使用英文标签避免字体问题）
# 优先使用系统默认英文字体，确保数字和英文显示清晰
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'

# 从工具脚本导入
from tool_sft import load_config, print_info, get_logger


# =============================================================================
# 日志解析：从统一接口日志 train_log.jsonl 提取指标
# =============================================================================

def parse_train_log(log_file: str) -> Dict[str, Dict[str, List]]:
    """
    从训练日志 JSONL 文件中提取指标，按 example/plot_Train_SFT.py 风格组织数据。

    日志格式模仿 example/sft：每行就是 Trainer logs 字典原样，
    例如: {"loss": 2.77, "grad_norm": 0.35, "learning_rate": 2e-4, ...}
    训练步日志含 "loss"/"learning_rate" 键；评估日志含 "eval_loss" 键。

    遍历每行，根据键名区分训练/评估日志，为每个指标维护 {'step': [], 'epoch': [], 'train': [], 'eval': []}。

    Args:
        log_file: 训练日志 JSONL 文件路径

    Returns:
        包含每个指标四列表的字典:
        {
            "loss": {'step': [...], 'epoch': [...], 'train': [...], 'eval': [...]},
            "eval_loss": {'step': [...], 'epoch': [...], 'train': [...], 'eval': [...]},
            "learning_rate": {'step': [...], 'epoch': [...], 'train': [...], 'eval': [...]},
            "grad_norm": {'step': [...], 'epoch': [...], 'train': [...], 'eval': [...]},
            "entropy": {'step': [...], 'epoch': [...], 'train': [...], 'eval': [...]},
            "mean_token_accuracy": {'step': [...], 'epoch': [...], 'train': [...], 'eval': [...]},
        }
    """
    logger = get_logger("plot")

    # 参数映射：key=train 键名，value=eval 键名（None 表示无对应 eval）
    param_mappings = {
        'loss': 'eval_loss',
        'learning_rate': None,
        'grad_norm': None,
        'entropy': 'eval_entropy',
        'num_tokens': 'eval_num_tokens',
        'mean_token_accuracy': 'eval_mean_token_accuracy',
    }

    data = {train_key: {'step': [], 'epoch': [], 'train': [], 'eval': []}
            for train_key in param_mappings.keys()}

    if not os.path.exists(log_file):
        print_info(logger, f"警告：日志文件不存在：{log_file}")
        return data

    # 逐行解析：每行即 Trainer logs 原样
    line_count = 0
    train_count = 0
    eval_count = 0
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            line_count += 1

            step = entry.get("global_step")
            epoch = entry.get("epoch")
            if step is None and epoch is None:
                continue

            # 判断当前行是训练日志还是评估日志
            is_eval = entry.get("eval_loss") is not None

            for train_key, eval_key in param_mappings.items():
                train_val = entry.get(train_key)
                eval_val = entry.get(eval_key) if eval_key else None

                # 跳过 None 值（对齐 example 逻辑）
                if train_val is None and eval_val is None:
                    continue

                if step is not None:
                    data[train_key]['step'].append(step)
                if epoch is not None:
                    data[train_key]['epoch'].append(epoch)
                
                if is_eval:
                    # 评估日志：train 放 None，eval 放值
                    data[train_key]['train'].append(None)
                    data[train_key]['eval'].append(eval_val if eval_val is not None else train_val)
                    # 只在第一个匹配的训练键上计数（避免重复）
                    if train_key == 'loss':
                        eval_count += 1
                else:
                    # 训练日志：train 放值，eval 放 None
                    data[train_key]['train'].append(train_val)
                    data[train_key]['eval'].append(None)
                    # 只在第一个匹配的训练键上计数（避免重复）
                    if train_key == 'loss':
                        train_count += 1

    # 统计报告
    parts = ", ".join(f"{k}={len(v['train'])}" for k, v in data.items())
    print_info(logger, f"解析日志完成：{line_count} 行 (训练={train_count}, 评估={eval_count}, {parts})")
    return data


# =============================================================================
# 数据后处理：stride 降采样平滑
# =============================================================================

def smooth_data_stride(data: List[float], stride: int = 10) -> Tuple[List[float], List[float]]:
    """
    滑动窗口降采样平滑（对齐 example 风格）。

    Args:
        data: 原始数据列表
        stride: 滑动窗口大小（降采样步长）

    Returns:
        (smoothed_epochs, smoothed_vals) 降采样后的数据
    """
    if stride <= 1 or len(data) < stride:
        return list(range(len(data))), data

    arr = np.array(data)
    smoothed_vals = []
    smoothed_indices = []

    for i in range(0, len(arr), stride):
        window = arr[i : i + stride]
        if len(window) > 0:
            smoothed_vals.append(float(np.mean(window)))
            smoothed_indices.append(i)

    return smoothed_indices, smoothed_vals


# =============================================================================
# 绘图配置与绘制
# =============================================================================

# 预定义的绘图配置（按 log_data 键名顺序，对齐 example/plot_Train_SFT.py 风格）
PLOT_META = [
    ("loss", "Loss", "Epoch", "Loss", "#2563eb"),
    ("learning_rate", "Learning Rate", "Epoch", "LR", "#16a34a"),
    ("grad_norm", "Grad Norm", "Epoch", "Norm", "#7c3aed"),
    ("entropy", "Entropy", "Epoch", "Entropy", "#0891b2"),
    ("mean_token_accuracy", "Token Accuracy", "Epoch", "Accuracy", "#059669"),
]


def plot_training_curves(
    log_data: Dict[str, Dict[str, List]],
    output_path: str,
    ncols: int = 3,
    hspace: float = 0.4,
    wspace: float = 0.3,
    title_y: float = 0.95,
    start_index: int = 0,
    stride: int = 10,
) -> None:
    """
    绘制训练过程可视化曲线，保存为 PNG 图片（对齐 example 风格）。

    Args:
        log_data: 从日志解析出的指标数据，键为指标名，值为包含 step/epoch/train/eval 的字典
        output_path: 输出 PNG 图片文件路径
        ncols: 子图列数
        hspace: 子图垂直间距
        wspace: 子图水平间距
        title_y: 总标题 Y 轴位置（0~1）
        start_index: 从第几个数据点开始绘制（跳过前 N 步）
        stride: 采样步长/滑动窗口大小（>1 时降采样 + 绘制原始线）
    """
    logger = get_logger("plot")

    # 处理 start_index（跳过前 N 个数据点）
    for key in log_data:
        if start_index > 0 and len(log_data[key]['epoch']) > start_index:
            for k in log_data[key]:
                log_data[key][k] = log_data[key][k][start_index:]

    # 构建待绘制列表：(title, x_label, y_label, train_key, color)
    plot_configs = []
    for key, title, x_label, y_label, color in PLOT_META:
        if log_data.get(key) and len(log_data[key]['epoch']) > 0:
            plot_configs.append((key, title, x_label, y_label, color))

    plots_needed = len(plot_configs)
    if plots_needed == 0:
        print_info(logger, "警告：没有指标可绘制")
        return

    # 计算布局
    nrows = (plots_needed + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.reshape(-1) if plots_needed > 1 else [axes]

    # 逐个绘制子图
    for idx, (key, title, x_label, y_label, color) in enumerate(plot_configs):
        ax = axes[idx]
        x_vals = log_data[key]['epoch']
        train_vals = log_data[key]['train']
        eval_vals = log_data[key]['eval']

        # 清理 None 值
        train_clean = [v for v in train_vals if v is not None]
        train_epochs = [x for x, v in zip(x_vals, train_vals) if v is not None]
        eval_clean = [v for v in eval_vals if v is not None]
        eval_epochs = [x for x, v in zip(x_vals, eval_vals) if v is not None]

        ax.set_xlabel(x_label, fontsize=10)

        # --- 绘制 Train 曲线 (含降采样逻辑) ---
        if train_clean:
            if stride > 1:
                # 1. 绘制原始曲线 (极浅色)
                ax.plot(train_epochs, train_clean, 'b-', linewidth=1, label='Train (Raw)', alpha=0.2)

                # 2. 执行滑动窗口平均降采样
                smoothed_indices, smoothed_vals = smooth_data_stride(train_clean, stride)
                smoothed_epochs = [train_epochs[i] for i in smoothed_indices]

                ax.plot(smoothed_epochs, smoothed_vals, 'b-', linewidth=1.5, label=f'Train (Stride={stride})', alpha=0.8)
            else:
                ax.plot(train_epochs, train_clean, 'b-', linewidth=1.5, label='Train', alpha=0.8)

        # --- 绘制 Eval 曲线 ---
        if eval_clean:
            ax.plot(eval_epochs, eval_clean, 'r--.', linewidth=1.5, label='Eval', alpha=0.8)

        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    # 隐藏多余子图
    for idx in range(plots_needed, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.delaxes(axes[idx])

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    # 全局标题
    log_name = Path(log_file).stem if 'log_file' in locals() else "training"
    fig.suptitle(f"SFT Training Logs: {log_name}", fontsize=16, fontweight='bold', y=title_y)

    # 输出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print_info(logger, f"训练曲线图已保存至：{output_path}")
    print_info(logger, f"采样步长 (stride)：{stride}")
    print_info(logger, f"布局：{nrows} 行 × {ncols} 列")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主函数：解析训练日志并生成可视化曲线。

    步骤:
        1. 加载配置文件
        2. 解析训练日志 JSONL
        3. 绘制并保存多子图训练曲线
    """
    logger = get_logger("plot")
    config = load_config("Config.yaml")

    output_cfg = config["OutputConfig"]
    plot_cfg = config["PlotConfig"]

    print_info(logger, "=" * 60)
    print_info(logger, "训练过程可视化")
    print_info(logger, "=" * 60)

    # 固定读取统一接口日志 train_log.jsonl
    global log_file
    log_dir = output_cfg["train_log_dir"]
    log_file = os.path.join(log_dir, "train_log.jsonl")

    print_info(logger, f"日志文件：{log_file}")

    # 解析日志
    log_data = parse_train_log(log_file)

    if not any(log_data[k] for k in log_data):
        print_info(logger, "错误：未从日志中找到可绘制的指标数据")
        print_info(logger, "请先运行训练脚本生成日志")
        return

    # 绘制曲线
    plot_dir = output_cfg["plot_dir"]
    output_path = os.path.join(plot_dir, "training_curves.png")

    plot_training_curves(
        log_data=log_data,
        output_path=output_path,
        ncols=plot_cfg.get("ncols", 3),
        hspace=plot_cfg.get("hspace", 0.4),
        wspace=plot_cfg.get("wspace", 0.3),
        title_y=plot_cfg.get("title_y", 0.95),
        start_index=plot_cfg.get("start_index", 0),
        stride=plot_cfg.get("stride", 10),
    )

    print_info(logger, "=" * 60)
    print_info(logger, "可视化完成!")
    print_info(logger, "=" * 60)


if __name__ == "__main__":
    main()
