# ====================================================
# GRPO 训练主脚本
# 用途：读取 Config.yaml，完成数据集加载、LoRA 配置、
#       训练器初始化与训练、最终模型保存。
# 用法：
#   本地 Windows（conda cuda118env）:
#       python train_grpo.py
#   Linux 服务器（多 GPU）:
#       accelerate launch --config_file accelerate_config.yaml train_grpo.py
# ====================================================

import os
import sys
import json
import time
import torch
import traceback
from pathlib import Path
from datetime import datetime

# Windows GBK 编码修复
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 禁用 tqdm 进度条（防止 stderr 覆盖 stdout 日志）
os.environ['TQDM_DISABLE'] = '1'

# ============================================================
# 依赖导入区
# ★ 注意：Windows conda 环境 0xC0000005 crash 防护
#   必须在任何 transformers/trl 导入之前先导入 pandas
# ============================================================
import pandas  # noqa: F401

# ★ 关键修复：劫持 tqdm.tqdm 类使其不输出进度条（防止 stderr 覆盖 stdout 日志）
#   必须在导入 datasets（会导入 tqdm）之前设置
#   保留 tqdm 模块结构，只替换 tqdm.tqdm 类为无输出版本
import sys as _sys
import types as _types

class _SilentTqdm:
    """静默 tqdm：保持接口但无任何输出"""
    def __init__(self, iterable=None, desc=None, total=None, n=0, leave=False, file=None,
                 ratemin=0.1, initial=0, position=None, ascii=None, disable=False,
                 unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.1,
                 bar_format=None, postfix=None, delay=0.0, ncols=0,
                 colour=None, bar_format_custom=None, mininterval=0.1, maxinterval=10.0,
                 miniters=None, ascii_desc=None, **kwargs):
        pass
    def __iter__(self):
        return iter([])
    def __call__(self, *args, **kwargs):
        return _SilentTqdm()
    def update(self, n=1):
        pass
    def close(self):
        pass
    def set_description(self, desc=None):
        pass
    def set_postfix(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# 用 _SilentTqdm 替换 tqdm.tqdm 类
_tqdm_module = _sys.modules.get('tqdm')
if _tqdm_module is not None:
    _tqdm_module.tqdm = _SilentTqdm
    if hasattr(_tqdm_module, 'auto') and hasattr(_tqdm_module.auto, 'tqdm'):
        _tqdm_module.auto.tqdm = _SilentTqdm
else:
    # tqdm 尚未导入，先创建一个占位模块让后续导入能正常工作
    _placeholder = _types.ModuleType('tqdm')
    _placeholder.tqdm = _SilentTqdm
    _placeholder.auto = _types.ModuleType('tqdm.auto')
    _placeholder.auto.tqdm = _SilentTqdm
    _placeholder.std = _types.ModuleType('tqdm.std')
    _placeholder.std.tqdm = _SilentTqdm
    _placeholder.contrib = _types.ModuleType('tqdm.contrib')
    _placeholder.contrib.concurrent = _types.ModuleType('tqdm.contrib.concurrent')
    # thread_map 和 process_map
    def _noop_thread_map(*args, **kwargs):
        return []
    def _noop_process_map(*args, **kwargs):
        return []
    _placeholder.contrib.concurrent.thread_map = _noop_thread_map
    _placeholder.contrib.concurrent.process_map = _noop_process_map
    _sys.modules['tqdm'] = _placeholder
    _sys.modules['tqdm.auto'] = _placeholder.auto
    _sys.modules['tqdm.std'] = _placeholder.std
    _sys.modules['tqdm.contrib'] = _placeholder.contrib
    _sys.modules['tqdm.contrib.concurrent'] = _placeholder.contrib.concurrent

# 将本脚本所在目录加入 sys.path，确保导入本地模块而非第三方库
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 标准库导入（pandas 之后）
import yaml

# 第三方库导入
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl
from trl import GRPOTrainer, GRPOConfig
from accelerate import Accelerator

# 本地模块导入
from tool_grpo import (
    ConfigLoader,
    load_dataset,
    validate_dataset_format,
    setup_lora,
    make_reward_function,
    save_peft_model,
    load_hf_model,
)

# 模型/分词器导入（在 pandas 之后、本地模块之前）
from transformers import AutoTokenizer
from reward_function import GRPORewardFunction


# ============================================================
# 自定义回调：训练日志（JSONL 双文件机制）
# ============================================================

class TrainingLoggerCallback(TrainerCallback):
    """
    训练日志回调。
    - 文件名格式：train_log_YYYYMMDD_HHMMSS.jsonl
    - 同时维护 train_log.jsonl 统一接口文件（追加写入）
    - 仅主进程写入日志（多 GPU 场景）
    - 日志保存至 output/logs/ 目录（对齐 SFT 输出结构）
    """

    def __init__(self, output_dir: str = "output/logs"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_timestamped = os.path.join(
            output_dir, f"train_log_{self.timestamp}.jsonl"
        )
        self.file_unified = os.path.join(output_dir, "train_log.jsonl")

        # 打开文件句柄
        os.makedirs(output_dir, exist_ok=True)
        self.f_timestamped = open(self.file_timestamped, "a", encoding="utf-8")
        self.f_unified = open(self.file_unified, "a", encoding="utf-8")

        self.log_entries = []  # 内存中累积所有日志

        print(f"\n{'=' * 60}")
        print(f"训练日志回调已启用")
        print(f"  时间戳文件: {self.file_timestamped}")
        print(f"  统一接口文件: {self.file_unified}")
        print(f"{'=' * 60}\n")

    def _is_main(self, args=None, state=None):
        """判断是否为主进程（多 GPU 场景下仅主进程写入日志）"""
        if args and hasattr(args, "is_main_process") and args.is_main_process:
            return True
        if state and hasattr(state, "local_rank"):
            return state.local_rank == 0
        return True  # 单 GPU 场景默认为主进程

    def on_init_end(self, args, state, control, **kwargs):
        """初始化结束时钩子（transformers 5.x 回调机制要求）"""
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时钩子"""
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每步日志触发"""
        if not logs or not self._is_main(args, state):
            return

        logs["global_step"] = state.global_step
        log_line = json.dumps(logs, ensure_ascii=False)

        # 写入两个文件
        self.f_timestamped.write(log_line + "\n")
        self.f_timestamped.flush()
        self.f_unified.write(log_line + "\n")
        self.f_unified.flush()
        self.log_entries.append(logs)

        # 终端输出关键指标
        if "loss" in logs:
            lr = logs.get("learning_rate", "N/A")
            reward = logs.get("reward", "N/A")
            kl = logs.get("kl", "N/A")
            print(
                f"  step={state.global_step} | "
                f"loss={logs.get('loss', 'N/A'):.4f} | "
                f"lr={lr} | "
                f"reward={reward} | "
                f"kl={kl}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时关闭文件"""
        if self._is_main(args, state):
            self.f_timestamped.close()
            self.f_unified.close()
            print(f"\n{'=' * 60}")
            print(f"训练完成！总步数: {state.global_step}")
            print(f"日志已保存:")
            print(f"  {self.file_timestamped}")
            print(f"  {self.file_unified}")
            print(f"{'=' * 60}\n")

    def __del__(self):
        """析构时确保文件关闭"""
        if hasattr(self, "f_timestamped") and not self.f_timestamped.closed:
            self.f_timestamped.close()
        if hasattr(self, "f_unified") and not self.f_unified.closed:
            self.f_unified.close()


# ============================================================
# 主流程
# ============================================================

def main():
    """
    GRPO 训练主函数。

    流程：
    1. 加载 Config.yaml 配置
    2. 加载数据集并验证
    3. 预处理数据集（应用 chat template → prompt + reference）
    4. 加载基座模型并配置 LoRA
    5. 实例化奖励函数
    6. 初始化 GRPOTrainer
    7. 训练模型
    8. 保存最终模型
    """

    print("=" * 70)
    print("  GRPO 训练管线")
    print("=" * 70)
    t_start = time.time()

    # -------------------------------------------
    # Step 1: 加载配置
    # -------------------------------------------
    config_path = os.path.join(_SCRIPT_DIR, "Config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    print(f"\n[1/8] 加载配置: {config_path}")
    loader = ConfigLoader(config_path)
    all_config = loader.load_all()

    model_cfg = all_config.get("ModelConfig", {})
    data_cfg = all_config.get("DataSetConfig", {})
    train_cfg = all_config.get("TrainConfig", {})
    lora_cfg = all_config.get("LoraParamConfig", {})
    reward_cfg = all_config.get("RewardFuncConfig", {})
    eval_cfg = all_config.get("EvalConfig", {})

    print("  ✓ 配置加载完成")

    # 打印关键配置摘要
    print(f"\n  基座模型: {model_cfg.get('model_name_or_path', 'N/A')}")
    # 兼容两种配置风格：平铺键名 vs 嵌套字典
    _train_ds = data_cfg.get("train_data_path") or data_cfg.get("train", {}).get("path", "")
    _valid_ds = data_cfg.get("valid_data_path") or data_cfg.get("valid", {}).get("path", "")
    print(f"  训练集: {_train_ds or 'N/A'}")
    print(f"  验证集: {_valid_ds or 'N/A'}")
    print(f"  输出目录: {train_cfg.get('output_dir', 'N/A')}")
    print(f"  LoRA: r={lora_cfg.get('r', 'N/A')}, alpha={lora_cfg.get('lora_alpha', 'N/A')}")
    print(f"  训练轮数: {train_cfg.get('num_train_epochs', 'N/A')}")

    # -------------------------------------------
    # Step 2: 加载数据集
    # -------------------------------------------
    print(f"\n[2/8] 加载数据集")
    # 兼容两种配置风格：
    #  ① 平铺键名: train_data_path / valid_data_path
    #  ② 嵌套字典: train.path / valid.path
    train_data_path = data_cfg.get("train_data_path", "")
    if not train_data_path:
        train_data_path = data_cfg.get("train", {}).get("path", "")

    valid_data_path = data_cfg.get("valid_data_path", "")
    if not valid_data_path:
        valid_data_path = data_cfg.get("valid", {}).get("path", "")

    # 获取验证集划分比例
    validation_split_ratio = data_cfg.get("validation_split_ratio", 0.0)

    if not train_data_path or not os.path.exists(train_data_path):
        raise FileNotFoundError(f"训练数据集不存在: {train_data_path}")

    prompts, answers, references = load_dataset(train_data_path)
    full_dataset = Dataset.from_dict({"prompt": prompts, "answer": answers, "reference": references})
    print(f"  原始数据集: {len(full_dataset)} 条样本")

    # 工作文档第 3.1 条：从训练数据集中按 validation_split_ratio 划分验证集
    if valid_data_path and os.path.exists(valid_data_path):
        # 优先使用独立的验证集文件
        eval_prompts, eval_answers, eval_refs = load_dataset(valid_data_path)
        eval_dataset = Dataset.from_dict({
            "prompt": eval_prompts,
            "answer": eval_answers,
            "reference": eval_refs,
        })
        # 使用剩余部分作为训练集
        train_indices = list(range(len(full_dataset)))
        eval_indices = list(range(len(eval_dataset)))
        # 合并索引重建训练集（排除验证集样本）
        eval_set_set = set(eval_indices)
        train_indices = [i for i in train_indices if i not in eval_set_set]
        train_dataset = full_dataset.select(train_indices)
        print(f"  训练集: {len(train_dataset)} 条样本（来自独立验证集文件，剩余部分）")
        print(f"  验证集: {len(eval_dataset)} 条样本（独立文件）")
    elif validation_split_ratio > 0 and validation_split_ratio < 1.0:
        # 按比例自动划分验证集
        print(f"  按 validation_split_ratio={validation_split_ratio} 自动划分验证集...")
        dataset_splits = full_dataset.train_test_split(test_size=validation_split_ratio, seed=42)
        train_dataset = dataset_splits["train"]
        eval_dataset = dataset_splits["test"]
        print(f"  训练集: {len(train_dataset)} 条样本")
        print(f"  验证集: {len(eval_dataset)} 条样本（自动划分）")
    else:
        # 无验证集划分
        train_dataset = full_dataset
        eval_dataset = None
        print(f"  训练集: {len(train_dataset)} 条样本（无验证集划分）")

    # -------------------------------------------
    # Step 3: 预处理数据集（应用 chat template）
    # -------------------------------------------
    print(f"\n[3/8] 预处理数据集（应用 chat template）")

    system_prompt = model_cfg.get("system_prompt", "你是一个有用的 AI 助手。")

    # 先加载 tokenizer 用于预处理
    print(f"  正在加载分词器: {model_cfg.get('model_name_or_path', 'N/A')}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def apply_chat_template(examples):
        """对每条样本应用 chat template 生成 prompt"""
        prompts_out = []
        # 兼容两种格式：有 answer 和无 answer
        has_answer = "answer" in examples
        for msg_list, reference in zip(examples["messages"], examples.get("reference", examples["answer"] if has_answer else [])):
            # 构造 system + user 消息
            system_msg = {"role": "system", "content": system_prompt}
            user_msg = msg_list[0]  # 第一条为用户消息
            prompt_messages = [system_msg, user_msg]

            # 应用 tokenizer 的 chat template
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts_out.append(prompt_text)

        # 返回 reference，兼容无 answer 的情况
        if has_answer:
            ref_list = examples.get("reference", examples["answer"])
        else:
            ref_list = examples.get("reference", [])

        return {"prompt": prompts_out, "reference": ref_list}

    # 确保数据集包含 messages 列
    if "messages" not in train_dataset.column_names:
        # 从 prompt/answer 列重建 messages
        columns_to_remove = [col for col in ["prompt", "answer"] if col in train_dataset.column_names]
        has_answer_col = "answer" in train_dataset.column_names
        
        if has_answer_col:
            # 完整格式: prompt + answer + reference
            train_dataset = train_dataset.map(
                lambda x: {
                    "messages": [
                        {"role": "user", "content": x["prompt"]},
                        {"role": "assistant", "content": x["answer"]},
                    ],
                    "reference": x["reference"] if "reference" in x else x["answer"],
                },
                remove_columns=columns_to_remove,
            )
        else:
            # 精简格式: prompt + reference（无 answer）
            train_dataset = train_dataset.map(
                lambda x: {
                    "messages": [
                        {"role": "user", "content": x["prompt"]},
                        {"role": "assistant", "content": ""},  # 空占位符
                    ],
                    "reference": x["reference"] if "reference" in x else "",
                },
                remove_columns=columns_to_remove,
            )

    train_dataset = train_dataset.map(
        apply_chat_template,
        batched=True,
        remove_columns=["messages"],
    )

    if eval_dataset is not None and "messages" in eval_dataset.column_names:
        eval_dataset = eval_dataset.map(
            apply_chat_template,
            batched=True,
            remove_columns=["messages"],
        )

    print(f"  ✓ 预处理完成")

    # -------------------------------------------
    # Step 4: 加载模型并配置 LoRA
    # -------------------------------------------
    print(f"\n[4/8] 加载模型并配置 LoRA")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": eval_cfg.get("torch_dtype", torch.float16),
    }
    if eval_cfg.get("load_in_8bit", False):
        model_kwargs["load_in_8bit"] = True
    if eval_cfg.get("load_in_4bit", False):
        model_kwargs["load_in_4bit"] = True

    _, model = setup_lora(
        model_name_or_path=model_cfg["model_name_or_path"],
        lora_config=lora_cfg,
        **model_kwargs,
    )
    print(f"  ✓ LoRA 配置完成")

    # 启用 require_grads（PEFT 兼容）
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # -------------------------------------------
    # Step 5: 实例化奖励函数
    # -------------------------------------------
    print(f"\n[5/8] 实例化奖励函数")
    reward_fn = make_reward_function(reward_cfg)
    print(f"  ✓ 奖励函数配置: weights={reward_fn.weights}")

    # 测试奖励函数（单样本）
    if len(prompts) > 0:
        test_reward = reward_fn(
            prompts=[prompts[0]],
            completions=[f"{tokenizer.eos_token}测试输出"],
            references=[references[0]],
        )
        print(f"  奖励函数测试: total_reward={test_reward}")

    # -------------------------------------------
    # Step 6: 初始化 GRPOTrainer
    # -------------------------------------------
    print(f"\n[6/8] 初始化 GRPOTrainer")

    # 创建训练参数（GRPOConfig）
    train_args = GRPOConfig(**train_cfg)

    # 创建回调（从 OutputConfig 读取路径，对齐 SFT 规范）
    output_cfg = all_config.get("OutputConfig", {})
    log_dir = output_cfg.get("train_log_dir", "output/logs")
    checkpoint_dir = output_cfg.get("checkpoint_dir", "output/checkpoints")
    final_model_dir = output_cfg.get("final_model_dir", "output/final_model")
    
    # 打印目录结构摘要
    output_root = output_cfg.get("base_dir", "output")
    eval_dir = output_cfg.get("eval_dir", "output/eval")
    plot_dir = output_cfg.get("plot_dir", "output/plots")
    print(f"\n  输出目录结构:")
    print(f"    训练日志:    {log_dir}")
    print(f"    检查点:      {checkpoint_dir}")
    print(f"    最终模型:    {final_model_dir}")
    print(f"    评估结果:    {eval_dir}")
    print(f"    绘图输出:    {plot_dir}")

    log_callback = TrainingLoggerCallback(log_dir)

    # 初始化 Trainer
    # 注意：GRPOTrainer 内部要求 tokenizer.padding_side == 'right'
    tokenizer.padding_side = "right"

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[log_callback],
    )

    # 禁用 tqdm 进度条（防止 stderr 覆盖 stdout 日志）
    # 保留训练日志回调的 step 输出
    trainer.args.disable_tqdm = True

    print(f"  ✓ Trainer 初始化完成")

    # -------------------------------------------
    # Step 7: 训练模型
    # -------------------------------------------
    print(f"\n[7/8] 开始训练")
    print(f"{'=' * 70}")

    try:
        trainer.train()
    except Exception as e:
        traceback.print_exc()
        print(f"\n✗ 训练异常: {e}")
        raise

    print(f"\n{'=' * 70}")
    t_total = time.time() - t_start
    print(f"✓ 训练完成！总耗时: {t_total:.1f} 秒")

    # -------------------------------------------
    # Step 8: 保存最终模型
    # -------------------------------------------
    print(f"\n[8/8] 保存最终模型")

    # 保存 LoRA adapter（仅适配器权重，对齐 SFT output/final_model 结构）
    save_peft_model(model, tokenizer, final_model_dir)
    print(f"  ✓ LoRA adapter 已保存至: {final_model_dir}")

    print(f"\n{'=' * 70}")
    print(f"训练管线全部完成！总耗时: {t_total:.1f} 秒")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
