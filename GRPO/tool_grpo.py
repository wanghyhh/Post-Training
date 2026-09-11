# ====================================================
# GRPO 工具模块（函数式）
# 用途：加载数据集、配置 LoRA、保存适配器、评估等
# 说明：训练和评估脚本应通过本模块的函数调用完成通用操作
# ====================================================

import os
import json
import traceback
from datetime import datetime

import pandas  # Windows 平台 0xC0000005 crash 防护（必须先于 transformers 导入）
import yaml
import torch
from datasets import load_dataset as hf_load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from reward_function import GRPORewardFunction


# 精度字符串 → torch.dtype 映射（供 ModelConfig.torch_dtype 使用）
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_torch_dtype(value) -> torch.dtype:
    """
    将配置中的精度描述解析为 torch.dtype。

    Args:
        value: 精度字符串（"bfloat16"/"float16"/"float32" 及常见缩写）
              或已经是 torch.dtype 的对象。缺省返回 torch.bfloat16。

    Returns:
        对应的 torch.dtype 对象。
    """
    if isinstance(value, torch.dtype):
        return value
    if not value:
        return torch.bfloat16
    key = str(value).strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"不支持的 torch_dtype 配置值: {value}（可选 {list(_DTYPE_MAP)}）")
    return _DTYPE_MAP[key]


# ================== 配置加载 ==================


class ConfigLoader:
    """
    配置加载与合并工具类。

    读取 Config.yaml 中所有配置节，合并返回一个 Python 字典。
    提供按节读取和全量加载两种方式。

    示例：
        >>> loader = ConfigLoader("Config.yaml")
        >>> train_cfg = loader.get_section("TrainConfig")
        >>> all_cfg = loader.load_all()
    """

    def __init__(self, config_path: str):
        """
        初始化配置加载器。

        Args:
            config_path: Config.yaml 文件路径。

        Raises:
            FileNotFoundError: 配置文件不存在。
        """
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.raw_config = yaml.safe_load(f)

    def get_section(self, section_name: str) -> dict:
        """
        读取指定配置节的内容。

        Args:
            section_name: 配置节名称（如 "TrainConfig"）。

        Returns:
            该配置节的字典内容。
        """
        return self.raw_config.get(section_name, {})

    def load_all(self) -> dict:
        """
        加载完整配置文件的所有配置节。

        Returns:
            包含所有配置节的字典。
        """
        return dict(self.raw_config)


# ================== 训练日志回调 ==================


class TrainingLoggerCallback(TrainerCallback):
    """
    训练日志回调（JSONL 双文件机制）。

    职责：
        1. 每次训练写入独立的时间戳文件 ``train_log_YYYYMMDD_HHMMSS.jsonl``（追加）。
        2. 同步维护统一接口文件 ``train_log.jsonl``，采用**清空后重写**，
           使其始终只包含最新一次训练的完整日志（供绘图脚本直接读取）。
        3. 从奖励函数实例提取各分量奖励，注入日志行（对齐工作文档 3.5.8）。
        4. 终端只打印风格统一的关键指标单行摘要。

    多进程约定：
        文件创建、日志写入、终端打印全部只在主进程执行；
        构造函数不做任何 I/O，避免非主进程产生空文件或重复打印。

    Args:
        output_dir: 日志目录（不存在时由主进程自动创建）。
        reward_fn: 奖励函数实例，可选。传入后 ``on_log`` 会读取其
                   ``rewards`` 缓存并把各分量均值注入日志行。
    """

    # 终端摘要中各指标的显示名与格式化方式（保证风格统一）
    _METRIC_FORMATS = (
        ("loss", "loss", "{:.3e}"),
        ("learning_rate", "lr", "{:.3e}"),
        ("reward", "reward", "{:.4f}"),
        ("reward_std", "r_std", "{:.4f}"),
        ("kl", "kl", "{:.3e}"),
        ("entropy", "ent", "{:.4f}"),
        ("grad_norm", "gnorm", "{:.3e}"),
    )

    def __init__(self, output_dir: str = "output/logs", reward_fn=None):
        self.output_dir = output_dir
        self.reward_fn = reward_fn
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_timestamped = os.path.join(
            output_dir, f"train_log_{self.timestamp}.jsonl"
        )
        self.file_unified = os.path.join(output_dir, "train_log.jsonl")

        # 文件句柄延迟到 on_train_begin 再打开（此时才能判定主进程）
        self.f_timestamped = None
        self.f_unified = None
        self._opened = False

    # ------------------------------------------------------------------
    # 主进程判定
    # ------------------------------------------------------------------

    @staticmethod
    def _is_main(args=None, state=None) -> bool:
        """
        判断当前是否为主进程。

        判定顺序：优先用 ``args.is_main_process``（Trainer 的权威标志），
        其次用 ``state.local_rank``，两者都不可得时才按单进程处理。
        非主进程必须显式返回 False，避免多卡重复写日志。

        Args:
            args: TrainingArguments 实例。
            state: TrainerState 实例。

        Returns:
            当前进程是否为主进程。
        """
        if args is not None and hasattr(args, "is_main_process"):
            return bool(args.is_main_process)
        if state is not None and hasattr(state, "local_rank"):
            return state.local_rank in (0, -1)
        # 既无 args 也无 state：按单进程场景处理
        return True

    def _ensure_files(self) -> None:
        """主进程首次写日志前创建目录并打开两个文件句柄（幂等）。"""
        if self._opened:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        # 时间戳文件：本次训练专属，追加写入
        self.f_timestamped = open(self.file_timestamped, "a", encoding="utf-8")
        # 统一接口文件：清空后重写，始终只含最新一次训练
        self.f_unified = open(self.file_unified, "w", encoding="utf-8")
        self._opened = True

        print("=" * 60)
        print("训练日志回调已启用")
        print(f"  时间戳文件  : {self.file_timestamped}")
        print(f"  统一接口文件: {self.file_unified}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 奖励分量提取（工作文档 3.5.8）
    # ------------------------------------------------------------------

    def _collect_reward_breakdown(self) -> dict:
        """
        从奖励函数实例读取最近一个 batch 的奖励分量均值。

        Returns:
            形如 ``{"reward/length": 0.5, "reward/match": 0.5,
            "reward/total": 0.5}`` 的字典；未配置奖励函数或无缓存时返回空字典。
        """
        if self.reward_fn is None:
            return {}
        cache = getattr(self.reward_fn, "rewards", None)
        if not cache:
            return {}
        breakdown = {}
        for key, alias in (
            ("length_rewards", "reward/length"),
            ("match_rewards", "reward/match"),
            ("total_rewards", "reward/total"),
        ):
            values = cache.get(key)
            if values is None:
                continue
            try:
                breakdown[alias] = float(torch.as_tensor(values).mean().item())
            except (TypeError, ValueError, RuntimeError):
                continue
        return breakdown

    # ------------------------------------------------------------------
    # Trainer 钩子
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始：主进程打开日志文件。"""
        if self._is_main(args, state):
            self._ensure_files()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        每步日志：写入双文件并打印统一格式的单行摘要。

        Args:
            args: TrainingArguments。
            state: TrainerState。
            control: TrainerControl。
            logs: Trainer 提供的原始指标字典。
        """
        if not logs or not self._is_main(args, state):
            return control

        entry = dict(logs)
        entry["global_step"] = state.global_step

        # 注入奖励函数各分量（工作文档 3.5.8）
        entry.update(self._collect_reward_breakdown())

        self._ensure_files()
        log_line = json.dumps(entry, ensure_ascii=False)
        for handle in (self.f_timestamped, self.f_unified):
            handle.write(log_line + "\n")
            handle.flush()

        # 终端摘要：仅训练步（含 loss）打印，评估结果单独标识，格式统一
        if "loss" in logs:
            parts = [f"step={state.global_step:>4}"]
            for src, label, fmt in self._METRIC_FORMATS:
                if src in logs and isinstance(logs[src], (int, float)):
                    parts.append(f"{label}=" + fmt.format(logs[src]))
            print("  " + " | ".join(parts))
        elif "eval_loss" in logs or "eval_reward" in logs:
            print(
                f"  [eval] step={state.global_step:>4} | "
                f"loss={logs.get('eval_loss', float('nan')):.3e} | "
                f"reward={logs.get('eval_reward', float('nan')):.4f} | "
                f"kl={logs.get('eval_kl', float('nan')):.3e} | "
                f"ent={logs.get('eval_entropy', float('nan')):.4f}"
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束：主进程关闭文件并打印产物路径。"""
        if not self._is_main(args, state):
            return control
        for handle in (self.f_timestamped, self.f_unified):
            if handle is not None and not handle.closed:
                handle.close()
        print("=" * 60)
        print(f"训练完成！总步数: {state.global_step}")
        print("日志已保存:")
        print(f"  {self.file_timestamped}")
        print(f"  {self.file_unified}")
        print("=" * 60)
        return control

    def __del__(self):
        """析构兜底：训练中途异常退出时确保文件句柄关闭。"""
        for attr in ("f_timestamped", "f_unified"):
            handle = getattr(self, attr, None)
            if handle is not None and not handle.closed:
                try:
                    handle.close()
                except OSError:
                    pass


# ================== 数据集加载 ==================


def load_dataset(data_path: str) -> tuple:
    """
    加载指定 JSONL 数据集并验证格式。

    读取 data_path 中的 JSONL 文件，验证每条记录的格式，
    提取 prompts、answers、references 并返回。

    支持两种数据格式：
        1. 完整格式: {"prompt": "...", "answer": "...", "reference": "..."}
           - answer: 完整解题过程（标注数据）
           - reference: 关键答案（用于奖励计算比对）
        2. 精简格式: {"prompt": "...", "reference": "..."}
           - 无 answer，只有 prompt 和 reference
           - 适用于只需要参考答案、不需要完整过程标注的场景

    Args:
        data_path: JSONL 数据文件路径。

    Returns:
        元组 (prompts, answers, references)，各为字符串列表。
              answers 在精简格式下为空字符串占位。

    Raises:
        FileNotFoundError: 数据文件不存在。
        ValueError: 数据格式错误（缺少必需字段或内容不完整）。
    """
    # 使用 datasets 库直接读取 JSONL（无需 pandas，与项目技术栈统一）
    dataset = hf_load_dataset("json", data_files=data_path, split="train")

    # 支持两种格式：prompt + answer (+ reference)，或 prompt + reference
    if "prompt" not in dataset.column_names:
        raise ValueError("数据格式错误：缺少必需字段 'prompt'")

    has_answer = "answer" in dataset.column_names
    has_reference = "reference" in dataset.column_names

    prompts = dataset["prompt"]
    if has_answer and has_reference:
        # 完整格式: prompt + answer + reference
        answers = dataset["answer"]
        references = dataset["reference"]
    elif not has_answer and has_reference:
        # 精简格式: prompt + reference（无 answer）
        answers = [""] * len(dataset)
        references = dataset["reference"]
    else:
        raise ValueError(
            "数据格式错误：至少需要 'reference' 字段\n"
            "完整格式: {prompt, answer, reference}\n"
            "精简格式: {prompt, reference}"
        )

    # 校验非空
    valid_mask = [p and isinstance(p, str) for p in prompts]
    if not all(valid_mask):
        invalid_idx = [i for i, v in enumerate(valid_mask) if not v]
        raise ValueError(f"数据格式错误：索引 {invalid_idx} 的 prompt 不完整")

    return prompts, answers, references


# ================== 数据集验证 ==================


def validate_dataset_format(dataset) -> dict:
    """
    验证数据集格式（接受 datasets.Dataset 或类列表对象）。

    支持两种格式：
        1. 完整格式: {prompt, answer, reference}
        2. 精简格式: {prompt, reference}

    Args:
        dataset: datasets.Dataset 对象或类列表字典。

    Returns:
        验证结果字典，包含 passed、issues、missing_fields、summary 等字段。
    """
    result = {
        "passed": True,
        "issues": [],
        "missing_fields": [],
        "summary": {},
    }

    # 兼容 datasets.Dataset 和普通 dict 两种类型
    columns = getattr(dataset, "column_names", list(dataset.keys()) if isinstance(dataset, dict) else [])
    total = len(dataset)

    if "prompt" not in columns:
        result["missing_fields"].append("prompt")
        result["issues"].append("缺少必需字段 'prompt'")
        result["passed"] = False
        return result

    has_answer = "answer" in columns
    has_reference = "reference" in columns

    if not has_reference:
        result["missing_fields"].append("reference")
        result["issues"].append("缺少必需字段 'reference'")
        result["passed"] = False
        return result

    # 逐行校验
    valid_count = 0
    for i in range(total):
        row = dataset[i]
        prompt_ok = bool(row.get("prompt")) and isinstance(row["prompt"], str)
        ref_ok = bool(row.get("reference")) and isinstance(row["reference"], str)
        if has_answer:
            answer_ok = bool(row.get("answer")) and isinstance(row["answer"], str)
            if prompt_ok and answer_ok and ref_ok:
                valid_count += 1
        else:
            if prompt_ok and ref_ok:
                valid_count += 1

    fmt_label = "完整格式 (prompt + answer + reference)" if has_answer else "精简格式 (prompt + reference)"
    result["summary"] = {
        "total_rows": total,
        "valid_rows": valid_count,
        "invalid_rows": total - valid_count,
        "format": fmt_label,
    }

    if valid_count < total:
        result["passed"] = False
        result["issues"].append(f"有效行 {valid_count}/{total}")

    return result


# ================== LoRA 配置 ==================


def setup_lora(
    model_name_or_path: str,
    lora_config: dict = None,
    tokenizer=None,
    **model_kwargs,
) -> tuple:
    """
    加载模型并配置 LoRA（PEFT）。

    从指定路径加载基座模型并应用 LoRA 适配。分词器可由调用方传入以复用，
    未传入时才在本函数内加载（避免重复加载）。

    Args:
        model_name_or_path: 模型名称（HF Hub）或本地路径。
        lora_config: LoRA 配置字典，键包括：
                     - target_modules: 目标模块列表，默认 ["q_proj", "v_proj"]
                     - r: LoRA 秩，默认 8
                     - lora_alpha: LoRA 缩放系数，默认 32
                     - lora_dropout: LoRA dropout，默认 0.1
                     其余参数透传 LoraConfig。
        tokenizer: 已加载的分词器，传入则直接复用；为 None 时内部加载。
        **model_kwargs: 仅传给 AutoModelForCausalLM.from_pretrained 的参数。
                     注意其中不得包含分词器不识别的键。

    Returns:
        元组 (tokenizer, model)，model 为 PEFT 包装后的对象。

    Raises:
        RuntimeError: 模型加载或 LoRA 配置失败。
    """
    lora_config = lora_config or {}
    default_lora = {
        "target_modules": ["q_proj", "v_proj"],
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
    default_lora.update(lora_config)

    try:
        if tokenizer is None:
            # 分词器只接受自身的参数，不能接收 torch_dtype / 量化开关等模型参数
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )

        lora_cfg = LoraConfig(**default_lora)
        model = get_peft_model(base_model, lora_cfg)
        model.print_trainable_parameters()

        return tokenizer, model
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"LoRA 配置失败: {e}") from e


# ================== 模型保存 ==================


def save_peft_model(
    model,
    tokenizer,
    save_directory: str,
    safe_serialization: bool = True,
) -> None:
    """
    保存 LoRA adapter 权重（仅保存适配器，不保存完整模型）。

    使用 PEFT 的标准接口保存，输出为 peft 格式。

    Args:
        model: PEFT 包装后的模型。
        tokenizer: 分词器。
        save_directory: 保存目录路径。
        safe_serialization: 是否使用 safetensors 格式。
    """
    model.save_pretrained(save_directory, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(save_directory)


# ================== 奖励函数实例化 ==================


def make_reward_function(
    reward_config: dict = None,
) -> GRPORewardFunction:
    """
    根据配置字典实例化 GRPORewardFunction。

    Args:
        reward_config: 奖励函数配置字典，键包括：
                       - weights: 各分量权重，如 {"length": 0.5, "match": 0.5}
                       - answer_tag: 答案提取标签
                       - forbidden_strings: 违禁字符串列表
                       缺省使用默认配置。

    Returns:
        GRPORewardFunction 实例。
    """
    reward_config = reward_config or {}
    return GRPORewardFunction(
        weights=reward_config.get("weights", {"length": 0.5, "match": 0.5}),
        answer_tag=reward_config.get("answer_tag", ""),
        forbidden_strings=reward_config.get("forbidden_strings", []),
    )


# ================== 模型加载/保存（完整） ==================


def load_hf_model(model_name_or_path: str, **kwargs) -> tuple:
    """
    从 Hugging Face Hub 或本地加载完整模型（含 tokenizer）。

    Args:
        model_name_or_path: 模型名称或本地路径。
        **kwargs: 传给 from_pretrained 的参数。

    Returns:
        元组 (model, tokenizer)。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model, tokenizer


def save_hf_model(model, tokenizer, save_directory: str) -> None:
    """
    保存完整模型（用于合并 LoRA 后的输出）。

    Args:
        model: 要保存的模型。
        tokenizer: 分词器。
        save_directory: 保存目录路径。
    """
    model.save_pretrained(save_directory, safe_serialization=True)
    tokenizer.save_pretrained(save_directory)


def merge_and_unload(model) -> "AutoModelForCausalLM":
    """
    将 LoRA adapter 合并到基础模型并卸载适配器。

    Args:
        model: PEFT 包装的模型。

    Returns:
        合并后的普通 transformers 模型。
    """
    merged = model.merge_and_unload()
    return merged


# ================== 评估配置加载 ==================


def load_eval_config(
    all_config: dict,
    test_data_path: str,
    model_path: str,
    checkpoint_path: str = None,
) -> dict:
    """
    加载并合并评估所需配置。

    Args:
        all_config: ConfigLoader.load_all() 返回的全量配置。
        test_data_path: 测试数据 JSONL 路径。
        model_path: 基座模型路径。
        checkpoint_path: LoRA adapter 路径（训练中途检查点）。
                        如果为 None，则使用 model_path 作为完整模型。

    Returns:
        评估配置字典，包含 model_kwargs、eval_params、test_data_path。
    """
    eval_cfg = all_config.get("EvalConfig", {})
    dataset_cfg = all_config.get("DataSetConfig", {})
    model_cfg = all_config.get("ModelConfig", {})

    # 从 DataSetConfig 中提取 test 路径（兼容平铺键名和嵌套字典两种风格）
    test_path = test_data_path
    if not test_path:
        test_path = dataset_cfg.get("test_data_path", "")
    if not test_path:
        if isinstance(dataset_cfg.get("test"), dict):
            test_path = dataset_cfg["test"].get("path", test_data_path)

    # 构建模型加载参数
    # 精度与量化开关统一从 ModelConfig 读取（与 train_grpo.py 保持一致），
    # transformers 5.x 使用 dtype，torch_dtype 已弃用
    model_kwargs = {}
    if model_cfg.get("load_in_8bit", False):
        model_kwargs["load_in_8bit"] = True
    if model_cfg.get("load_in_4bit", False):
        model_kwargs["load_in_4bit"] = True
    model_kwargs["device_map"] = eval_cfg.get("device_map", model_cfg.get("device", "auto"))
    model_kwargs["dtype"] = resolve_torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))

    eval_params = {
        "batch_size": eval_cfg.get("batch_size", 8),
        "max_new_tokens": eval_cfg.get("max_new_tokens", 512),
        "do_sample": eval_cfg.get("do_sample", False),
        "temperature": eval_cfg.get("temperature", 0.7),
        "top_p": eval_cfg.get("top_p", 0.9),
        "generate_repeat": eval_cfg.get("generate_repeat", 1),
        "max_length": eval_cfg.get("max_length", 4096),
    }

    return {
        "model_kwargs": model_kwargs,
        "eval_params": eval_params,
        "test_data_path": test_path,
        "model_path": model_path,
        "checkpoint_path": checkpoint_path,
    }
