#!/usr/bin/env python3
"""vv-asr - 语音识别工具"""

# 在导入其他库前设置日志和警告过滤
import logging
import os
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*[Aa]ttention mask.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import argparse
import re
import shutil
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import torch

# DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_MODEL = "iic/SenseVoiceSmall"
DEFAULT_VAD_MAX_TIME = 30000
DEFAULT_BATCH_SIZE = 1

# 预编译正则表达式
_LANG_TAG_RE = re.compile(r"<\|(?:zh|en|ja|yue|ko)\|>")
_OTHER_TAG_RE = re.compile(r"<\|[^|]+\|>")


@lru_cache(maxsize=1)
def get_device() -> str:
    """自动检测最佳计算设备"""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """根据设备选择最佳数据类型"""
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def format_timestamp(ms: int) -> str:
    """毫秒转 HH:MM:SS.mmm"""
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ensure_qwen_model(model_name: str):
    """确保 Qwen3-0.6B 模型存在"""
    cache_dir = Path.home() / f".cache/modelscope/hub/models/{model_name}"
    qwen_dir = cache_dir / "Qwen3-0.6B"
    if qwen_dir.is_dir() and not (qwen_dir / "model.safetensors").exists():
        print("正在下载 Qwen3-0.6B 模型...")
        from huggingface_hub import snapshot_download
        hf_path = Path(snapshot_download(repo_id="Qwen/Qwen3-0.6B"))
        for f in hf_path.iterdir():
            if f.is_file() and not (qwen_dir / f.name).exists():
                shutil.copy2(f, qwen_dir / f.name)


def build_model(model_name: str, device: str, use_vad: bool, vad_max_time: int,
                use_compile: bool = True, use_half: bool = True):
    """构建ASR模型"""
    from funasr import AutoModel

    # Fun-ASR-Nano 需要额外的 Qwen 模型
    if "Fun-ASR-Nano" in model_name:
        from funasr.models.fun_asr_nano.model import FunASRNano  # noqa: F401 注册模型类
        ensure_qwen_model(model_name)

    kwargs = {"model": model_name, "device": device, "disable_update": True}
    if use_vad:
        kwargs["vad_model"] = "fsmn-vad"
        kwargs["vad_kwargs"] = {"max_single_segment_time": vad_max_time}

    dtype = get_dtype(device) if use_half else torch.float32
    dtype_name = str(dtype).split('.')[-1]
    print(f"加载模型: {model_name} (设备: {device}, 精度: {dtype_name})")

    model = AutoModel(**kwargs)

    # 转换为半精度
    if use_half and dtype != torch.float32:
        try:
            model.model = model.model.to(dtype)
            print(f"已转换为 {dtype_name} 精度")
        except Exception as e:
            print(f"半精度转换失败: {e}")

    # torch.compile 优化 (PyTorch 2.0+) - 默认禁用，首次编译太慢
    if use_compile and hasattr(torch, "compile") and device != "mps":
        try:
            model.model = torch.compile(model.model, mode="default")
            print("已启用 torch.compile 优化 (首次运行需要编译)")
        except Exception as e:
            print(f"torch.compile 失败: {e}")

    return model


def clean_text(text: str) -> list[str]:
    """清理 SenseVoice 输出，按识别的句子分行"""
    return [
        clean for part in _LANG_TAG_RE.split(text)
        if (clean := _OTHER_TAG_RE.sub("", part).strip())
    ]


def transcribe(model, audio_path: str, **kwargs) -> dict:
    """识别单个音频"""
    audio = Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"文件不存在: {audio_path}")

    with torch.inference_mode():
        res = model.generate(input=[audio_path], cache={}, **kwargs)[0]

    sentences = clean_text(res.get("text", ""))

    # 构建时间戳段落
    segments = []
    if sentence_info := res.get("sentence_info"):
        segments = [
            {"text": sent, "start": s["start"], "end": s["end"]}
            for s in sentence_info
            for sent in clean_text(s["text"])
        ]
    elif ts := res.get("timestamp"):
        segments = [{"text": sent, "start": ts[0][0], "end": ts[-1][1]} for sent in sentences]

    return {"text": "\n".join(sentences), "segments": segments}


def save_result(result: dict, path: Path):
    """保存结果到文件"""
    with path.open("w", encoding="utf-8") as f:
        if result["segments"]:
            lines = (
                f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}] {s['text']}"
                for s in result["segments"]
            )
            f.write("\n".join(lines) + "\n")
        else:
            f.write(result["text"])


def main():
    parser = argparse.ArgumentParser(
        description="ASR工具 - 语音识别",
        epilog="示例: uv run python asr_tool.py audio.mp3 -o ./output")
    parser.add_argument("files", nargs="+", type=Path, help="音频文件")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("-l", "--language", default="auto", choices=["auto", "zh", "en", "ja"])
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hotwords", nargs="*")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--vad-max-time", type=int, default=DEFAULT_VAD_MAX_TIME)
    parser.add_argument("--no-itn", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile 优化")
    parser.add_argument("--no-half", action="store_true", help="禁用半精度 (float16/bfloat16)")
    parser.add_argument("--threads", type=int, help="CPU 线程数 (默认: 全部核心)")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else args.device

    num_threads = args.threads or os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    print(f"设备: {device}, CPU线程: {num_threads}")

    model = build_model(args.model, device, not args.no_vad, args.vad_max_time,
                        use_compile=not args.no_compile, use_half=not args.no_half)
    gen_kwargs = {"batch_size": args.batch_size, "language": args.language, "itn": not args.no_itn}
    if args.hotwords:
        gen_kwargs["hotwords"] = args.hotwords

    success, failed = 0, 0
    for audio in args.files:
        try:
            result = transcribe(model, str(audio), **gen_kwargs)
            print(f"\n[{audio}] {result['text']}")

            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                out_path = args.output_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{audio.stem}.txt"
                save_result(result, out_path)
                print(f"保存: {out_path}")
            success += 1
        except Exception as e:
            print(f"失败 [{audio}]: {e}")
            failed += 1

    print(f"\n{'='*40}\n完成 - 成功: {success}, 失败: {failed}")


if __name__ == "__main__":
    main()
