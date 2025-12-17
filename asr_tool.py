#!/usr/bin/env python3
"""ASR工具 - 基于FunAudioLLM/Fun-ASR模型进行语音识别"""

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_VAD_MAX_TIME = 30000


def get_device() -> str:
    """自动检测最佳计算设备"""
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def build_model(model_name: str, device: str, use_vad: bool, vad_max_time: int):
    """构建ASR模型"""
    from funasr import AutoModel
    from funasr.models.fun_asr_nano.model import FunASRNano  # noqa: F401 注册模型类

    ensure_qwen_model(model_name)

    kwargs = {"model": model_name, "device": device, "disable_update": True}
    if use_vad:
        kwargs["vad_model"] = "fsmn-vad"
        kwargs["vad_kwargs"] = {"max_single_segment_time": vad_max_time}

    print(f"加载模型: {model_name} (设备: {device})")
    return AutoModel(**kwargs)


def transcribe(model, audio_path: str, **kwargs) -> dict:
    """识别单个音频"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"文件不存在: {audio_path}")

    res = model.generate(input=[audio_path], cache={}, **kwargs)[0]
    text = res.get("text", "")
    segments = []

    if res.get("sentence_info"):
        segments = [{"text": s["text"], "start": s["start"], "end": s["end"]}
                    for s in res["sentence_info"]]
    elif res.get("timestamp"):
        ts = res["timestamp"]
        segments = [{"text": text, "start": ts[0][0], "end": ts[-1][1]}]

    return {"text": text, "segments": segments}


def save_result(result: dict, path: str):
    """保存结果到文件"""
    with open(path, "w", encoding="utf-8") as f:
        if result["segments"]:
            for seg in result["segments"]:
                f.write(f"[{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}] {seg['text']}\n")
        else:
            # 按标点分句
            for sent in re.split(r'(?<=[。！？!?])', result["text"]):
                if sent.strip():
                    f.write(f"{sent.strip()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ASR工具 - 语音识别",
        epilog="示例: uv run python asr_tool.py audio.mp3 -o ./output")
    parser.add_argument("files", nargs="+", help="音频文件")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("-l", "--language", default="auto", choices=["auto", "zh", "en", "ja"])
    parser.add_argument("-o", "--output-dir", default="./output")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("--hotwords", nargs="*")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--vad-max-time", type=int, default=DEFAULT_VAD_MAX_TIME)
    parser.add_argument("--no-itn", action="store_true")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else args.device
    print(f"设备: {device}")

    model = build_model(args.model, device, not args.no_vad, args.vad_max_time)
    gen_kwargs = {"batch_size": args.batch_size, "language": args.language, "itn": not args.no_itn}
    if args.hotwords:
        gen_kwargs["hotwords"] = args.hotwords

    success, failed = 0, 0
    for audio in args.files:
        try:
            result = transcribe(model, audio, **gen_kwargs)
            print(f"\n[{audio}] {result['text']}")

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                out_path = f"{args.output_dir}/{Path(audio).stem}_{datetime.now():%Y%m%d_%H%M%S}.txt"
                save_result(result, out_path)
                print(f"保存: {out_path}")
            success += 1
        except Exception as e:
            print(f"失败 [{audio}]: {e}")
            failed += 1

    print(f"\n{'='*40}\n完成 - 成功: {success}, 失败: {failed}")


if __name__ == "__main__":
    main()
