#!/usr/bin/env python3
"""ASR工具 - 基于FunAudioLLM/Fun-ASR模型进行语音识别"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

# 默认配置
DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
DEFAULT_VAD_MAX_TIME = 30000


def get_best_device() -> str:
    """自动检测最佳计算设备"""
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_timestamp(ms: int) -> str:
    """将毫秒转换为 HH:MM:SS.mmm 格式"""
    seconds = ms / 1000
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def split_sentences(text: str) -> list[str]:
    """按标点符号分割文本为句子列表"""
    parts = re.split(r'([。！？!?])', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i + 1]
        if sentence.strip():
            sentences.append(sentence.strip())
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences


def build_model(model_name: str, device: str, use_vad: bool, vad_max_time: int):
    """构建并返回ASR模型"""
    from funasr import AutoModel

    # 确定 remote_code 路径
    if os.path.isdir(model_name):
        remote_code = os.path.join(model_name, "model.py")
    else:
        cache_dir = os.path.expanduser(f"~/.cache/modelscope/hub/models/{model_name}")
        remote_code = os.path.join(cache_dir, "model.py") if os.path.exists(cache_dir) else "./model.py"

    model_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "remote_code": remote_code,
        "device": device,
        "disable_update": True,
    }

    if use_vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": vad_max_time}

    print(f"正在加载模型: {model_name}")
    print(f"使用设备: {device}")
    return AutoModel(**model_kwargs)


def transcribe_audio(model, audio_path: str, language: str = "auto",
                     hotwords: list = None, itn: bool = True, batch_size: int = 1) -> dict:
    """对单个音频文件进行识别"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    generate_kwargs = {
        "input": [audio_path],
        "cache": {},
        "batch_size": batch_size,
        "language": language,
        "itn": itn,
    }
    if hotwords:
        generate_kwargs["hotwords"] = hotwords

    print(f"正在识别音频: {audio_path}")
    result = model.generate(**generate_kwargs)
    res = result[0]
    text = res.get("text", "")

    output = {"text": text, "segments": []}

    # 提取时间戳信息
    if "sentence_info" in res and res["sentence_info"]:
        output["segments"] = [
            {"text": s.get("text", ""), "start": s.get("start", 0), "end": s.get("end", 0)}
            for s in res["sentence_info"]
        ]
    elif "timestamp" in res and res["timestamp"]:
        ts = res["timestamp"]
        output["segments"] = [{"text": text, "start": ts[0][0], "end": ts[-1][1]}]

    return output


def save_result(result: dict, output_path: str):
    """保存识别结果到文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        if result.get("segments"):
            for seg in result["segments"]:
                start_ts = format_timestamp(seg["start"])
                end_ts = format_timestamp(seg["end"])
                f.write(f"[{start_ts} --> {end_ts}] {seg['text']}\n")
        else:
            for sentence in split_sentences(result["text"]):
                f.write(f"{sentence}\n")


def transcribe_batch(audio_paths: list, output_dir: str = None, **kwargs) -> dict:
    """批量识别多个音频文件"""
    # 只构建一次模型
    model = build_model(
        kwargs.pop("model_name", DEFAULT_MODEL),
        kwargs.pop("device", "cpu"),
        kwargs.pop("use_vad", True),
        kwargs.pop("vad_max_time", DEFAULT_VAD_MAX_TIME),
    )

    results = {}
    for audio_path in audio_paths:
        try:
            result = transcribe_audio(model, audio_path, **kwargs)
            results[audio_path] = result
            print(f"\n[{audio_path}]")
            print(f"识别结果: {result['text']}")

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = Path(audio_path).stem
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"{base_name}_{ts}.txt")
                save_result(result, output_path)
                print(f"结果已保存到: {output_path}")

        except Exception as e:
            print(f"识别失败 [{audio_path}]: {e}")
            results[audio_path] = None

    return results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ASR工具 - 基于FunAudioLLM/Fun-ASR模型进行语音识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python asr_tool.py audio.mp3                    # 识别单个文件
  python asr_tool.py *.mp3 -o ./results           # 批量识别并保存
  python asr_tool.py audio.mp3 -l zh              # 指定中文
  python asr_tool.py audio.mp3 --hotwords 人工智能  # 添加热词
        """)

    parser.add_argument("audio_files", nargs="+", help="音频文件路径")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"模型名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("-d", "--device", default="auto", help="计算设备 (默认: auto)")
    parser.add_argument("-l", "--language", default="auto", choices=["auto", "zh", "en", "ja"], help="语言")
    parser.add_argument("-o", "--output-dir", default="./output", help="输出目录 (默认: ./output)")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument("--hotwords", nargs="*", help="热词列表")
    parser.add_argument("--no-vad", action="store_true", help="禁用VAD")
    parser.add_argument("--vad-max-time", type=int, default=DEFAULT_VAD_MAX_TIME, help="VAD最大分段时间(ms)")
    parser.add_argument("--no-itn", action="store_true", help="禁用逆文本规范化")

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device != "auto" else get_best_device()
    print(f"使用计算设备: {device}")

    results = transcribe_batch(
        audio_paths=args.audio_files,
        output_dir=args.output_dir,
        model_name=args.model,
        device=device,
        language=args.language,
        hotwords=args.hotwords,
        use_vad=not args.no_vad,
        vad_max_time=args.vad_max_time,
        itn=not args.no_itn,
        batch_size=args.batch_size,
    )

    # 打印汇总
    success = sum(1 for v in results.values() if v is not None)
    failed = len(results) - success
    print(f"\n{'=' * 50}")
    print(f"识别完成 - 成功: {success}, 失败: {failed}")


if __name__ == "__main__":
    main()
