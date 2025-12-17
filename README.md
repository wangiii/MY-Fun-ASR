# vv-asr

语音识别命令行工具。

## 特性

- **多模型支持**：SenseVoiceSmall、Fun-ASR-Nano 等
- **自动设备检测**：CUDA / MPS (Apple Silicon) / CPU
- **半精度优化**：自动选择 bfloat16 / float16 加速推理
- **VAD 分段**：语音活动检测，输出带时间戳的字幕
- **热词增强**：提高特定词汇的识别准确度
- **批量处理**：支持同时处理多个音频文件

## 快速开始

### 环境要求

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) 包管理器

### 安装

```bash
git clone https://github.com/wangiii/vv-asr.git
cd vv-asr
uv sync
```

### 基本使用

```bash
# 识别单个音频
uv run python asr_tool.py audio.mp3

# 批量识别
uv run python asr_tool.py *.mp3 -o ./results

# 指定语言
uv run python asr_tool.py audio.mp3 -l zh

# 添加热词
uv run python asr_tool.py audio.mp3 --hotwords 人工智能 机器学习
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | ASR 模型名称 | `iic/SenseVoiceSmall` |
| `-d, --device` | 计算设备 (`auto`/`cuda`/`mps`/`cpu`) | `auto` |
| `-l, --language` | 识别语言 (`auto`/`zh`/`en`/`ja`) | `auto` |
| `-o, --output-dir` | 输出目录 | `./output` |
| `-b, --batch-size` | 批处理大小 | `1` |
| `--hotwords` | 热词列表 | - |
| `--no-vad` | 禁用 VAD 分段 | - |
| `--vad-max-time` | VAD 最大分段时长 (ms) | `30000` |
| `--no-itn` | 禁用逆文本规范化 | - |
| `--no-compile` | 禁用 torch.compile | - |
| `--no-half` | 禁用半精度 | - |
| `--threads` | CPU 线程数 | 全部核心 |

## 输出格式

转录结果保存为带时间戳的文本文件：

```text
[00:00:00.630 --> 00:00:05.090] 第一句话
[00:00:05.370 --> 00:00:10.200] 第二句话
```

## 架构设计

### 核心流程

```text
音频文件 → VAD 分段 → 重采样 (16kHz) → ASR 识别 → 清理标签 → 输出字幕
```

## 支持的模型

| 模型 | 说明 | 适用场景 |
|------|------|----------|
| `iic/SenseVoiceSmall` | 阿里达摩院，速度快，多语言 | 通用场景 (推荐) |
| `FunAudioLLM/Fun-ASR-Nano-2512` | 更小的模型，需要 Qwen3-0.6B | 资源受限环境 |

## License

MIT
