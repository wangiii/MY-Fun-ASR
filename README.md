# vv-asr

基于 [FunASR](https://github.com/modelscope/FunASR) 框架的语音识别命令行工具。

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

### 为什么分离 VAD 和 ASR？

FunASR 的 `AutoModel` 支持内置 VAD，但合并后会**丢失时间戳信息**。

本工具分离加载两个模型：
1. **VAD 模型** (`fsmn-vad`)：检测语音片段，获取时间区间
2. **ASR 模型** (`SenseVoiceSmall`)：对每个片段进行识别

这样可以保留每句话的起止时间，输出带时间戳的字幕。

### 项目结构

```text
vv-asr/
├── asr_tool.py      # 主程序 (模型加载、转录逻辑、CLI)
├── pyproject.toml   # 项目配置和依赖
├── uv.lock          # 依赖锁定文件
├── output/          # 默认输出目录
└── data/            # 音频文件目录
    ├── todo/        # 待处理
    └── done/        # 已处理
```

### 核心函数

| 函数 | 说明 |
|------|------|
| `build_model()` | 加载 VAD 和 ASR 模型，配置设备和精度 |
| `transcribe()` | 核心转录逻辑：VAD → 重采样 → ASR → 清理 |
| `clean_text()` | 清理 SenseVoice 输出的标签 (`<\|zh\|>` 等) |
| `resample_audio()` | 重采样到 16kHz (模型要求) |
| `save_result()` | 保存带时间戳的字幕文件 |

## 支持的模型

| 模型 | 说明 | 适用场景 |
|------|------|----------|
| `iic/SenseVoiceSmall` | 阿里达摩院，速度快，多语言 | 通用场景 (推荐) |
| `FunAudioLLM/Fun-ASR-Nano-2512` | 更小的模型，需要 Qwen3-0.6B | 资源受限环境 |

## 开发指南

### 本地开发

```bash
# 克隆项目
git clone https://github.com/wangiii/vv-asr.git
cd vv-asr

# 安装依赖
uv sync

# 运行测试
uv run python asr_tool.py data/todo/test.wav
```

### 代码风格

项目使用 Python 类型注解，代码中包含详细的中文注释。

### 添加新模型

1. 在 `build_model()` 中添加模型特殊处理逻辑
2. 如需特殊的输出清理，修改 `clean_text()`
3. 更新 README 的模型列表

## 常见问题

### Q: 识别结果只有一个词？

**原因**：音频采样率不是 16kHz，直接传入 numpy 数组时模型无法正确处理。

**解决**：代码已自动重采样到 16kHz，如仍有问题请检查音频文件是否损坏。

### Q: MPS 设备上 torch.compile 报错？

**原因**：Apple MPS 不支持 torch.compile。

**解决**：代码已自动跳过 MPS 设备的 compile，无需处理。

### Q: 模型下载很慢？

**解决**：模型从 ModelScope 下载，可设置镜像：

```bash
export MODELSCOPE_CACHE=~/.cache/modelscope
```

## 依赖

- [FunASR](https://github.com/modelscope/FunASR) - 语音识别框架
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [librosa](https://librosa.org/) - 音频处理
- [soundfile](https://pysoundfile.readthedocs.io/) - 音频读写

## License

MIT
