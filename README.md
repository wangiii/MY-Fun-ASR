# MY-Fun-ASR

基于 FunAudioLLM/Fun-ASR 模型的语音识别工具。

## 安装

```bash
uv sync
```

## 使用

```bash
uv run python asr_tool.py audio.mp3                     # 识别音频
uv run python asr_tool.py *.mp3 -o ./results            # 批量识别
uv run python asr_tool.py audio.mp3 -l zh               # 指定语言
uv run python asr_tool.py audio.mp3 --hotwords 人工智能   # 添加热词
```

## 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m` | 模型名称 | FunAudioLLM/Fun-ASR-Nano-2512 |
| `-d` | 计算设备 (auto/cuda/mps/cpu) | auto |
| `-l` | 语言 (auto/zh/en/ja) | auto |
| `-o` | 输出目录 | ./output |
| `--hotwords` | 热词列表 | - |
| `--no-vad` | 禁用VAD | - |
| `--no-itn` | 禁用逆文本规范化 | - |
