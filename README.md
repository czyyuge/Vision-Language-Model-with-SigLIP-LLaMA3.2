# 视频文字索引多模态模型

输入一段视频，按每秒切分帧画面，通过训练好的 SigLIP + LLaMA 3.2 视觉语言模型（VLM）为每一帧生成自然语言描述。

支持的场景：视频内容摘要、监控关键帧提取、辅助标注等。

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers peft pillow opencv-python
```

### 2. 下载模型

项目需要两个模型组件，均缺一不可：

| 组件 | 说明 | 大小 | 下载方式 |
|---|---|---|---|
| LLaMA 3.2 (3B) 基座 | Meta 的开源大语言模型 | ~7 GB | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B) 申请授权后下载 |
| 训练好的 VLM checkpoint | 原项目训练好的权重文件 | ~1 GB | [HuggingFace](https://huggingface.co/RadiumLR/Siglip-LLama3.2-VLM-Model) 直接下载 |

下载后放入对应目录：

```
Llama-3.2-3B/    ← 放 LLaMA 基座模型的所有文件
checkpoints/     ← 放下载的 .pth 权重文件
```

> **注意：** 磁盘需预留约 **10 GB**，推理时显卡需 **8 GB 以上显存**。

### 3. 运行

```bash
python run.py 你的视频.mp4 --llama ./Llama-3.2-3B --checkpoint ./checkpoints/xxx.pth
```

其中的 `xxx.pth` 替换为实际下载的 checkpoint 文件名。

---

## run.py 参数详解

```
python run.py <video> --llama <dir> --checkpoint <file> [可选参数...]
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `video` | **是** | — | 输入视频路径，第一个位置参数，不用加 `--` |
| `--llama` | **是** | — | LLaMA 3.2 基座模型所在**目录**，即放 `config.json`、`model.safetensors` 等文件的文件夹 |
| `--checkpoint` | **是** | — | 训练好的 VLM 权重**文件**，即 `.pth` 文件的完整路径 |
| `-o` / `--output` | 否 | `results.json` | 推理结果保存到的 JSON 文件路径 |
| `--frames-dir` | 否 | `frames` | 抽帧图片的暂存目录，跑完后不会自动删除 |
| `--prompt` | 否 | `Describe this image in detail.` | 对每一帧向模型提问的文本 |

### 最小运行命令

只填必填项，其余用默认值：

```bash
python run.py my_video.mp4 \
  --llama ./Llama-3.2-3B \
  --checkpoint ./checkpoints/checkpoint_epoch1_step28000.pth
```

### 完整运行命令

自定义所有选项：

```bash
python run.py my_video.mp4 \
  -o ./output/captions.json \
  --llama ./Llama-3.2-3B \
  --checkpoint ./checkpoints/checkpoint_epoch1_step28000.pth \
  --frames-dir ./temp_frames \
  --prompt "What is happening in this scene?"
```

---

## 分步运行

如果不想一键运行，也可以分两步手动执行。

### 第一步：抽帧

```bash
python extract_frames.py video.mp4 -o ./frames
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `video` | **是** | — | 输入视频路径 |
| `-o` / `--output` | 否 | `frames` | 抽帧图片输出目录 |
| `-p` / `--prefix` | 否 | `frame` | 输出文件名前缀 |

输出示例：

```
frames/
├── frame_0000s.jpg   # 第 0 秒
├── frame_0001s.jpg   # 第 1 秒
├── frame_0002s.jpg   # 第 2 秒
└── ...
```

### 第二步：批量推理

```bash
python batch_inference.py \
  --input ./frames \
  --output results.json \
  --llama ./Llama-3.2-3B \
  --checkpoint ./checkpoints/checkpoint_epoch1_step28000.pth
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--input` | **是** | — | 抽帧图片所在目录 |
| `--llama` | **是** | — | LLaMA 基座模型目录 |
| `--checkpoint` | **是** | — | checkpoint .pth 文件路径 |
| `--output` | 否 | `results.json` | 结果输出 JSON 路径 |
| `--prompt` | 否 | `Describe this image in detail.` | 每帧的推理提示词 |

---

## 输出格式

运行完成后，`results.json` 内容如下：

```json
[
  {"time_sec": 0, "frame": "frame_0000s.jpg", "caption": "A large jetliner is parked on the runway."},
  {"time_sec": 1, "frame": "frame_0001s.jpg", "caption": "The plane is taxiing towards the terminal."},
  {"time_sec": 2, "frame": "frame_0002s.jpg", "caption": "Ground crew is preparing the aircraft for departure."}
]
```

| 字段 | 含义 |
|---|---|
| `time_sec` | 帧在视频中的秒数 |
| `frame` | 帧图片文件名 |
| `caption` | 模型生成的描述文本 |

---

## 完整流程

```
视频输入 → 每秒抽帧 → VLM 推理 → 逐帧描述输出（JSON）
```

内部两步对应：

1. `extract_frames.py` — 读取视频，按 FPS 计算间隔，每秒保存一张图片
2. `batch_inference.py` — 加载模型（仅一次），遍历所有帧逐一生成描述

---

## 硬件需求

| 项目 | 最低要求 |
|---|---|
| 磁盘空间 | **10 GB**（模型 8 GB + 帧图片视视频长度） |
| 显卡显存 | **8 GB VRAM**（3B 模型 + SigLIP + 推理开销） |
| 内存 | 16 GB 以上 |

---

## 模型架构

基于 **SigLIP ViT** 视觉编码器 + **LLaMA 3.2 (3B) Instruct** 语言主干，受 LLaVA / MiniGPT-4 启发：

```
Image → SigLIP ViT → MLP Projector → <IMG_START> ... <IMG_END> → LLaMA 3.2 → Text
```

- **Multi-Token 视觉注入**：使用 64 个 patch token，保留空间信息
- **可学习视觉位置编码**：投影后叠加位置信息
- **LoRA 微调**：低秩适配器降低显存，消费级显卡可训练

## 训练策略

| Stage | 数据集 | 目的 | GPU |
|---|---|---|---|
| 1 | Flickr30k | 视觉-语言对齐 | AMD RX 7900 XTX 24GB |
| 2 | MS COCO | 提升泛化能力 | NVIDIA RTX 3080 20GB |

## 评估

| 指标 | 得分 | 说明 |
|---|---|---|
| CIDEr | 0.6093 | 语义一致性 |
| ROUGE-L | 0.3541 | 语言流畅度 |
| METEOR | 0.2455 | 词干/同义词匹配 |
| BLEU-1 | 0.0238 | 精确词匹配（自回归模型天然偏低） |

## 项目结构

```
├── run.py               # 一键串联脚本
├── extract_frames.py    # 视频每秒抽帧
├── batch_inference.py   # 批量 VLM 推理
├── inference.py         # 单图推理（原项目）
├── train.py             # 模型训练（原项目）
├── evaluation.py        # 模型评估（原项目）
├── merge.py             # LoRA 权重合并（原项目）
├── checkpoints/         # 训练好的模型权重（需下载）
├── Llama-3.2-3B/        # LLaMA 基座模型（需下载）
└── testpics/            # 测试图片
```

## 环境依赖

```bash
pip install torch transformers peft pillow opencv-python
```

## Acknowledgements

本项目 Fork 自 [XiaoKaite/Vision-Language-Model-with-SigLIP-LLaMA3.2](https://github.com/XiaoKaite/Vision-Language-Model-with-SigLIP-LLaMA3.2)，在其 VLM 模型基础上增加了视频抽帧与批量推理管道。

原作者 HuggingFace：[RadiumLR](https://huggingface.co/RadiumLR)，训练好的模型权重托管于 [Siglip-LLama3.2-VLM-Model](https://huggingface.co/RadiumLR/Siglip-LLama3.2-VLM-Model)。

模型部分受 LLaVA、MiniGPT-4 启发。

## License

遵循 SigLIP 及 LLaMA 的原始许可。
