# Video-to-Text: 视频逐秒抽帧 + VLM 看图说话

输入一段视频，按每秒切分帧画面，通过训练好的 SigLIP + LLaMA 3.2 视觉语言模型（VLM）为每一帧生成自然语言描述。

支持的场景：视频内容摘要、监控关键帧提取、辅助标注等。

## 整体流程

```
视频输入 → 每秒抽帧 → VLM 推理 → 逐帧描述输出（JSON）
```

## 模型架构

基于 **SigLIP ViT** 视觉编码器 + **LLaMA 3.2 (3B) Instruct** 语言主干，受 LLaVA / MiniGPT-4 启发：

```
Image ────────► SigLIP ViT ────────► Visual Tokens
                                          │
                                    MLP Projector
                                          │
                              Visual Embeddings + Pos Embed
                                          │
                           <IMG_START> ... <IMG_END>
                                          │
                                    LLaMA 3.2 LLM ────► Generated Text
```

关键技术点：

- **Multi-Token 视觉注入**：使用 patch-level 视觉 token（64个），而非仅 CLS token，保留空间信息
- **可学习视觉位置编码**：投影后叠加位置信息，防止空间语义丢失
- **模态分隔符**：`<IMG_START> ... <IMG_END>` 包裹视觉 token，提升多模态对齐稳定性
- **LoRA 微调**：低秩适配器降低显存开销，消费级显卡即可训练

## 训练策略

| Stage | 数据集 | 目的 | GPU |
|---|---|---|---|
| Stage 1 | Flickr30k | 视觉-语言初始对齐 | AMD RX 7900 XTX 24GB |
| Stage 2 | MS COCO | 提升描述质量与泛化 | NVIDIA RTX 3080 20GB |

训练内容：MLP 投影器 + LoRA 微调 LLaMA + 视觉位置编码 + 模态分隔 token。

## 评估指标

| 指标 | 得分 | 说明 |
|---|---|---|
| CIDEr | 0.6093 | 语义一致性高，捕获描述主旨 |
| ROUGE-L | 0.3541 | 语言流畅度 |
| METEOR | 0.2455 | 词干/同义词匹配 |
| BLEU-1 | 0.0238 | 自回归模型倾向丰富细节，非逐字匹配 |

## 项目结构

```
├── run.py               # 一键：视频 → 抽帧 → 推理
├── extract_frames.py    # 视频每秒抽帧
├── batch_inference.py   # 批量 VLM 推理
├── inference.py         # 单图推理
├── train.py             # 模型训练
├── evaluation.py        # 模型评估
├── merge.py             # LoRA 权重合并
├── checkpoints/         # 训练好的模型权重
├── Llama-3.2-3B/        # LLaMA 3.2 基座模型
└── testpics/            # 测试图片
```

## 环境依赖

```bash
pip install torch transformers peft pillow opencv-python
```

## 模型准备

| 组件 | 来源 | 存放路径 |
|---|---|---|
| LLaMA 3.2 (3B) 基座 | Meta 官方下载 | `Llama-3.2-3B/` |
| 训练好的 checkpoint | [HuggingFace](https://huggingface.co/RadiumLR/Siglip-LLama3.2-VLM-Model) | `checkpoints/` |

> **注意：** 所有 Python 文件中的路径变量请根据本地实际路径修改。

## 使用方法

### 方式一：一键运行（推荐）

```bash
python run.py video.mp4 \
  -o results.json \
  --llama ./Llama-3.2-3B \
  --checkpoint ./checkpoints/checkpoint_epoch1_step28000.pth
```

### 方式二：分步运行

**Step 1 — 抽帧**

```bash
python extract_frames.py video.mp4 -o ./frames
```

**Step 2 — 批量推理**

```bash
python batch_inference.py \
  --input ./frames \
  --output results.json \
  --llama ./Llama-3.2-3B \
  --checkpoint ./checkpoints/checkpoint_epoch1_step28000.pth
```

### 命令行参数

| 脚本 | 参数 | 说明 | 默认值 |
|---|---|---|---|
| `extract_frames.py` | `video` | 输入视频路径（必填） | — |
| | `-o` / `--output` | 输出图片目录 | `frames` |
| | `-p` / `--prefix` | 文件名前缀 | `frame` |
| `batch_inference.py` | `--input` | 抽帧图片目录（必填） | — |
| | `--output` | 结果输出 JSON 路径 | `results.json` |
| | `--llama` | LLaMA 基座模型目录（必填） | — |
| | `--checkpoint` | checkpoint .pth 路径（必填） | — |
| | `--prompt` | 每帧的推理提示词 | `Describe this image in detail.` |
| `run.py` | `video` | 输入视频路径（必填） | — |
| | `-o` / `--output` | 结果输出 JSON 路径 | `results.json` |
| | `--llama` | LLaMA 基座模型目录（必填） | — |
| | `--checkpoint` | checkpoint .pth 路径（必填） | — |
| | `--frames-dir` | 临时抽帧存放目录 | `frames` |
| | `--prompt` | 每帧的推理提示词 | `Describe this image in detail.` |

### 输出格式（results.json）

```json
[
  {"time_sec": 0, "frame": "frame_0000s.jpg", "caption": "A large jetliner is parked on the runway."},
  {"time_sec": 1, "frame": "frame_0001s.jpg", "caption": "The plane is taxiing towards the terminal."},
  {"time_sec": 2, "frame": "frame_0002s.jpg", "caption": "Ground crew is preparing the aircraft for departure."}
]
```

### 单图推理（原项目用法）

```python
from inference import run_inference
run_inference("path/to/image.jpg", "Describe this image in detail.")
```

## 示例输出

| 输入 | 输出 |
|---|---|
| 飞机跑道图 | "A large jetliner is parked on the runway." |
| 摩托车交警图 | "A police officer on a motorcycle on a city street." |

## Acknowledgements

灵感来源：LLaVA、MiniGPT-4

## License

遵循 SigLIP 及 LLaMA 的原始许可。
