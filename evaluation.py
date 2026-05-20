import json
import torch
import nltk

from tqdm import tqdm
from torch.utils.data import DataLoader

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

from train import Flickr30kDataset
from train import VisionLLM
from transformers import AutoProcessor
import warnings
warnings.filterwarnings("ignore",category=UserWarning, module="transformers")

# =========================
# NLTK
# =========================
nltk.download("wordnet")
nltk.download("omw-1.4")

# =========================
# Config
# =========================
LLM_PATH = r"D:\prj_wsl\Llama-3.2-3B"

CHECKPOINT_PATH = r"D:\prj_wsl\FINAL_MODEL\checkpoint_epoch1_step28000.pth"

IMAGE_DIR = r"D:\prj_wsl\validation\val2017"

CAPTION_FILE = r"D:\prj_wsl\validation\captions.txt"

BATCH_SIZE = 1

SAVE_JSON = True
#==========================
# Processor
#==========================
processor = AutoProcessor.from_pretrained(
    "google/siglip-base-patch16-224"
)
transform = lambda image: processor(
    images=image,
    return_tensors="pt"

)["pixel_values"].squeeze(0)
# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# =========================
# Load Model
# =========================

print("Loading model...")

model = VisionLLM(
    llm_path=LLM_PATH
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

state_dict = checkpoint["model_state_dict"]

# -------------- 核心修复逻辑 --------------
# 获取当前模型中期望的词表大小 (128257)
target_vocab_size = model.llm.config.vocab_size

for key in ["llm.base_model.model.model.embed_tokens.weight", "llm.base_model.model.lm_head.weight"]:
    if key in state_dict:
        ckpt_shape = state_dict[key].shape
        # 如果 checkpoint 里的形状和当前模型不一致
        if ckpt_shape[0] != target_vocab_size:
            print(f"检测到维度不匹配: {key} 形状为 {list(ckpt_shape)}，正在将其扩充至 [{target_vocab_size}, {ckpt_shape[1]}]...")
            
            # 创建一个符合当前模型维度 (128257, 3072) 的新全零张量（或随机张量）
            new_weight = torch.zeros((target_vocab_size, ckpt_shape[1]), dtype=state_dict[key].dtype, device=state_dict[key].device)
            
            # 把原来的 128256 个 token 的权重复制过去
            new_weight[:ckpt_shape[0], :] = state_dict[key]
            
            # 如果是最后那一个新加的 <image> token，用随机初始化兜底（防止全零影响模型）
            new_weight[ckpt_shape[0]:, :] = torch.randn((target_vocab_size - ckpt_shape[0], ckpt_shape[1]), dtype=state_dict[key].dtype, device=state_dict[key].device) * 0.02
            
            # 替换掉原有的不合规权重
            state_dict[key] = new_weight
# ------------------------------------------

# 此时 state_dict 里的权重形状已经完美契合 128257，可以安全加载了
model.load_state_dict(
    state_dict,
    strict=False
)

model.to(device)
model.eval()

print("Model loaded.")

# =========================
# Dataset
# =========================
dataset = Flickr30kDataset(
    image_dir=IMAGE_DIR,
    captions_file=CAPTION_FILE,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Dataset size:", len(dataset))

# =========================
# Metrics
# =========================
rouge = rouge_scorer.RougeScorer(
    ['rougeL'],
    use_stemmer=True
)

cider_scorer = Cider()

bleu_scores = []
meteor_scores = []
rouge_scores = []

gts = {}
res = {}

all_results = []

# =========================
# Evaluation
# =========================
print("\nStarting evaluation...\n")

with torch.no_grad():

    for idx, (image, gt_caption) in enumerate(tqdm(loader)):

        image = image.to(device)

        if isinstance(gt_caption, (list,tuple)):
            gt_caption = gt_caption[0]
        prefix="Describe the image in detail: "
        if gt_caption.startswith(prefix):
            gt_caption = gt_caption[len(prefix):]

        # =========================
        # Generate Caption
        # =========================
        pred_caption = model.generate(image)

        # =========================
        # BLEU
        # =========================
        bleu = sentence_bleu(
            [gt_caption.split()],
            pred_caption.split()
        )

        bleu_scores.append(bleu)

        # =========================
        # METEOR
        # =========================
        meteor = meteor_score(
            [gt_caption.split()],
            pred_caption.split()
        )

        meteor_scores.append(meteor)

        # =========================
        # ROUGE-L
        # =========================
        rouge_l = rouge.score(
            gt_caption,
            pred_caption
        )["rougeL"].fmeasure

        rouge_scores.append(rouge_l)

        # =========================
        # CIDEr
        # =========================
        gts[idx] = [gt_caption]

        res[idx] = [pred_caption]

        # =========================
        # Save Sample
        # =========================
        sample = {
            "id": idx,
            "gt": gt_caption,
            "pred": pred_caption,
            "bleu": bleu,
            "meteor": meteor,
            "rougeL": rouge_l
        }

        all_results.append(sample)

        # =========================
        # Print Samples
        # =========================
        if idx < 10:

            print("\n========================")
            print(f"Sample {idx}")
            print("------------------------")
            print("GT   :", gt_caption)
            print("PRED :", pred_caption)

# =========================
# CIDEr Final
# =========================
print("\nCalculating CIDEr...")

cider_score, _ = cider_scorer.compute_score(
    gts,
    res
)

# =========================
# Final Metrics
# =========================
avg_bleu = sum(bleu_scores) / len(bleu_scores)

avg_meteor = sum(meteor_scores) / len(meteor_scores)

avg_rouge = sum(rouge_scores) / len(rouge_scores)

print("\n================================")
print("FINAL RESULTS")
print("================================")

print(f"BLEU     : {avg_bleu:.4f}")

print(f"METEOR   : {avg_meteor:.4f}")

print(f"ROUGE-L  : {avg_rouge:.4f}")

print(f"CIDEr    : {cider_score:.4f}")

print("================================")

# =========================
# Save JSON
# =========================
if SAVE_JSON:

    output = {
        "BLEU": avg_bleu,
        "METEOR": avg_meteor,
        "ROUGE-L": avg_rouge,
        "CIDEr": cider_score,
        "samples": all_results
    }

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nSaved evaluation_results.json")