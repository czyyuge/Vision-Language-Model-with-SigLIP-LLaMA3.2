import torch
import os
import json
import argparse
import re
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from train import VisionLLM


def natural_sort_key(name):
    """Sort filenames by the numeric second extracted from _XXXXs."""
    m = re.search(r"_(\d+)s", name)
    return int(m.group(1)) if m else 0


class VideoCaptioner:
    def __init__(self, llama_path, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        print(f"[1/3] Loading LLaMA backbone from {llama_path} ...")
        self.model = VisionLLM(llama_path)

        print(f"[2/3] Loading checkpoint from {checkpoint_path} ...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        tokenizer_dir = os.path.dirname(checkpoint_path)
        self.model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

        # 对齐模型 vocab_size 与 checkpoint，防止维度不匹配导致崩溃
        for k, v in state_dict.items():
            if k.endswith("embed_tokens.weight"):
                ckpt_vocab_size = v.size(0)
                self.model.llm.base_model.resize_token_embeddings(ckpt_vocab_size)
                print(f"Adjusted embed_tokens vocab size to {ckpt_vocab_size}")
                break

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

        print("[3/3] Model ready.\n")

    def caption(self, image_path, prompt="Describe this image in detail."):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device, dtype=self.dtype)

        self.model.tokenizer.padding_side = "right"
        inputs = self.model.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            vision_outputs = self.model.vision(pixel_values=image_tensor.to(self.model.vision.dtype))
            vision_feat = vision_outputs.last_hidden_state[:, 1:65].to(self.dtype)

            visual_emb = self.model.projector(vision_feat)
            B, N, _ = visual_emb.shape
            visual_emb = visual_emb + self.model.visual_pos_embed[:, :N, :]

            text_emb = self.model.llm.get_input_embeddings()(input_ids)

            img_start = self.model.img_start_token.expand(B, 1, -1)
            img_end = self.model.img_end_token.expand(B, 1, -1)

            inputs_embeds = torch.cat([img_start, visual_emb, img_end, text_emb], dim=1)

            img_start_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=self.device)
            visual_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=self.device)
            img_end_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=self.device)

            full_attention_mask = torch.cat(
                [img_start_mask, visual_mask, img_end_mask, attention_mask], dim=1
            )

            output_ids = self.model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=32,
                min_new_tokens=2,
                do_sample=False,
                eos_token_id=self.model.tokenizer.eos_token_id,
                pad_token_id=self.model.tokenizer.pad_token_id,
            )

        return self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def batch_inference(
    frames_dir,
    output_json,
    llama_path,
    checkpoint_path,
    prompt="Describe this image in detail.",
):
    files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=natural_sort_key,
    )

    if not files:
        print(f"No image files found in {frames_dir}")
        return

    print(f"Found {len(files)} frames to process.\n")

    captioner = VideoCaptioner(llama_path, checkpoint_path)

    results = []
    for i, filename in enumerate(files):
        img_path = os.path.join(frames_dir, filename)
        print(f"[{i + 1}/{len(files)}] {filename} ...", end=" ", flush=True)

        caption = captioner.caption(img_path, prompt)
        second = natural_sort_key(filename)

        results.append({"time_sec": second, "frame": filename, "caption": caption})
        print(caption[:80])

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(results)} captions saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch VLM inference on extracted frames")
    parser.add_argument("--input", required=True, help="Directory of extracted frames")
    parser.add_argument("--output", default="results.json", help="Output JSON path")
    parser.add_argument("--llama", required=True, help="Path to LLaMA-3.2-3B directory")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--prompt", default="Describe this image in detail.", help="Prompt for each frame")
    args = parser.parse_args()

    batch_inference(args.input, args.output, args.llama, args.checkpoint, args.prompt)
