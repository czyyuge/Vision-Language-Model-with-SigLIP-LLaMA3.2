import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel
from torchvision import transforms
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
# ======================
# 1. Dataset
# ======================
class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = []

        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) < 2:
                    continue
                img, caption = parts
                self.data.append((img, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, "Describe the image in detail: " + caption


# ======================
# 2. Model
# ======================
class VisionLLM(nn.Module):
    def __init__(self, llm_path):
        super().__init__()

        # Vision encoder
        self.vision = SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        vision_dim = self.vision.config.hidden_size

        # LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_tokens(["<image>"])
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            #device_map="auto"
        )
        self.llm.resize_token_embeddings(
            len(self.tokenizer)
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.llm = get_peft_model(self.llm, lora_config)
        
        self.llm.gradient_checkpointing_enable()
        self.img_start_token = nn.Parameter(torch.randn(1, 1, self.llm.config.hidden_size))  # Learnable token to indicate the start of visual input
        self.img_end_token = nn.Parameter(torch.randn(1, 1, self.llm.config.hidden_size))    # Learnable token to indicate the end of visual input
        self.max_visual_tokens = 64
        self.visual_pos_embed = nn.Parameter(torch.randn(1, self.max_visual_tokens, self.llm.config.hidden_size))  # Definition of positional embedding for visual tokens


        llm_dim = self.llm.config.hidden_size

        # Projector（核心）
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            #nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            #nn.GELU(),
            #nn.Linear(llm_dim, llm_dim),
        )
        self.projector = self.projector.to(torch.bfloat16)
        self.img_start_token.data = self.img_start_token.data.to(torch.bfloat16)
        self.img_end_token.data = self.img_end_token.data.to(torch.bfloat16)
        self.visual_pos_embed.data = self.visual_pos_embed.data.to(torch.bfloat16)
    
    def forward(self, images, input_ids, attention_mask):
        # Vision
        vision_outputs = self.vision(pixel_values=images.to(self.vision.dtype))
        vision_feat = vision_outputs.last_hidden_state[:, 1:65]
        vision_feat = vision_feat.to(torch.bfloat16)
        visual_emb = self.projector(vision_feat)
        #vision_feat = vision_feat.mean(dim=1, keepdim=True) #pooling
        # Project
        visual_emb = self.projector(vision_feat)
        B, N, _ =visual_emb.shape
        visual_emb = visual_emb + self.visual_pos_embed[:, :N, :] #Adding positional embedding to visual tokens
        # Text embedding
        text_emb = self.llm.get_input_embeddings()(input_ids)

        # Concat
        img_start=self.img_start_token.expand(B, -1, -1)
        img_end=self.img_end_token.expand(B, -1, -1)
        inputs_embeds = torch.cat([img_start, visual_emb, img_end, text_emb], dim=1)
        
        # Mask
        img_special_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        visual_mask = torch.ones((B, N), dtype=attention_mask.dtype,
                                 device=attention_mask.device)
        img_end_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device) # end token
        attention_mask = torch.cat([img_special_mask, visual_mask, img_end_mask, attention_mask], dim=1)

        # Forward
        ignore_labels = torch.full((B, N + 2), -100, dtype=input_ids.dtype, device=input_ids.device)
        full_labels = torch.cat([ignore_labels, input_ids], dim=1)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels
        )

        return outputs
    def generate(self, image, max_new_tokens=40):

        self.eval()

        with torch.no_grad():

            # =========================
            # Vision Encoder
            # =========================
            vision_outputs = self.vision(
                pixel_values=image.to(self.vision.dtype)
            )

            vision_feat = vision_outputs.last_hidden_state[:, 1:65]

            vision_feat = vision_feat.to(torch.bfloat16)

            # =========================
            # Projector
            # =========================
            visual_emb = self.projector(vision_feat)

            B, N, _ = visual_emb.shape

            visual_emb = visual_emb + self.visual_pos_embed[:, :N, :]

            # =========================
            # Image Special Tokens
            # =========================
            img_start = self.img_start_token.expand(B, -1, -1).to(
                visual_emb.device,
                dtype=visual_emb.dtype
            )

            img_end = self.img_end_token.expand(B, -1, -1).to(
                visual_emb.device,
                dtype=visual_emb.dtype
            )

            # =========================
            # Prompt
            # =========================
            prompt = "User: <image>\nDescribe this image.\nAssistant:"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(image.device)

            # =========================
            # Text Embedding
            # =========================
            text_emb = self.llm.get_input_embeddings()(
                inputs.input_ids
            )

            text_emb = text_emb.to(torch.bfloat16)

            # =========================
            # Combine Embeddings
            # =========================
            inputs_embeds = torch.cat(
                [
                    img_start,
                    visual_emb,
                    img_end,
                    text_emb
                ],
                dim=1
            )

            # =========================
            # Attention Mask
            # =========================
            visual_mask = torch.ones(
                (B, N + 2),
                dtype=inputs.attention_mask.dtype,
                device=image.device
            )

            attention_mask = torch.cat(
                [
                    visual_mask,
                    inputs.attention_mask
                ],
                dim=1
            )

            # =========================
            # Generate
            # =========================
            output_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )

            text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            # =========================
            # Clean Output
            # =========================
            if "Assistant:" in text:
                text = text.split("Assistant:")[-1]

            return text.strip()

def save_checkpoint(model, optimizer, epoch, step, loss, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 构建文件名，包含 epoch 和 step 方便追溯
    checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pth"
    full_path = os.path.join(save_path, checkpoint_name)
    
    # 核心：保存模型参数 + 优化器状态
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, full_path)
    # 同时保存一次 tokenizer，防止丢失
    model.tokenizer.save_pretrained(save_path)
    print(f"--- Checkpoint 已保存至: {full_path} ---")

# ======================
# 3. Freeze
# ======================
def freeze_model(model):
    for p in model.vision.parameters():
        p.requires_grad = False

    #for p in model.llm.parameters():
    #    p.requires_grad = False

    for p in model.projector.parameters():
        p.requires_grad = True
    # train visual tokens
    model.img_start_token.requires_grad = True
    model.img_end_token.requires_grad = True
    model.visual_pos_embed.requires_grad = True


# ======================
# 4. Main
# ======================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # Dataset
    dataset = Flickr30kDataset(
        image_dir=r"D:\prj_wsl\coco\train2017\train2017",
        captions_file=r"D:\prj_wsl\coco\captions.txt",
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # Model
    model = VisionLLM("Llama-3.2-3B/")
    freeze_model(model)
    #for name, p in model.llm.named_parameters():
    #    if "layers.0" in  name or "layers.1" in name or "layers.2" in name or "layers.3" in name:
    #        p.requires_grad=True
    model.to(device)
    trainable_params = [p for p in model.llm.parameters() if p.requires_grad]
    accum_steps = 8
    optimizer = AdamW(trainable_params, lr=3e-6)
    optimizer.zero_grad()
    resume_path = r"D:\prj_wsl\saved_models\coco_3e-6\checkpoint_epoch0_step56000.pth"

    if os.path.exists(resume_path):
        print(f"加载checkpoint: {resume_path}")

        checkpoint = torch.load(resume_path, map_location="cpu")

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr']=3e-6

        start_epoch = 0
        start_step = 0

        print(f"恢复训练: epoch={start_epoch}, step={start_step}")
    else:
        start_epoch = 0
        start_step = 0
    # Training
    for epoch in range(start_epoch,10):
        for i, (images, texts) in enumerate(dataloader):
            if epoch==start_epoch and i<start_step:
                continue

            images = images.to(device)

            inputs = model.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            model.tokenizer.padding_side = "right"
            inputs = inputs.to(device)

            outputs = model(
                images,
                inputs.input_ids,
                inputs.attention_mask
            )

            loss = outputs.loss/accum_steps

            loss.backward()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % (10*accum_steps) == 0:
                print(f"Epoch {epoch} Step {i} Loss {accum_steps*loss.item():.4f}")
            if i%(500*accum_steps)==0:
                save_checkpoint(
                    model, 
                    optimizer, 
                    epoch, 
                    i, 
                    loss.item() * accum_steps, 
                    r"E:\checkpoints"
                )
    save_dir = r"E:\checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "vision_llm.pth"))
    


if __name__ == "__main__":
    main()
    
