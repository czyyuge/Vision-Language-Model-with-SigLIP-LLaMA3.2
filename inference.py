import torch
from PIL import Image
from torchvision import transforms
import os
from transformers import AutoTokenizer
from train import VisionLLM
from peft import LoraConfig, get_peft_model
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_inference(image_path, user_prompt):
    # ===== 1. PATH =====
    llama_origin_path = r"D:\prj_wsl\Llama-3.2-3B"
    checkpoint_path = r"D:\prj_wsl\checkpoints\checkpoint_epoch1_step28000.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("Loading llama...")
    model = VisionLLM(llama_origin_path)

    print("Loading weight...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict']
    model.tokenizer = AutoTokenizer.from_pretrained(r"D:\prj_wsl\checkpoints")
    model.tokenizer.pad_token = model.tokenizer.eos_token

    model.load_state_dict(state_dict, strict=False)

    model.to(device, dtype=dtype)
    model.eval()

    print("Accomplished Model Loading.")

    # ===== 2. Image Tokenization =====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)

    # ===== 3. Prompt=====
    prompt =user_prompt

    model.tokenizer.padding_side = "right"
    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # ===== 4. Construct inputs_embeds that match forward=====
    with torch.no_grad():

        # ---- vision ----
        vision_outputs = model.vision(pixel_values=image_tensor.to(model.vision.dtype))
        vision_feat = vision_outputs.last_hidden_state[:, 1:65]
        vision_feat = vision_feat.to(dtype)

        # ---- projector ----
        visual_emb = model.projector(vision_feat)

        B, N, _ = visual_emb.shape

        # ---- Add positional embedding ----
        visual_emb = visual_emb + model.visual_pos_embed[:, :N, :]

        # ---- Text embedding ----
        text_emb = model.llm.get_input_embeddings()(input_ids)

        # ---- img_start / img_end ----
        img_start = model.img_start_token.expand(B, 1, -1)
        img_end = model.img_end_token.expand(B, 1, -1)

        inputs_embeds = torch.cat([
            img_start,
            visual_emb,
            img_end,
            text_emb
        ], dim=1)

        # ----  attention mask ----
        img_start_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)
        visual_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=device)
        img_end_mask = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)

        full_attention_mask = torch.cat([
            img_start_mask,
            visual_mask,
            img_end_mask,
            attention_mask
        ], dim=1)

        print(f"inputs_embeds shape: {inputs_embeds.shape}")
        print(f"attention_mask shape: {full_attention_mask.shape}")

        # ===== 5. Generation =====
        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,

            max_new_tokens=32,
            min_new_tokens=2,          # Prevent instant EOS

            do_sample=False,            # Prevent collapsing
            #temperature=0.5,
            #top_p=0.9,

            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id
        )

    # ===== 6. EOS issue debugging =====
    print("EOS token id:", model.tokenizer.eos_token_id)
    print("First generated token:", output_ids[0][0].item())

    # ===== 7. Decoding =====
    output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "="*50)
    print("Inference Result: ")
    print(output_text)
    print("="*50)

    return output_text


if __name__ == "__main__":
    test_img = r"D:\prj_wsl\testpics\testpic2.jpg"
    test_query = "Describe this image in detail."

    run_inference(test_img, test_query)
