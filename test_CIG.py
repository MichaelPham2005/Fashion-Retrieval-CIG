import os
import sys
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datasets.dataset_utils import ComposedEmbedsDataset
import config


"""
Parameters
"""
import argparse

parser = argparse.ArgumentParser(description="Generate composed images with SDXL using embeddings.")
parser.add_argument("--text_embeddings_dir", type=str, default=None,
                    help="Directory containing {pairid}.pt embedding files")
parser.add_argument("--dataset_dir", type=str, default=None,
                    help="Dataset root directory (for CIRR)")
parser.add_argument("--dataset", type=str, default='cirr', choices=['cirr', 'fashioniq'],
                    help="Dataset name")
parser.add_argument("--split", type=str, default='test1',
                    help="Dataset split (e.g., 'test1' for CIRR)")
parser.add_argument("--model_path", type=str, default=None,
                    help="Path to SDXL UNet checkpoint directory (expects a 'unet' subfolder)")
parser.add_argument("--save_path", type=str, default=None,
                    help="Directory to save generated images")
parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE,
                    help="Batch size for DataLoader and batched generation")
parser.add_argument("--num_workers", type=int, default=config.DEFAULT_NUM_WORKERS,
                    help="Number of DataLoader workers")
parser.add_argument("--height", type=int, default=config.DEFAULT_IMAGE_HEIGHT,
                    help="Output image height")
parser.add_argument("--width", type=int, default=config.DEFAULT_IMAGE_WIDTH,
                    help="Output image width")
parser.add_argument("--steps", type=int, default=config.DEFAULT_INFERENCE_STEPS,
                    help="Number of inference steps")
parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED,
                    help="Base random seed per sample")
parser.add_argument("--brightness_thresh", type=float, default=config.DEFAULT_BRIGHTNESS_THRESH,
                    help="Brightness threshold to accept an image")
parser.add_argument("--max_retries", type=int, default=config.DEFAULT_MAX_RETRIES,
                    help="Max retries per image if brightness is too low")
parser.add_argument("--cache_dir", type=str, default=None,
                    help="HuggingFace cache directory")
parser.add_argument("--vae_repo", type=str, default=None,
                    help="VAE repo id for SDXL")
parser.add_argument("--sdxl_repo", type=str, default=None,
                    help="SDXL base repo id")
parser.add_argument("--device", type=str, default=config.DEFAULT_DEVICE,
                    help="Device to use (cuda or cpu)")
args = parser.parse_args(sys.argv[1:])

# Setup paths using config
dataset_dir = config.get_dataset_path(args.dataset, args.dataset_dir)
cache_dir = args.cache_dir or config.HF_CACHE_DIR
vae_repo = args.vae_repo or config.SDXL_MODELS['vae']
sdxl_repo = args.sdxl_repo or config.SDXL_MODELS['base']
model_path = args.model_path or config.MODEL_PATHS['sdxl_checkpoint']

# Setup embeddings directory
if args.text_embeddings_dir:
    text_embeddings_dir = args.text_embeddings_dir
else:
    text_embeddings_dir = os.path.join(
        config.OUTPUT_PATHS['embeddings'],
        f"{args.dataset}_{args.split}"
    )

# Setup output directory
if args.save_path:
    save_path = args.save_path
else:
    save_path = os.path.join(
        config.OUTPUT_PATHS['generated_images'],
        f"{args.dataset}_{args.split}"
    )

# Resolve and create directories
os.makedirs(save_path, exist_ok=True)
os.makedirs(text_embeddings_dir, exist_ok=True)

print("=" * 70)
print("Composed Image Generation with SDXL")
print("=" * 70)
print(f"Dataset: {args.dataset}")
print(f"Dataset Directory: {dataset_dir}")
print(f"Split: {args.split}")
print(f"Embeddings Directory: {text_embeddings_dir}")
print(f"Model Path: {model_path}")
print(f"Output Directory: {save_path}")
print(f"Batch Size: {args.batch_size}")
print(f"Image Size: {args.height}x{args.width}")
print(f"Inference Steps: {args.steps}")
print(f"Brightness Threshold: {args.brightness_thresh}")
print(f"Device: {args.device}")
print("=" * 70)

# Data
print("\n📂 Loading dataset...")
dataset = ComposedEmbedsDataset(dataset_dir, text_embeddings_dir, split=args.split)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print(f"✅ Dataset loaded: {len(dataset)} samples")

dtype = torch.float16
device = args.device

print("\n📥 Loading SDXL models...")
unet = UNet2DConditionModel.from_pretrained(
    f"{model_path}/unet", use_safetensors=True, torch_dtype=dtype
).to(device)

vae = AutoencoderKL.from_pretrained(
    vae_repo,
    torch_dtype=dtype,
    cache_dir=cache_dir
).to(device)

pipe = StableDiffusionXLPipeline.from_pretrained(
    sdxl_repo,
    unet=unet,
    vae=vae,
    torch_dtype=dtype,
    cache_dir=cache_dir
)

pipe.to(device)
print("✅ SDXL pipeline loaded")

print(f"\n🚀 Generating images...")
print(f"Total samples to process: {len(dataset)}")
generated_count = 0
skipped_count = 0
for batch in tqdm(loader, desc="Generating images", unit="batch"):
    pairids = batch['pairid']  # list of str
    prompt_embeds_batch = batch['prompt_embeds']  # Tensor [B, S, D]
    pooled2_batch = batch['pooled2']  # Tensor [B, D]

    # Filter out items already generated
    indices = [i for i, pid in enumerate(pairids) if not os.path.exists(os.path.join(save_path, f"{pid}.png"))]
    if len(indices) == 0:
        skipped_count += len(pairids)
        continue

    # Prepare sub-batch tensors and seeds
    pe_sub = prompt_embeds_batch[indices].to(device, dtype=dtype)
    pooled_sub = pooled2_batch[indices].to(device, dtype=dtype)
    pids_sub = [pairids[i] for i in indices]

    # Initial generators per item
    seeds = [args.seed for _ in range(len(indices))]
    generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]

    # Generate batched
    images = pipe(
        prompt_embeds=pe_sub,
        pooled_prompt_embeds=pooled_sub,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generators
    ).images

    # Check brightness and collect failures
    valid_mask = []
    for img in images:
        brightness = np.asarray(img).mean()
        valid_mask.append(brightness > args.brightness_thresh)

    # Save valid images
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            out_path = os.path.join(save_path, f"{pids_sub[i]}.png")
            images[i].save(out_path)
            generated_count += 1

    # Retry failing ones up to a few times (e.g., 5 attempts)
    max_retries = args.max_retries
    attempt = 1
    while attempt <= max_retries and (not all(valid_mask)):
        fail_indices = [i for i, ok in enumerate(valid_mask) if not ok]
        if len(fail_indices) == 0:
            break

        # Prepare subset for retry
        pe_retry = pe_sub[fail_indices]
        pooled_retry = pooled_sub[fail_indices]
        pids_retry = [pids_sub[i] for i in fail_indices]

        # Bump seeds for retries
        for j in range(len(fail_indices)):
            seeds[fail_indices[j]] += 1
        generators_retry = [torch.Generator(device=device).manual_seed(seeds[i]) for i in fail_indices]

        images_retry = pipe(
            prompt_embeds=pe_retry,
            pooled_prompt_embeds=pooled_retry,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            generator=generators_retry
        ).images

        # Evaluate and save passing images
        retry_valid = []
        for img in images_retry:
            brightness = np.asarray(img).mean()
            retry_valid.append(brightness > args.brightness_thresh)

        k = 0
        for idx_ok, ok in zip(fail_indices, retry_valid):
            if ok:
                out_path = os.path.join(save_path, f"{pids_sub[idx_ok]}.png")
                images_retry[k].save(out_path)
                valid_mask[idx_ok] = True
                generated_count += 1
            k += 1

        attempt += 1

print("\n" + "=" * 70)
print("✅ Image generation complete!")
print(f"Generated: {generated_count}")
print(f"Skipped (already exist): {skipped_count}")
print(f"Output directory: {save_path}")
print("=" * 70)
