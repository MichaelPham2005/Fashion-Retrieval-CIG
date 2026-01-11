"""
Debug script để test SDXL loading
"""

import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config

print("="*70)
print("SDXL Loading Debug")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Check paths
sdxl_checkpoint = config.MODEL_PATHS['sdxl_checkpoint']
unet_path = os.path.join(sdxl_checkpoint, 'checkpoint-20000', 'unet')

print(f"\nSDXL Checkpoint: {sdxl_checkpoint}")
print(f"UNet Path: {unet_path}")
print(f"UNet exists: {os.path.exists(unet_path)}")

if os.path.exists(unet_path):
    files = os.listdir(unet_path)
    print(f"UNet files: {files}")

print("\n" + "="*70)
print("Step 1: Loading UNet")
print("="*70)

try:
    from diffusers import UNet2DConditionModel
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    print(f"Loading from: {unet_path}")
    print(f"dtype: {dtype}")
    print(f"use_safetensors: True")
    
    unet = UNet2DConditionModel.from_pretrained(
        unet_path,
        use_safetensors=True,
        torch_dtype=dtype
    )
    print("✅ UNet loaded to CPU")
    
    unet = unet.to(device)
    print(f"✅ UNet moved to {device}")
    
except Exception as e:
    print(f"❌ Error loading UNet: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("Step 2: Loading VAE")
print("="*70)

try:
    from diffusers import AutoencoderKL
    
    vae_repo = config.SDXL_MODELS['vae']
    print(f"VAE repo: {vae_repo}")
    print(f"Cache dir: {config.HF_CACHE_DIR}")
    
    vae = AutoencoderKL.from_pretrained(
        vae_repo,
        torch_dtype=dtype,
        cache_dir=config.HF_CACHE_DIR
    )
    print("✅ VAE loaded to CPU")
    
    vae = vae.to(device)
    print(f"✅ VAE moved to {device}")
    
except Exception as e:
    print(f"❌ Error loading VAE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("Step 3: Creating SDXL Pipeline")
print("="*70)

try:
    from diffusers import StableDiffusionXLPipeline
    
    sdxl_repo = config.SDXL_MODELS['base']
    print(f"SDXL base repo: {sdxl_repo}")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_repo,
        unet=unet,
        vae=vae,
        torch_dtype=dtype,
        cache_dir=config.HF_CACHE_DIR
    )
    print("✅ Pipeline created")
    
    pipe = pipe.to(device)
    print(f"✅ Pipeline moved to {device}")
    
except Exception as e:
    print(f"❌ Error creating pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nSDXL pipeline is ready to use.")
print("You can now run: python api_pseudo_target.py")
