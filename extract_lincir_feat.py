# Imports
import os
import sys
import torch
import clip
from PIL import Image, ImageFile
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
from phi import Phi
from datasets.dataset_utils import CIRRDataset, FashionIQDataset
import argparse
from tqdm.auto import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import config
import config

# PIL image settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

# Argument parser for dataset selection
parser = argparse.ArgumentParser(description="Extract LinCIR features for composed image retrieval")
parser.add_argument('--dataset', type=str, choices=['cirr', 'fashioniq'], default='cirr',
                    help="Specify the dataset to load: 'cirr' or 'fashioniq'")
parser.add_argument('--dataset_path', type=str, default=None,
                    help="Path to dataset (uses config.py default if not specified)")
parser.add_argument('--split', type=str, default='test1',
                    help="Dataset split to use (e.g., 'test1' for CIRR, 'test' for FashionIQ)")
parser.add_argument('--text_embeddings_dir', type=str, default=None,
                    help="Directory to save extracted embeddings (.pt files)")
parser.add_argument('--phi_vit_path', type=str, default=None,
                    help="Path to phi_best.pt model")
parser.add_argument('--phi_giga_path', type=str, default=None,
                    help="Path to phi_best_giga.pt model")
parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE,
                    help="Batch size for processing")
parser.add_argument('--num_workers', type=int, default=config.DEFAULT_NUM_WORKERS,
                    help="Number of DataLoader workers")
parser.add_argument('--device', type=str, default=config.DEFAULT_DEVICE,
                    help="Device to use (cuda or cpu)")
parser.add_argument('--cache_dir', type=str, default=None,
                    help="HuggingFace cache directory")
args = parser.parse_args()

# Setup paths using config
dataset_path = config.get_dataset_path(args.dataset, args.dataset_path)
phi_vit_path = config.get_model_path('phi_vit', args.phi_vit_path)
phi_giga_path = config.get_model_path('phi_giga', args.phi_giga_path)
cache_dir = args.cache_dir or config.HF_CACHE_DIR

# Setup output directory
if args.text_embeddings_dir:
    text_embeddings_dir = args.text_embeddings_dir
else:
    text_embeddings_dir = os.path.join(
        config.OUTPUT_PATHS['embeddings'], 
        f"{args.dataset}_{args.split}"
    )
os.makedirs(text_embeddings_dir, exist_ok=True)

print("=" * 70)
print("LinCIR Feature Extraction")
print("=" * 70)
print(f"Dataset: {args.dataset}")
print(f"Dataset Path: {dataset_path}")
print(f"Split: {args.split}")
print(f"Output Directory: {text_embeddings_dir}")
print(f"Phi ViT Model: {phi_vit_path}")
print(f"Phi Giga Model: {phi_giga_path}")
print(f"Batch Size: {args.batch_size}")
print(f"Device: {args.device}")
print("=" * 70)

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Function to encode text with pseudo tokens
def encode_with_pseudo_tokens_HF(clip_model: CLIPTextModelWithProjection, text: torch.Tensor, pseudo_tokens: torch.Tensor, num_tokens=1, return_last_states=True) -> torch.Tensor:
    """
    Encode text with pseudo tokens using HuggingFace CLIP model.
    """
    x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype)
    x = torch.where(text.unsqueeze(-1) == 259, pseudo_tokens.unsqueeze(1).type(clip_model.dtype), x)
    x = x + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
    _causal_attention_mask = _make_causal_mask(text.shape, x.dtype, device=x.device)
    
    # Call encoder - handle different versions
    try:
        encoder_output = clip_model.text_model.encoder(
            inputs_embeds=x,
            attention_mask=None,
            causal_attention_mask=_causal_attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=False
        )
        if isinstance(encoder_output, tuple):
            x_out = encoder_output[0]
            hidden_states = encoder_output[1] if len(encoder_output) > 1 else None
        else:
            x_out = encoder_output.last_hidden_state
            hidden_states = encoder_output.hidden_states
    except TypeError:
        # Newer versions don't accept return_dict
        encoder_output = clip_model.text_model.encoder(
            inputs_embeds=x,
            attention_mask=None,
            causal_attention_mask=_causal_attention_mask
        )
        if hasattr(encoder_output, 'last_hidden_state'):
            x_out = encoder_output.last_hidden_state
            hidden_states = None
        else:
            x_out = encoder_output
            hidden_states = None
    
    prompt_embeds = hidden_states[-2] if hidden_states is not None else x_out
    x_last = clip_model.text_model.final_layer_norm(x_out)
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device), text.to(dtype=torch.int, device=x_last.device).argmax(dim=-1)]
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)
    if return_last_states:
        return x, x_last, prompt_embeds
    else:
        return x

# Model and processor setup
clip_model_name = config.CLIP_MODELS['vit_large']
clip_model_name2 = config.CLIP_MODELS['giga']

clip_preprocess = CLIPImageProcessor(
    crop_size=config.IMAGE_PREPROCESS['crop_size'],
    do_center_crop=True,
    do_convert_rgb=True,
    do_normalize=True,
    do_rescale=True,
    do_resize=True,
    image_mean=config.IMAGE_PREPROCESS['image_mean'],
    image_std=config.IMAGE_PREPROCESS['image_std'],
    resample=3,
    size=config.IMAGE_PREPROCESS['size'],
)

print("\n📥 Loading CLIP models...")
# Load CLIP models
clip_text_model1 = CLIPTextModelWithProjection.from_pretrained(
    clip_model_name, torch_dtype=torch.float32, cache_dir=cache_dir
).float().to(args.device)

clip_vision_model1 = CLIPVisionModelWithProjection.from_pretrained(
    clip_model_name, torch_dtype=torch.float32, cache_dir=cache_dir
).float().to(args.device)

clip_text_model2 = CLIPTextModelWithProjection.from_pretrained(
    clip_model_name2, torch_dtype=torch.float32, cache_dir=cache_dir
).float().to(args.device)

clip_vision_model2 = CLIPVisionModelWithProjection.from_pretrained(
    clip_model_name2, torch_dtype=torch.float32, cache_dir=cache_dir
).float().to(args.device)

print("✅ CLIP models loaded")

print("\n📥 Loading Phi models...")
# Load Phi models
phi = Phi(input_dim=768, hidden_dim=768 * 4, output_dim=768, dropout=0)
phi_2 = Phi(input_dim=1280, hidden_dim=1280 * 4, output_dim=1280, dropout=0)

phi.load_state_dict(torch.load(phi_vit_path, map_location=args.device)[phi.__class__.__name__])
phi_2.load_state_dict(torch.load(phi_giga_path, map_location=args.device)[phi.__class__.__name__])

phi = phi.to(device=args.device).eval()
phi_2 = phi_2.to(device=args.device).eval()
print("✅ Phi models loaded")

# Dataset selection
print("\n📂 Loading dataset...")
if args.dataset == 'cirr':
    dataset = CIRRDataset(dataset_path, split=args.split, preprocess=clip_preprocess)
elif args.dataset == 'fashioniq':
    dataset = FashionIQDataset(
        dataset_path, 
        split=args.split, 
        dress_types=config.FASHIONIQ_CATEGORIES, 
        preprocess=clip_preprocess,
        use_downloaded_images=True  # Try local images first, fall back to URLs
    )
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

print(f"✅ Dataset loaded: {len(dataset)} samples")

# DataLoader setup
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

# Main loop for extracting and saving features
print(f"\n🚀 Extracting features...")
processed_count = 0
error_count = 0

for batch in tqdm(dataloader, desc=f"Extracting features ({args.dataset})", unit="batch"):
    try:
        # Get pairid (CIRR) or construct from candidate/target (FashionIQ)
        if args.dataset == 'cirr':
            pairids = batch['pairid']
        else:
            # For FashionIQ, create unique IDs from candidate and target
            pairids = [f"{c}_{t}" for c, t in zip(batch['candidate'], batch['target'])]
        
        reference_image = batch['reference_image']

        # Extract image features
        image_features1 = clip_vision_model1(reference_image.to(args.device)).image_embeds
        image_features2 = clip_vision_model2(reference_image.to(args.device)).image_embeds

        # Predict pseudo tokens
        predicted_tokens = phi(image_features1.to(torch.float32))
        predicted_tokens2 = phi_2(image_features2.to(torch.float32))

        # Prepare input captions
        relative_captions = batch['relative_caption']
        input_captions = [f"a photo of $ that {cap}" for cap in relative_captions]
        tokenized_input_captions = clip.tokenize(input_captions, context_length=77, truncate=True).to(args.device)
        tokenized_input_captions2 = clip.tokenize(input_captions, context_length=77, truncate=True).to(args.device)

        # Encode with pseudo tokens
        _, pooled, conditioning = encode_with_pseudo_tokens_HF(clip_text_model1, tokenized_input_captions, predicted_tokens)
        pooled2, _, conditioning2 = encode_with_pseudo_tokens_HF(clip_text_model2, tokenized_input_captions2, predicted_tokens2)

        # Save results
        for idx in range(len(reference_image)):
            save_dict = {
                'pooled': pooled[idx].cpu().data,
                'conditioning': conditioning[idx].cpu().data,
                'pooled2': pooled2[idx].cpu().data,
                'conditioning2': conditioning2[idx].cpu().data
            }

            save_path = os.path.join(text_embeddings_dir, f"{pairids[idx]}.pt")
            torch.save(save_dict, save_path)
            processed_count += 1
    
    except Exception as e:
        print(f"\n⚠️  Error processing batch: {str(e)}")
        error_count += 1
        continue

print("\n" + "=" * 70)
print("✅ Feature extraction complete!")
print(f"Total processed: {processed_count}")
print(f"Errors: {error_count}")
print(f"Output directory: {text_embeddings_dir}")
print("=" * 70)
       
