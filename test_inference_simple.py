"""
Simple Inference Test - Test nhanh xem model có chạy được không

Script này test inference trên 1 ảnh để verify setup
Không cần download full dataset
"""

import os
import sys
import torch
import clip
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config
from phi import Phi

def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    """Make causal mask for attention"""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def encode_with_pseudo_tokens_HF(clip_model, text, pseudo_tokens, num_tokens=1, return_last_states=True):
    """Encode text with pseudo tokens"""
    x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype)
    x = torch.where(text.unsqueeze(-1) == 259, pseudo_tokens.unsqueeze(1).type(clip_model.dtype), x)
    x = x + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
    _causal_attention_mask = _make_causal_mask(text.shape, x.dtype, device=x.device)
    
    # Call encoder - different versions return different formats
    try:
        # Try with return_dict parameter (older versions)
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
        # Handle output based on type
        if hasattr(encoder_output, 'last_hidden_state'):
            x_out = encoder_output.last_hidden_state
            hidden_states = None
        else:
            x_out = encoder_output
            hidden_states = None
    
    # Get prompt embeddings if available
    prompt_embeds = hidden_states[-2] if hidden_states is not None else x_out
    
    # Final layer norm and projection
    x_last = clip_model.text_model.final_layer_norm(x_out)
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device), text.to(dtype=torch.int, device=x_last.device).argmax(dim=-1)]
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)
    if return_last_states:
        return x, x_last, prompt_embeds
    else:
        return x

def test_inference():
    """Test inference với 1 ảnh mẫu"""
    
    print("=" * 70)
    print("Simple Inference Test")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Check models exist
    phi_vit_path = config.get_model_path('phi_vit')
    phi_giga_path = config.get_model_path('phi_giga')
    
    if not os.path.exists(phi_vit_path) or not os.path.exists(phi_giga_path):
        print("\n❌ ERROR: Pretrained models not found!")
        print("Please download models from:")
        print("https://drive.google.com/drive/folders/1hpIpI0X26ox-uY-QdOPKDKKZlnWkftIA")
        print(f"And place them in: {config.MODEL_PATHS['phi_vit']}")
        return False
    
    print("\n📥 Loading models...")
    
    # Load CLIP models
    clip_model_name = config.CLIP_MODELS['vit_large']
    clip_text_model = CLIPTextModelWithProjection.from_pretrained(
        clip_model_name, 
        torch_dtype=torch.float32, 
        cache_dir=config.HF_CACHE_DIR
    ).float().to(device)
    
    clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
        clip_model_name,
        torch_dtype=torch.float32,
        cache_dir=config.HF_CACHE_DIR
    ).float().to(device)
    
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
    
    # Load Phi model
    phi = Phi(input_dim=768, hidden_dim=768 * 4, output_dim=768, dropout=0)
    phi.load_state_dict(torch.load(phi_vit_path, map_location=device)[phi.__class__.__name__])
    phi = phi.to(device=device).eval()
    
    print("✅ Models loaded successfully!")
    
    # Create a dummy image for testing
    print("\n🖼️  Creating test image...")
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Preprocess
    pixel_values = clip_preprocess(test_image, return_tensors='pt')['pixel_values'].to(device)
    
    # Extract image features
    print("\n🔍 Extracting image features...")
    with torch.no_grad():
        image_features = clip_vision_model(pixel_values).image_embeds
    print(f"✅ Image features shape: {image_features.shape}")
    
    # Phi network
    print("\n🧠 Running Phi network...")
    with torch.no_grad():
        predicted_tokens = phi(image_features.to(torch.float32))
    print(f"✅ Pseudo tokens shape: {predicted_tokens.shape}")
    
    # Text encoding
    print("\n📝 Encoding text query...")
    text_query = "a photo of $ that is more blue"
    tokenized = clip.tokenize([text_query], context_length=77, truncate=True).to(device)
    
    with torch.no_grad():
        text_embedding, _, _ = encode_with_pseudo_tokens_HF(
            clip_text_model, 
            tokenized, 
            predicted_tokens
        )
    print(f"✅ Text embedding shape: {text_embedding.shape}")
    
    print("\n" + "=" * 70)
    print("✅ INFERENCE TEST PASSED!")
    print("=" * 70)
    print("\n✅ Your setup is working correctly!")
    print("\nNext steps:")
    print("1. Download dataset images (if needed)")
    print("2. Run: python extract_database_features.py")
    print("3. Build website API")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    try:
        success = test_inference()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
