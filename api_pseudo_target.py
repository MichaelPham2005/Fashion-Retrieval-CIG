"""
Flask API Backend cho Composed Image Retrieval Website
Sử dụng Pseudo-Target Generation approach (đúng với model CIG)

Workflow:
1. User upload reference image + text query
2. Extract image features với CLIP Vision
3. Phi network predicts pseudo tokens
4. Generate composed embeddings
5. SDXL generates pseudo-target image
6. Extract features từ pseudo-target image
7. Search database với pseudo-target features
8. Return top-K results + pseudo-target image

Endpoints:
- POST /search: Composed image search với pseudo-target generation
- GET /health: Health check
- GET /stats: Database statistics
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import torch
import clip
import io
import os
import sys
import base64
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from phi import Phi
import config

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for models (load once when server starts)
clip_vision_model = None
clip_text_model = None
phi_model = None
clip_preprocess = None
sdxl_pipe = None
database_embeddings = None
url_mapping = None
device = None

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

def encode_with_pseudo_tokens_HF(clip_model, text, pseudo_tokens):
    """Encode text with pseudo tokens"""
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
        x_out = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output.last_hidden_state
    except TypeError:
        # Newer versions don't accept return_dict
        encoder_output = clip_model.text_model.encoder(
            inputs_embeds=x,
            attention_mask=None,
            causal_attention_mask=_causal_attention_mask
        )
        x_out = encoder_output.last_hidden_state if hasattr(encoder_output, 'last_hidden_state') else encoder_output
    
    x_last = clip_model.text_model.final_layer_norm(x_out)
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device), text.to(dtype=torch.int, device=x_last.device).argmax(dim=-1)]
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)
    return x, x_out

def load_models():
    """Load all models when server starts"""
    global clip_vision_model, clip_text_model, phi_model, clip_preprocess
    global sdxl_pipe, database_embeddings, url_mapping, device
    
    print("🚀 Starting server and loading models...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load CLIP models
    print("📥 Loading CLIP models...")
    clip_model_name = config.CLIP_MODELS['vit_large']
    clip_text_model = CLIPTextModelWithProjection.from_pretrained(
        clip_model_name,
        torch_dtype=torch.float32,
        cache_dir=config.HF_CACHE_DIR
    ).float().to(device)
    clip_text_model.eval()
    
    clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
        clip_model_name,
        torch_dtype=torch.float32,
        cache_dir=config.HF_CACHE_DIR
    ).float().to(device)
    clip_vision_model.eval()
    
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
    print("📥 Loading Phi model...")
    phi_vit_path = config.get_model_path('phi_vit')
    phi_model = Phi(input_dim=768, hidden_dim=768 * 4, output_dim=768, dropout=0)
    phi_model.load_state_dict(torch.load(phi_vit_path, map_location=device)[phi_model.__class__.__name__])
    phi_model = phi_model.to(device=device).eval()
    
    # Load SDXL pipeline
    print("📥 Loading SDXL pipeline (this may take a while)...")
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    # Load UNet from checkpoint
    model_path = config.MODEL_PATHS['sdxl_checkpoint']
    unet_path = os.path.join(model_path, 'checkpoint-20000', 'unet')
    
    if not os.path.exists(unet_path):
        print(f"⚠️  UNet not found at {unet_path}")
        print(f"   Please ensure models/checkpoint-20000-SDXL/checkpoint-20000/unet/ exists")
        sdxl_pipe = None
    else:
        unet = UNet2DConditionModel.from_pretrained(
            unet_path, 
            use_safetensors=True, 
            torch_dtype=dtype
        ).to(device)
        
        vae = AutoencoderKL.from_pretrained(
            config.SDXL_MODELS['vae'],
            torch_dtype=dtype,
            cache_dir=config.HF_CACHE_DIR
        ).to(device)
        
        sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            config.SDXL_MODELS['base'],
            unet=unet,
            vae=vae,
            torch_dtype=dtype,
            cache_dir=config.HF_CACHE_DIR
        )
        sdxl_pipe.to(device)
        print("✅ SDXL pipeline loaded")
    
    # Load database embeddings
    print("📥 Loading database embeddings...")
    db_path = './database_embeddings/fashioniq_database.pt'
    if os.path.exists(db_path):
        db_data = torch.load(db_path, map_location=device)
        database_embeddings = db_data['embeddings']
        url_mapping = db_data['url_mapping']
        print(f"✅ Loaded {len(database_embeddings)} database embeddings")
    else:
        print(f"⚠️  Database not found at {db_path}")
        print("   Run: python extract_database_features.py")
        database_embeddings = {}
        url_mapping = {}
    
    print("✅ All models loaded successfully!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'database_size': len(database_embeddings) if database_embeddings else 0,
        'sdxl_loaded': sdxl_pipe is not None,
        'approach': 'Pseudo-Target Generation'
    })


@app.route('/search', methods=['POST'])
def composed_search():
    """
    Composed image search endpoint with Pseudo-Target Generation
    
    Request:
        - image: Reference image file
        - query: Text modification (e.g., "more red color, without sleeves")
        - top_k: Number of results to return (default: 20)
        - height: Generated image height (default: 512)
        - width: Generated image width (default: 512)
        - steps: Inference steps (default: 50)
        - seed: Random seed (default: 42)
        - brightness_thresh: Minimum brightness (default: 50)
        - max_retries: Max retries if too dark (default: 3)
    
    Response:
        - results: List of {url, score, asin}
        - pseudo_target_image: Base64 encoded pseudo-target image
        - generation_time: Time to generate pseudo-target
        - search_time: Time to search database
    """
    try:
        # Check if models are loaded
        if database_embeddings is None or len(database_embeddings) == 0:
            return jsonify({
                'error': 'Database not loaded. Run extract_database_features.py first'
            }), 500
        
        if sdxl_pipe is None:
            return jsonify({
                'error': 'SDXL pipeline not loaded. Check model paths.'
            }), 500
        
        # Get inputs
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        text_query = request.form.get('query', '')
        top_k = int(request.form.get('top_k', 20))
        height = int(request.form.get('height', 512))
        width = int(request.form.get('width', 512))
        steps = int(request.form.get('steps', 50))
        seed = int(request.form.get('seed', 42))
        brightness_thresh = float(request.form.get('brightness_thresh', 50))
        max_retries = int(request.form.get('max_retries', 3))
        
        if not text_query:
            return jsonify({'error': 'No text query provided'}), 400
        
        print(f"\n{'='*50}")
        print(f"🔍 New search request")
        print(f"Query: {text_query}")
        print(f"{'='*50}")
        
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        pixel_values = clip_preprocess(img, return_tensors='pt')['pixel_values'].to(device)
        
        # Step 1: Extract image features
        print("1️⃣ Extracting reference image features...")
        with torch.no_grad():
            image_features = clip_vision_model(pixel_values).image_embeds
        
        # Step 2: Phi network
        print("2️⃣ Running Phi network...")
        with torch.no_grad():
            predicted_tokens = phi_model(image_features.to(torch.float32))
        
        # Step 3: Encode text with pseudo tokens
        print("3️⃣ Encoding text with pseudo tokens...")
        text_input = f"a photo of $ that {text_query}"
        tokenized = clip.tokenize([text_input], context_length=77, truncate=True).to(device)
        
        with torch.no_grad():
            composed_embedding, composed_last_hidden = encode_with_pseudo_tokens_HF(
                clip_text_model,
                tokenized,
                predicted_tokens
            )
        
        # Step 4: Generate pseudo-target image with SDXL
        print("4️⃣ Generating pseudo-target image with SDXL...")
        generation_start = datetime.now()
        
        # Prepare embeddings for SDXL
        prompt_embeds = composed_last_hidden.to(dtype=sdxl_pipe.unet.dtype)
        pooled_prompt_embeds = composed_embedding.to(dtype=sdxl_pipe.unet.dtype)
        
        # Generate with retry mechanism
        pseudo_target_image = None
        attempts = 0
        
        while attempts <= max_retries:
            current_seed = seed + attempts
            generator = torch.Generator(device=device).manual_seed(current_seed)
            
            images = sdxl_pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                height=height,
                width=width,
                num_inference_steps=steps,
                generator=generator
            ).images
            
            pseudo_target_image = images[0]
            brightness = np.asarray(pseudo_target_image).mean()
            
            if brightness > brightness_thresh:
                break
            
            print(f"   Retry {attempts+1}/{max_retries} (brightness: {brightness:.1f})")
            attempts += 1
        
        generation_time = (datetime.now() - generation_start).total_seconds()
        print(f"   ✅ Generated in {generation_time:.2f}s")
        
        # Step 5: Extract features from pseudo-target image
        print("5️⃣ Extracting features from pseudo-target image...")
        pseudo_target_pil = pseudo_target_image.resize((224, 224))
        pseudo_pixel_values = clip_preprocess(pseudo_target_pil, return_tensors='pt')['pixel_values'].to(device)
        
        with torch.no_grad():
            pseudo_target_features = clip_vision_model(pseudo_pixel_values).image_embeds
        
        # Step 6: Search in database
        print("6️⃣ Searching database...")
        search_start = datetime.now()
        
        pseudo_target_features = pseudo_target_features.squeeze()
        similarities = {}
        
        for asin, db_embedding in database_embeddings.items():
            db_embedding = db_embedding.to(device)
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                pseudo_target_features.unsqueeze(0),
                db_embedding.unsqueeze(0)
            ).item()
            similarities[asin] = similarity
        
        # Get top-k results
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        search_time = (datetime.now() - search_start).total_seconds()
        print(f"   ✅ Searched in {search_time:.2f}s")
        
        results = [
            {
                'asin': asin,
                'url': url_mapping.get(asin, ''),
                'score': float(score)
            }
            for asin, score in sorted_results
        ]
        
        # Convert pseudo-target image to base64
        buffered = io.BytesIO()
        pseudo_target_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        print(f"✅ Search complete! Found {len(results)} results")
        print(f"{'='*50}\n")
        
        return jsonify({
            'success': True,
            'query': text_query,
            'num_results': len(results),
            'results': results,
            'pseudo_target_image': f'data:image/png;base64,{img_str}',
            'generation_time': generation_time,
            'search_time': search_time,
            'total_time': generation_time + search_time,
            'brightness': float(brightness),
            'attempts': attempts + 1
        })
    
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    return jsonify({
        'database_size': len(database_embeddings) if database_embeddings else 0,
        'device': str(device),
        'approach': 'Pseudo-Target Generation',
        'models_loaded': {
            'clip_vision': clip_vision_model is not None,
            'clip_text': clip_text_model is not None,
            'phi': phi_model is not None,
            'sdxl': sdxl_pipe is not None,
            'database': database_embeddings is not None
        }
    })


if __name__ == '__main__':
    # Load models before starting server
    load_models()
    
    # Start server
    print("\n" + "=" * 70)
    print("🌐 Starting Flask API server...")
    print("=" * 70)
    print("Approach: Pseudo-Target Generation (CIG Model)")
    print("\nEndpoints:")
    print("  - GET  /health  : Health check")
    print("  - POST /search  : Composed image search with pseudo-target generation")
    print("  - GET  /stats   : Database statistics")
    print("\nServer will run on: http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
