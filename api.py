"""
Flask API Backend cho Composed Image Retrieval Website

Endpoints:
- POST /search: Composed image search
- GET /health: Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import clip
import io
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor
from phi import Phi
import config

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for models (load once when server starts)
clip_vision_model = None
clip_text_model = None
phi_model = None
clip_preprocess = None
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
    return x

def load_models():
    """Load all models when server starts"""
    global clip_vision_model, clip_text_model, phi_model, clip_preprocess
    global database_embeddings, url_mapping, device
    
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
        'database_size': len(database_embeddings) if database_embeddings else 0
    })


@app.route('/search', methods=['POST'])
def composed_search():
    """
    Composed image search endpoint
    
    Request:
        - image: Reference image file
        - query: Text modification (e.g., "more red color, without sleeves")
        - top_k: Number of results to return (default: 20)
    
    Response:
        - results: List of {url, score, asin}
    """
    try:
        # Check if models are loaded
        if database_embeddings is None or len(database_embeddings) == 0:
            return jsonify({
                'error': 'Database not loaded. Run extract_database_features.py first'
            }), 500
        
        # Get inputs
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        text_query = request.form.get('query', '')
        top_k = int(request.form.get('top_k', 20))
        
        if not text_query:
            return jsonify({'error': 'No text query provided'}), 400
        
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        pixel_values = clip_preprocess(img, return_tensors='pt')['pixel_values'].to(device)
        
        # Extract image features
        with torch.no_grad():
            image_features = clip_vision_model(pixel_values).image_embeds
        
        # Phi network
        with torch.no_grad():
            predicted_tokens = phi_model(image_features.to(torch.float32))
        
        # Encode text with pseudo tokens
        text_input = f"a photo of $ that {text_query}"
        tokenized = clip.tokenize([text_input], context_length=77, truncate=True).to(device)
        
        with torch.no_grad():
            composed_embedding = encode_with_pseudo_tokens_HF(
                clip_text_model,
                tokenized,
                predicted_tokens
            )
        
        # Search in database
        composed_embedding = composed_embedding.squeeze()
        similarities = {}
        
        for asin, db_embedding in database_embeddings.items():
            db_embedding = db_embedding.to(device)
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                composed_embedding.unsqueeze(0),
                db_embedding.unsqueeze(0)
            ).item()
            similarities[asin] = similarity
        
        # Get top-k results
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = [
            {
                'asin': asin,
                'url': url_mapping.get(asin, ''),
                'score': float(score)
            }
            for asin, score in sorted_results
        ]
        
        return jsonify({
            'success': True,
            'query': text_query,
            'num_results': len(results),
            'results': results
        })
    
    except Exception as e:
        import traceback
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
        'models_loaded': all([
            clip_vision_model is not None,
            clip_text_model is not None,
            phi_model is not None
        ])
    })


if __name__ == '__main__':
    # Load models before starting server
    load_models()
    
    # Start server
    print("\n" + "=" * 70)
    print("🌐 Starting Flask API server...")
    print("=" * 70)
    print("Endpoints:")
    print("  - GET  /health  : Health check")
    print("  - POST /search  : Composed image search")
    print("  - GET  /stats   : Database statistics")
    print("\nServer will run on: http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
