"""
Test Script cho Pseudo-Target Generation Workflow

Script này test toàn bộ pipeline:
1. Check models có load được không
2. Test SDXL generation
3. Test database search
4. Verify kết quả

Usage:
    python test_pseudo_target_workflow.py
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config

def test_models_exist():
    """Test 1: Kiểm tra models có tồn tại không"""
    print("\n" + "="*70)
    print("TEST 1: Checking Model Files")
    print("="*70)
    
    checks = {
        'Phi ViT': config.MODEL_PATHS['phi_vit'],
        'Phi Giga': config.MODEL_PATHS['phi_giga'],
        'SDXL UNet': os.path.join(config.MODEL_PATHS['sdxl_checkpoint'], 'checkpoint-20000', 'unet'),
    }
    
    all_exist = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ All models found!")
    else:
        print("\n❌ Some models missing. Please check paths.")
    
    return all_exist


def test_load_models():
    """Test 2: Test load models vào memory"""
    print("\n" + "="*70)
    print("TEST 2: Loading Models into Memory")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    try:
        # Load CLIP
        print("📥 Loading CLIP Vision...")
        from transformers import CLIPVisionModelWithProjection
        clip_vision = CLIPVisionModelWithProjection.from_pretrained(
            config.CLIP_MODELS['vit_large'],
            cache_dir=config.HF_CACHE_DIR
        ).to(device)
        print("   ✅ CLIP Vision loaded")
        
        # Load Phi
        print("📥 Loading Phi...")
        from phi import Phi
        phi = Phi(input_dim=768, hidden_dim=768 * 4, output_dim=768, dropout=0)
        phi.load_state_dict(torch.load(config.get_model_path('phi_vit'), map_location=device)['Phi'])
        phi = phi.to(device).eval()
        print("   ✅ Phi loaded")
        
        # Load SDXL
        print("📥 Loading SDXL (this may take a while)...")
        from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionXLPipeline
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(config.MODEL_PATHS['sdxl_checkpoint'], 'checkpoint-20000', 'unet'),
            use_safetensors=True,
            torch_dtype=dtype
        ).to(device)
        
        vae = AutoencoderKL.from_pretrained(
            config.SDXL_MODELS['vae'],
            torch_dtype=dtype,
            cache_dir=config.HF_CACHE_DIR
        ).to(device)
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            config.SDXL_MODELS['base'],
            unet=unet,
            vae=vae,
            torch_dtype=dtype,
            cache_dir=config.HF_CACHE_DIR
        ).to(device)
        print("   ✅ SDXL loaded")
        
        print("\n✅ All models loaded successfully!")
        return True, {'clip_vision': clip_vision, 'phi': phi, 'sdxl': pipe, 'device': device}
        
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_sdxl_generation(models):
    """Test 3: Test SDXL generation"""
    print("\n" + "="*70)
    print("TEST 3: Testing SDXL Generation")
    print("="*70)
    
    try:
        pipe = models['sdxl']
        device = models['device']
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        # Create dummy embeddings
        print("Creating dummy embeddings...")
        prompt_embeds = torch.randn(1, 77, 2048).to(device, dtype=dtype)
        pooled_embeds = torch.randn(1, 1280).to(device, dtype=dtype)
        
        # Generate
        print("Generating image (this may take 10-30s)...")
        generator = torch.Generator(device=device).manual_seed(42)
        
        images = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            height=512,
            width=512,
            num_inference_steps=20,  # Fewer steps for testing
            generator=generator
        ).images
        
        # Check result
        img = images[0]
        brightness = np.asarray(img).mean()
        
        print(f"✅ Image generated!")
        print(f"   Size: {img.size}")
        print(f"   Brightness: {brightness:.1f}")
        
        # Save test image
        test_output = './test_sdxl_output.png'
        img.save(test_output)
        print(f"   Saved to: {test_output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """Test 4: Test database embeddings"""
    print("\n" + "="*70)
    print("TEST 4: Testing Database Embeddings")
    print("="*70)
    
    db_path = './database_embeddings/fashioniq_database.pt'
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("   Run: python extract_database_features.py")
        return False
    
    try:
        print(f"Loading database from {db_path}...")
        db_data = torch.load(db_path, map_location='cpu')
        
        embeddings = db_data['embeddings']
        url_mapping = db_data['url_mapping']
        
        print(f"✅ Database loaded!")
        print(f"   Number of embeddings: {len(embeddings)}")
        print(f"   Number of URLs: {len(url_mapping)}")
        
        # Check embedding shape
        first_asin = list(embeddings.keys())[0]
        first_embed = embeddings[first_asin]
        print(f"   Embedding shape: {first_embed.shape}")
        print(f"   Sample ASIN: {first_asin}")
        print(f"   Sample URL: {url_mapping[first_asin][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_server():
    """Test 5: Test API server connection"""
    print("\n" + "="*70)
    print("TEST 5: Testing API Server")
    print("="*70)
    
    try:
        import requests
        
        print("Checking API health...")
        response = requests.get('http://localhost:5000/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API server is running!")
            print(f"   Status: {data.get('status')}")
            print(f"   Approach: {data.get('approach')}")
            print(f"   Database size: {data.get('database_size')}")
            print(f"   SDXL loaded: {data.get('sdxl_loaded')}")
            
            if data.get('approach') != 'Pseudo-Target Generation':
                print("⚠️  Warning: API is not using Pseudo-Target Generation!")
                print("   Make sure to run: python api_pseudo_target.py")
            
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Start server with: python api_pseudo_target.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 CIG Pseudo-Target Generation - Test Suite")
    print("="*70)
    
    results = {}
    
    # Test 1: Check files
    results['files'] = test_models_exist()
    
    if not results['files']:
        print("\n⚠️ Stopping tests - models not found")
        return
    
    # Test 2: Load models
    results['load'], models = test_load_models()
    
    if not results['load']:
        print("\n⚠️ Stopping tests - cannot load models")
        return
    
    # Test 3: SDXL generation
    results['sdxl'] = test_sdxl_generation(models)
    
    # Test 4: Database
    results['database'] = test_database()
    
    # Test 5: API server
    results['api'] = test_api_server()
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.upper()}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\n✅ You are ready to use the system!")
        print("\nNext steps:")
        print("1. Make sure API server is running: python api_pseudo_target.py")
        print("2. Open demo_website_pseudo_target.html in browser")
        print("3. Upload an image and try a query!")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    print("="*70)


if __name__ == '__main__':
    main()
