"""
Configuration file for ComposedImageGen project
Centralized management of paths, models, and settings
"""

import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASET_PATHS = {
    'cirr': os.path.join(PROJECT_ROOT, 'datasets', 'CIRR'),
    'fashioniq': os.path.join(PROJECT_ROOT, 'datasets', 'FashionIQ'),
}

# Model paths
MODEL_PATHS = {
    'phi_vit': os.path.join(PROJECT_ROOT, 'models', 'phi_best.pt'),
    'phi_giga': os.path.join(PROJECT_ROOT, 'models', 'phi_best_giga.pt'),
    'sdxl_checkpoint': os.path.join(PROJECT_ROOT, 'models', 'checkpoint-20000-SDXL'),
}

# Output paths
OUTPUT_PATHS = {
    'embeddings': os.path.join(PROJECT_ROOT, 'outputs', 'embeddings'),
    'generated_images': os.path.join(PROJECT_ROOT, 'outputs', 'generated_images'),
}

# CLIP model names
CLIP_MODELS = {
    'vit_large': 'openai/clip-vit-large-patch14',
    'giga': 'Geonmo/CLIP-Giga-config-fixed',
}

# HuggingFace cache directory
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')

# SDXL models
SDXL_MODELS = {
    'vae': 'madebyollin/sdxl-vae-fp16-fix',
    'base': 'stabilityai/stable-diffusion-xl-base-1.0',
}

# Image preprocessing settings
IMAGE_PREPROCESS = {
    'crop_size': {'height': 224, 'width': 224},
    'size': {'shortest_edge': 224},
    'image_mean': [0.48145466, 0.4578275, 0.40821073],
    'image_std': [0.26862954, 0.26130258, 0.27577711],
}

# FashionIQ settings
FASHIONIQ_CATEGORIES = ['dress', 'shirt', 'toptee']
FASHIONIQ_SPLITS = ['train', 'val', 'test']

# CIRR settings
CIRR_SPLITS = ['train', 'val', 'test1', 'test2']

# Training/Inference settings
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = 'cuda'

# Image generation settings
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_INFERENCE_STEPS = 50
DEFAULT_SEED = 1600
DEFAULT_BRIGHTNESS_THRESH = 10.0
DEFAULT_MAX_RETRIES = 5


def get_dataset_path(dataset_name, custom_path=None):
    """
    Get dataset path with validation
    
    Args:
        dataset_name: 'cirr' or 'fashioniq'
        custom_path: Custom path to override default
    
    Returns:
        Validated dataset path
    """
    if custom_path:
        if not os.path.exists(custom_path):
            print(f"Warning: Custom dataset path does not exist: {custom_path}")
        return custom_path
    
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_PATHS.keys())}")
    
    path = DATASET_PATHS[dataset_name]
    if not os.path.exists(path):
        print(f"Warning: Dataset path does not exist: {path}")
        print(f"Please download the {dataset_name} dataset and place it in the correct location.")
    
    return path


def get_model_path(model_name, custom_path=None):
    """
    Get model path with validation
    
    Args:
        model_name: Model identifier (e.g., 'phi_vit', 'phi_giga')
        custom_path: Custom path to override default
    
    Returns:
        Validated model path
    """
    if custom_path:
        if not os.path.exists(custom_path):
            print(f"Warning: Custom model path does not exist: {custom_path}")
        return custom_path
    
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_PATHS.keys())}")
    
    path = MODEL_PATHS[model_name]
    if not os.path.exists(path):
        print(f"Warning: Model path does not exist: {path}")
        print(f"Please download the {model_name} model and place it in the correct location.")
        print(f"Download link: https://drive.google.com/drive/folders/1hpIpI0X26ox-uY-QdOPKDKKZlnWkftIA")
    
    return path


def get_output_path(output_type, custom_path=None, create=True):
    """
    Get output path and optionally create directory
    
    Args:
        output_type: 'embeddings' or 'generated_images'
        custom_path: Custom path to override default
        create: Create directory if it doesn't exist
    
    Returns:
        Output path
    """
    if custom_path:
        path = custom_path
    elif output_type in OUTPUT_PATHS:
        path = OUTPUT_PATHS[output_type]
    else:
        raise ValueError(f"Unknown output type: {output_type}. Available: {list(OUTPUT_PATHS.keys())}")
    
    if create:
        os.makedirs(path, exist_ok=True)
    
    return path


def validate_environment():
    """
    Validate that all required paths and files exist
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'datasets': {},
        'models': {},
        'warnings': []
    }
    
    # Check datasets
    for name, path in DATASET_PATHS.items():
        exists = os.path.exists(path)
        results['datasets'][name] = exists
        if not exists:
            results['warnings'].append(f"Dataset not found: {name} at {path}")
    
    # Check models
    for name, path in MODEL_PATHS.items():
        exists = os.path.exists(path)
        results['models'][name] = exists
        if not exists:
            results['warnings'].append(f"Model not found: {name} at {path}")
    
    return results


if __name__ == "__main__":
    # Print configuration when run directly
    print("=" * 70)
    print("ComposedImageGen Configuration")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    
    print("\n📂 Dataset Paths:")
    for name, path in DATASET_PATHS.items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {name}: {path}")
    
    print("\n🤖 Model Paths:")
    for name, path in MODEL_PATHS.items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {name}: {path}")
    
    print("\n📤 Output Paths:")
    for name, path in OUTPUT_PATHS.items():
        exists = "✅" if os.path.exists(path) else "📁"
        print(f"  {exists} {name}: {path}")
    
    print("\n🔍 Validation Results:")
    validation = validate_environment()
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    else:
        print("  ✅ All paths validated successfully!")
    
    print("=" * 70)
