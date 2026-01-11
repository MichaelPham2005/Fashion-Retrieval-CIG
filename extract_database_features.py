"""
Extract Database Features - Chỉ chạy 1 lần khi setup

Script này extract visual features cho TẤT CẢ ảnh trong database
Output: database_embeddings/{dataset}_database.pt
Sau khi chạy xong có thể XÓA downloaded images để tiết kiệm space
"""

import os
import sys
import torch
import json
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from tqdm.auto import tqdm
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import config

def extract_database_features(args):
    """Extract visual features for all database images"""
    
    print("=" * 70)
    print("Database Feature Extraction")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_path}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Setup paths
    dataset_path = config.get_dataset_path(args.dataset, args.dataset_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load CLIP model
    print("\n📥 Loading CLIP model...")
    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        config.CLIP_MODELS['vit_large'],
        cache_dir=config.HF_CACHE_DIR
    ).to(args.device)
    clip_model.eval()
    
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
    print("✅ CLIP model loaded")
    
    # Database to store embeddings
    database = {}
    url_mapping = {}
    
    # Process each category
    if args.dataset == 'fashioniq':
        categories = args.categories or config.FASHIONIQ_CATEGORIES
        
        for category in categories:
            print(f"\n📂 Processing {category}...")
            
            # Load image URLs
            json_path = os.path.join(dataset_path, 'images', f'{category}.json')
            if not os.path.exists(json_path):
                print(f"⚠️  JSON not found: {json_path}")
                continue
            
            with open(json_path, 'r') as f:
                items = json.load(f)
            
            # Load split to know which images are in test set
            split_path = os.path.join(dataset_path, 'image_splits', f'split.{category}.test.json')
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    test_asins = set(json.load(f))
                print(f"  Found {len(test_asins)} test images in split")
            else:
                # If no split file, use all images
                test_asins = set(item['asin'] for item in items)
                print(f"  No split file, using all {len(test_asins)} images")
            
            # Extract features
            processed = 0
            errors = 0
            
            for item in tqdm(items, desc=f'Extracting {category}'):
                asin = item['asin']
                url = item['url']
                
                # Skip if not in test set
                if asin not in test_asins:
                    continue
                
                try:
                    # Try to load from downloaded folder first
                    img_path = os.path.join(dataset_path, 'images', 'downloaded', category, f'{asin}.jpg')
                    
                    if not os.path.exists(img_path):
                        # Try alternative extension
                        img_path = os.path.join(dataset_path, 'images', 'downloaded', category, f'{asin}.png')
                    
                    if not os.path.exists(img_path):
                        errors += 1
                        continue
                    
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    pixel_values = clip_preprocess(img, return_tensors='pt')['pixel_values']
                    
                    # Extract features
                    with torch.no_grad():
                        features = clip_model(pixel_values.to(args.device)).image_embeds
                    
                    # Store
                    database[asin] = features.cpu().squeeze()
                    url_mapping[asin] = url
                    processed += 1
                    
                except Exception as e:
                    errors += 1
                    if args.verbose:
                        print(f"\n⚠️  Error processing {asin}: {str(e)}")
                    continue
            
            print(f"  ✅ {category}: Processed {processed}, Errors {errors}")
    
    elif args.dataset == 'cirr':
        print(f"\n📂 Processing CIRR...")
        
        # Load CIRR test images
        # Implementation similar to above but for CIRR structure
        print("⚠️  CIRR support not fully implemented yet")
    
    # Save database
    print(f"\n💾 Saving database...")
    save_dict = {
        'embeddings': database,
        'url_mapping': url_mapping,
        'dataset': args.dataset,
        'num_images': len(database)
    }
    torch.save(save_dict, args.output_path)
    
    print("\n" + "=" * 70)
    print("✅ Database feature extraction complete!")
    print(f"Total images: {len(database)}")
    print(f"Output file: {args.output_path}")
    print(f"File size: {os.path.getsize(args.output_path) / (1024**2):.2f} MB")
    print("\n💡 You can now DELETE downloaded images to save space:")
    print(f"   rm -rf {dataset_path}/images/downloaded/")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract database features for website")
    parser.add_argument('--dataset', type=str, default='fashioniq', 
                        choices=['fashioniq', 'cirr'],
                        help="Dataset name")
    parser.add_argument('--dataset_path', type=str, default=None,
                        help="Path to dataset")
    parser.add_argument('--categories', type=str, nargs='+', 
                        default=None,
                        help="Categories to process (for FashionIQ)")
    parser.add_argument('--output_path', type=str, 
                        default='./database_embeddings/fashioniq_database.pt',
                        help="Output path for database embeddings")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use")
    parser.add_argument('--verbose', action='store_true',
                        help="Print detailed errors")
    
    args = parser.parse_args()
    extract_database_features(args)
