"""
Download Images from URLs for FashionIQ Dataset

This script downloads images from URLs stored in JSON files and saves them locally.
Supports retry mechanism and error handling for failed downloads.
"""

import os
import json
import argparse
import requests
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def download_image(url, save_path, max_retries=3, timeout=10):
    """
    Download a single image from URL and save to local path
    
    Args:
        url: Image URL
        save_path: Local path to save image
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        (success: bool, message: str)
    """
    for attempt in range(max_retries):
        try:
            # Send HTTP request
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Open image with PIL to validate
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')  # Ensure RGB format
            
            # Save image
            img.save(save_path, 'JPEG', quality=95)
            return True, "Success"
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return False, f"RequestError: {str(e)}"
            time.sleep(1)  # Wait before retry
            
        except Exception as e:
            return False, f"ImageError: {str(e)}"
    
    return False, "Max retries exceeded"


def download_category(category, json_path, output_dir, max_workers=10, skip_existing=True):
    """
    Download all images for a specific category
    
    Args:
        category: Category name (e.g., 'dress', 'shirt', 'toptee')
        json_path: Path to JSON file containing URLs
        output_dir: Directory to save downloaded images
        max_workers: Number of parallel download threads
        skip_existing: Skip already downloaded images
    
    Returns:
        Statistics dictionary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    print(f"\n📂 Loading {category} data from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Found {len(data)} images in {category}")
    
    # Prepare download tasks
    tasks = []
    for item in data:
        asin = item['asin']
        url = item['url']
        save_path = os.path.join(output_dir, f"{asin}.jpg")
        
        # Skip if already exists
        if skip_existing and os.path.exists(save_path):
            continue
        
        tasks.append((asin, url, save_path))
    
    if len(tasks) == 0:
        print(f"✅ All {category} images already downloaded!")
        return {
            'total': len(data),
            'success': len(data),
            'failed': 0,
            'skipped': len(data)
        }
    
    print(f"📥 Downloading {len(tasks)} images for {category}...")
    
    # Statistics
    success_count = 0
    failed_count = 0
    failed_items = []
    
    # Download with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_image, url, save_path): (asin, url, save_path)
            for asin, url, save_path in tasks
        }
        
        with tqdm(total=len(futures), desc=f"Downloading {category}", unit="img") as pbar:
            for future in as_completed(futures):
                asin, url, save_path = futures[future]
                try:
                    success, message = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_items.append({
                            'asin': asin,
                            'url': url,
                            'error': message
                        })
                except Exception as e:
                    failed_count += 1
                    failed_items.append({
                        'asin': asin,
                        'url': url,
                        'error': str(e)
                    })
                
                pbar.update(1)
    
    # Save failed items log
    if failed_items:
        log_path = os.path.join(output_dir, f"failed_downloads_{category}.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(failed_items, f, indent=2, ensure_ascii=False)
        print(f"⚠️  Failed downloads logged to: {log_path}")
    
    stats = {
        'total': len(data),
        'success': success_count,
        'failed': failed_count,
        'skipped': len(data) - len(tasks)
    }
    
    print(f"✅ {category.upper()} Complete:")
    print(f"   - Total: {stats['total']}")
    print(f"   - Success: {stats['success']}")
    print(f"   - Failed: {stats['failed']}")
    print(f"   - Skipped: {stats['skipped']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download FashionIQ images from URLs")
    parser.add_argument('--dataset', type=str, default='fashioniq', choices=['fashioniq'],
                        help="Dataset name (currently only 'fashioniq' supported)")
    parser.add_argument('--categories', type=str, nargs='+', default=['dress', 'shirt', 'toptee'],
                        help="Categories to download (e.g., dress shirt toptee)")
    parser.add_argument('--json_dir', type=str, default='./datasets/FashionIQ/images',
                        help="Directory containing JSON files with URLs")
    parser.add_argument('--output_dir', type=str, default='./datasets/FashionIQ/images/downloaded',
                        help="Directory to save downloaded images")
    parser.add_argument('--max_workers', type=int, default=10,
                        help="Number of parallel download threads")
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help="Skip already downloaded images")
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                        help="Re-download all images")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📥 FashionIQ Image Downloader")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Categories: {', '.join(args.categories)}")
    print(f"JSON Directory: {args.json_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Skip Existing: {args.skip_existing}")
    print("=" * 70)
    
    # Overall statistics
    total_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # Download each category
    for category in args.categories:
        json_path = os.path.join(args.json_dir, f"{category}.json")
        category_output_dir = os.path.join(args.output_dir, category)
        
        if not os.path.exists(json_path):
            print(f"⚠️  JSON file not found: {json_path}")
            print(f"   Skipping {category}...")
            continue
        
        stats = download_category(
            category=category,
            json_path=json_path,
            output_dir=category_output_dir,
            max_workers=args.max_workers,
            skip_existing=args.skip_existing
        )
        
        # Accumulate statistics
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Print overall statistics
    print("\n" + "=" * 70)
    print("📊 OVERALL STATISTICS")
    print("=" * 70)
    print(f"Total Images: {total_stats['total']}")
    print(f"Successfully Downloaded: {total_stats['success']}")
    print(f"Failed: {total_stats['failed']}")
    print(f"Skipped (Already Exist): {total_stats['skipped']}")
    print("=" * 70)
    
    if total_stats['failed'] > 0:
        print(f"\n⚠️  {total_stats['failed']} images failed to download.")
        print("   Check the failed_downloads_*.json files for details.")
        print("   You can retry by running the script again.")
    else:
        print("\n✅ All images downloaded successfully!")


if __name__ == "__main__":
    main()
