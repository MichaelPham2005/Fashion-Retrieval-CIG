import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
import json
from PIL import Image
from torchvision import transforms
import random
from torchvision.transforms.functional import crop
import numpy as np
from PIL import ImageFile
import requests
from io import BytesIO
import warnings


Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image_from_path_or_url(image_path, timeout=10, max_retries=2):
    """
    Load image from local path or URL
    
    Args:
        image_path: Local file path or HTTP URL
        timeout: Timeout for URL requests
        max_retries: Number of retries for URL downloads
    
    Returns:
        PIL Image in RGB format
    """
    # Check if it's a local file
    if os.path.exists(image_path):
        return Image.open(image_path).convert('RGB')
    
    # Check if it's a URL
    if image_path.startswith('http://') or image_path.startswith('https://'):
        for attempt in range(max_retries):
            try:
                response = requests.get(image_path, timeout=timeout, stream=True)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')
                return img
            except Exception as e:
                if attempt == max_retries - 1:
                    warnings.warn(f"Failed to download image from {image_path}: {str(e)}")
                    raise
    
    # If neither local nor URL, raise error
    raise FileNotFoundError(f"Image not found: {image_path}")


class CIRRDataset(Dataset):
    def __init__(self, dataset_path, split='test', preprocess=None):
        self.dataset_path = dataset_path
        self.split = split
        self.preprocess = preprocess
        # Load dataset metadata
        caption_path = os.path.join(dataset_path, f'captions/cap.rc2.{split}.json')
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Caption file not found: {caption_path}")
        
        with open(caption_path, 'r') as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        reference_image_path = os.path.join(self.dataset_path, self.split, item['reference']+'.png')
        
        # Load image
        reference_image = load_image_from_path_or_url(reference_image_path)
        
        if self.preprocess:
            reference_image = self.preprocess(reference_image)['pixel_values'][0]

        return {
            'reference_image': reference_image,
            'relative_caption': item['caption'],
            'pairid': item['pairid']
        }


class FashionIQDataset(Dataset):
    def __init__(self, dataset_path, split='test', dress_types=['dress', 'shirt', 'toptee'], 
                 preprocess=None, use_downloaded_images=True):
        """
        FashionIQ Dataset Loader
        
        Args:
            dataset_path: Root path to FashionIQ dataset
            split: 'train', 'val', or 'test'
            dress_types: List of categories to load
            preprocess: Preprocessing function for images
            use_downloaded_images: If True, load from downloaded/ folder, else try URLs
        """
        self.dataset_path = dataset_path
        self.split = split
        self.dress_types = dress_types
        self.preprocess = preprocess
        self.use_downloaded_images = use_downloaded_images
        
        # Load ASIN to URL mapping
        self.asin_to_url = {}
        for dress_type in dress_types:
            json_path = os.path.join(dataset_path, 'images', f'{dress_type}.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        self.asin_to_url[item['asin']] = item['url']
        
        # Load dataset metadata
        self.metadata = []
        for dress_type in dress_types:
            caption_path = os.path.join(dataset_path, 'captions', f'cap.{dress_type}.{split}.json')
            if not os.path.exists(caption_path):
                warnings.warn(f"Caption file not found: {caption_path}, skipping {dress_type}")
                continue
            
            with open(caption_path, 'r') as f:
                captions = json.load(f)
                for item in captions:
                    # Combine two captions into one
                    combined_caption = '. '.join(item['captions'])
                    self.metadata.append({
                        'candidate': item['candidate'],
                        'target': item['target'],
                        'relative_caption': combined_caption,
                        'dress_type': dress_type
                    })
    
    def _get_image_path(self, asin, dress_type):
        """Get image path (local or URL) for an ASIN"""
        if self.use_downloaded_images:
            # Try local downloaded images first
            local_path = os.path.join(self.dataset_path, 'images', 'downloaded', dress_type, f'{asin}.jpg')
            if os.path.exists(local_path):
                return local_path
        
        # Fall back to URL if available
        if asin in self.asin_to_url:
            return self.asin_to_url[asin]
        
        # If neither available, raise error
        raise FileNotFoundError(f"Image not found for ASIN {asin} in {dress_type}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        dress_type = item['dress_type']
        candidate_asin = item['candidate']
        
        # Get reference image path
        reference_image_path = self._get_image_path(candidate_asin, dress_type)
        
        # Load image
        reference_image = load_image_from_path_or_url(reference_image_path)
        
        if self.preprocess:
            reference_image = self.preprocess(reference_image)['pixel_values'][0]

        return {
            'reference_image': reference_image,
            'relative_caption': item['relative_caption'],
            'candidate': item['candidate'],
            'target': item['target'],
            'dress_type': dress_type
        }


class ComposedEmbedsDataset(Dataset):
    """
    Dataset that loads saved composed embeddings(.pt) .
    Expects files saved as {pairid}.pt containing keys:
      - 'conditioning'  (Tensor [seq_len, d1] or [1, seq_len, d1])
      - 'conditioning2' (Tensor [seq_len, d2] or [1, seq_len, d2])
      - 'pooled2'       (Tensor [d2] or [1, d2])

    Returns for each item:
      {
        'pairid': str,
        'prompt_embeds': Tensor [seq_len, d1+d2],
        'pooled2': Tensor [d2]
      }
    """

    def __init__(self, dataset_path: str, text_embeddings_dir: str, split: str = 'val'):
        self.dataset_path = dataset_path
        self.text_embeddings_dir = text_embeddings_dir
        self.split = split

        # Load caption metadata to get pairids
        cap_path = os.path.join(dataset_path, f'captions/cap.rc2.{split}.json')
        with open(cap_path, 'r') as f:
            metadata = json.load(f)

        # Keep only entries with an existing .pt embedding file
        self.items = []
        for item in metadata:
            pairid = str(item['pairid'])
            pt_path = os.path.join(text_embeddings_dir, f"{pairid}.pt")
            if os.path.exists(pt_path):
                self.items.append({'pairid': pairid, 'pt_path': pt_path})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entry = self.items[idx]
        pairid = entry['pairid']
        pt_path = entry['pt_path']

        save_dict = torch.load(pt_path, map_location='cpu')

        cond1 = save_dict['conditioning']  # [seq_len, d1] or [1, seq_len, d1]
        cond2 = save_dict['conditioning2'] # [seq_len, d2] or [1, seq_len, d2]
        pooled2 = save_dict['pooled2']     # [d2] or [1, d2]

        # Normalize shapes to [seq_len, d]
        if cond1.dim() == 3:
            cond1 = cond1.squeeze(0)
        if cond2.dim() == 3:
            cond2 = cond2.squeeze(0)
        if pooled2.dim() == 2:
            pooled2 = pooled2.squeeze(0)

        prompt_embeds = torch.concat([cond1, cond2], dim=-1)

        return {
            'pairid': pairid,
            'prompt_embeds': prompt_embeds,  # CPU tensor; caller can .to('cuda')
            'pooled2': pooled2,
        }