import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List, Tuple, Optional


def load_sample(
    sample_idx: int = 0,
    output_dir: str = './output_data',
    device: str = 'cpu',
) -> Dict:
    """
    Load a single sample from the preprocessed dataset.
    
    Args:
        sample_idx: index of sample to load
        output_dir: directory containing preprocessed data
        device: device to load tensors on
    
    Returns:
        sample_dict: dictionary containing:
            - image: PIL Image
            - image_id: str
            - global_caption: str
            - width: int
            - height: int
            - objects: list of dicts with:
                - object_id: int
                - category_name: str
                - bbox: [x1, y1, x2, y2] normalized
                - depth_label: str (foreground/midground/background)
                - caption: str
                - clip_embedding: (768,) tensor
                - dino_embedding: (1024,) tensor
    """
    images_dir = os.path.join(output_dir, 'images')
    captions_dir = os.path.join(output_dir, 'captions')
    clip_embeddings_dir = os.path.join(output_dir, 'embedding', 'clip_cropped_boxes')
    dino_embeddings_dir = os.path.join(output_dir, 'embedding', 'dino_cropped_boxes')
    
    json_files = sorted([f for f in os.listdir(captions_dir) if f.endswith('.json')])
    
    if sample_idx >= len(json_files):
        raise ValueError(f"Sample index {sample_idx} out of range (only {len(json_files)} samples)")
    
    json_file = json_files[sample_idx]
    json_path = os.path.join(captions_dir, json_file)
    
    with open(json_path, 'r') as f:
        caption_data = json.load(f)
    
    image_id = caption_data['image_id']
    image_path = caption_data['image_path']
    
    if not os.path.isabs(image_path):
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            image_filename = os.path.basename(image_path)
            image_path = os.path.join(images_dir, image_filename)
    
    image = Image.open(image_path).convert('RGB')
    
    objects = []
    for obj in caption_data.get('objects', []):
        obj_id = obj['object_id']
        
        clip_emb_path = os.path.join(clip_embeddings_dir, f"{image_id}_obj{obj_id}.npy")
        dino_emb_path = os.path.join(dino_embeddings_dir, f"{image_id}_obj{obj_id}.npy")
        
        clip_embedding = None
        dino_embedding = None
        
        if os.path.exists(clip_emb_path):
            clip_embedding = torch.from_numpy(np.load(clip_emb_path)).to(device)
        
        if os.path.exists(dino_emb_path):
            dino_embedding = torch.from_numpy(np.load(dino_emb_path)).to(device)
        
        objects.append({
            'object_id': obj_id,
            'category_name': obj.get('category_name', 'object'),
            'bbox': obj['bbox'],
            'depth_label': obj.get('depth_label', 'midground'),
            'caption': obj.get('caption', ''),
            'clip_embedding': clip_embedding,
            'dino_embedding': dino_embedding,
        })
    
    return {
        'image': image,
        'image_id': image_id,
        'global_caption': caption_data.get('global_caption', ''),
        'width': caption_data['width'],
        'height': caption_data['height'],
        'objects': objects,
    }


def prepare_batch_from_sample(
    sample: Dict,
    max_refs: int = 50,
    device: str = 'cpu',
) -> Dict:
    """
    Convert a loaded sample into batch-ready tensors.
    
    Args:
        sample: dict from load_sample()
        max_refs: maximum number of references (pad to this)
        device: device to place tensors on
    
    Returns:
        batch_dict: dictionary with:
            - bboxes: (1, max_refs, 4) tensor
            - depth_labels: (1, max_refs) list of strings
            - clip_embeddings: (1, max_refs, 768) tensor
            - dino_embeddings: (1, max_refs, 1024) tensor
            - num_refs: int (actual number of refs)
            - image: PIL Image
            - global_caption: str
    """
    objects = sample['objects']
    num_refs = min(len(objects), max_refs)
    
    bboxes = torch.zeros(1, max_refs, 4, device=device)
    depth_labels = ['background'] * max_refs
    
    clip_dim = 768
    if len(objects) > 0 and objects[0]['clip_embedding'] is not None:
        clip_dim = objects[0]['clip_embedding'].shape[0]
    
    clip_embeddings = torch.zeros(1, max_refs, clip_dim, device=device)
    dino_embeddings = torch.zeros(1, max_refs, 1024, device=device)
    
    for i, obj in enumerate(objects[:max_refs]):
        bbox = obj['bbox']
        if len(bbox) == 4:
            bboxes[0, i] = torch.tensor(bbox, device=device)
        
        depth_labels[i] = obj['depth_label']
        
        if obj['clip_embedding'] is not None:
            clip_embeddings[0, i] = obj['clip_embedding'].to(device)
        
        if obj['dino_embedding'] is not None:
            dino_embeddings[0, i] = obj['dino_embedding'].to(device)
    
    return {
        'bboxes': bboxes,
        'depth_labels': depth_labels,
        'clip_embeddings': clip_embeddings,
        'dino_embeddings': dino_embeddings,
        'num_refs': num_refs,
        'image': sample['image'],
        'global_caption': sample['global_caption'],
        'width': sample['width'],
        'height': sample['height'],
    }


class MoviePostProductionDataset(Dataset):
    """
    PyTorch Dataset for MoviePostProduction training.
    
    Loads precomputed:
    - CLIP embeddings (per-object crops)
    - DINO embeddings (per-object crops)
    - VAE latents (full images)
    - Caption JSONs with bboxes and metadata
    """
    
    def __init__(
        self,
        output_dir: str = './output_data',
        max_refs: int = 50,
        split: str = 'train',
        train_ratio: float = 0.9,
    ):
        """
        Args:
            output_dir: Directory containing preprocessed data
            max_refs: Maximum number of references per sample (pad/truncate)
            split: 'train' or 'val'
            train_ratio: Ratio of data to use for training
        """
        self.output_dir = output_dir
        self.max_refs = max_refs
        self.split = split
        
        self.captions_dir = os.path.join(output_dir, 'captions')
        self.clip_embeddings_dir = os.path.join(output_dir, 'embedding', 'clip_cropped_boxes')
        self.dino_embeddings_dir = os.path.join(output_dir, 'embedding', 'dino_cropped_boxes')
        self.latents_dir = os.path.join(output_dir, 'latents', 'vae_full_images')
        self.text_embeddings_dir = os.path.join(output_dir, 'latents', 'text_embeddings')
        
        json_files = sorted([f for f in os.listdir(self.captions_dir) if f.endswith('.json')])
        
        split_idx = int(len(json_files) * train_ratio)
        if split == 'train':
            self.json_files = json_files[:split_idx]
        else:
            self.json_files = json_files[split_idx:]
        
        print(f"Loaded {len(self.json_files)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.json_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single training sample.
        
        Returns:
            dict with:
                - latents: (16, H, W) tensor - precomputed VAE latents
                - bboxes: (max_refs, 4) tensor - normalized [x1, y1, x2, y2]
                - depth_labels: (max_refs,) list of strings
                - clip_embeddings: (max_refs, 512) tensor
                - dino_embeddings: (max_refs, 1024) tensor
                - num_refs: int - actual number of objects
                - image_id: str
                - global_caption: str
                - width: int
                - height: int
        """
        json_file = self.json_files[idx]
        json_path = os.path.join(self.captions_dir, json_file)
        
        with open(json_path, 'r') as f:
            caption_data = json.load(f)
        
        image_id = caption_data['image_id']
        width = caption_data['width']
        height = caption_data['height']
        global_caption = caption_data.get('global_caption', '')
        objects = caption_data.get('objects', [])
        
        num_refs = min(len(objects), self.max_refs)
        
        bboxes = torch.zeros(self.max_refs, 4)
        depth_labels = ['background'] * self.max_refs
        clip_embeddings = torch.zeros(self.max_refs, 512)
        dino_embeddings = torch.zeros(self.max_refs, 1024)
        
        for i, obj in enumerate(objects[:self.max_refs]):
            obj_id = obj['object_id']
            bbox = obj['bbox']
            
            if len(bbox) == 4:
                bboxes[i] = torch.tensor(bbox, dtype=torch.float32)
            
            depth_labels[i] = obj.get('depth_label', 'midground')
            
            clip_emb_path = os.path.join(self.clip_embeddings_dir, f"{image_id}_obj{obj_id}.npy")
            if os.path.exists(clip_emb_path):
                clip_emb = np.load(clip_emb_path)
                clip_embeddings[i] = torch.from_numpy(clip_emb).float()
            
            dino_emb_path = os.path.join(self.dino_embeddings_dir, f"{image_id}_obj{obj_id}.npy")
            if os.path.exists(dino_emb_path):
                dino_emb = np.load(dino_emb_path)
                dino_embeddings[i] = torch.from_numpy(dino_emb).float()
        
        latents_path = os.path.join(self.latents_dir, f"{image_id}.pt")
        if os.path.exists(latents_path):
            latents = torch.load(latents_path, map_location='cpu')
            if latents.dim() == 4:
                latents = latents.squeeze(0)
        else:
            latents = torch.zeros(16, 128, 128)
            print(f"Warning: Latents not found for {image_id}, using zeros")
        
        # Load cached text embeddings
        text_embeddings_path = os.path.join(self.text_embeddings_dir, f"{image_id}.pt")
        if os.path.exists(text_embeddings_path):
            text_data = torch.load(text_embeddings_path, map_location='cpu')
            
            # Load T5 embeddings (for main text encoding)
            t5_text_embeddings = text_data.get('t5_text_embeddings', text_data.get('text_embeddings'))  # (1, 512, 4096)
            t5_pooled_embeddings = text_data.get('t5_pooled_embeddings')  # (1, 4096)
            
            # Load CLIP embeddings (for pooled projection)
            clip_text_embeddings = text_data.get('clip_text_embeddings')  # (1, 77, 768)
            clip_pooled_embeddings = text_data.get('clip_pooled_embeddings', text_data.get('pooled_embeddings'))  # (1, 768)
            
            # Squeeze batch dimension
            if t5_text_embeddings.dim() == 3:
                t5_text_embeddings = t5_text_embeddings.squeeze(0)  # (512, 4096)
            if t5_pooled_embeddings is not None and t5_pooled_embeddings.dim() == 2:
                t5_pooled_embeddings = t5_pooled_embeddings.squeeze(0)  # (4096,)
            if clip_text_embeddings is not None and clip_text_embeddings.dim() == 3:
                clip_text_embeddings = clip_text_embeddings.squeeze(0)  # (77, 768)
            if clip_pooled_embeddings.dim() == 2:
                clip_pooled_embeddings = clip_pooled_embeddings.squeeze(0)  # (768,)
            
            # Use legacy keys for backward compatibility (T5 text, CLIP pooled)
            text_embeddings = t5_text_embeddings
            pooled_embeddings = clip_pooled_embeddings
        else:
            # Fallback to zeros if not cached
            t5_text_embeddings = torch.zeros(512, 4096)  # T5 dimensions
            t5_pooled_embeddings = torch.zeros(4096)  # T5 pooled dimensions
            clip_text_embeddings = torch.zeros(77, 768)  # CLIP dimensions
            clip_pooled_embeddings = torch.zeros(768)  # CLIP pooled dimensions
            text_embeddings = t5_text_embeddings
            pooled_embeddings = clip_pooled_embeddings
            print(f"Warning: Text embeddings not found for {image_id}, using zeros")
        
        return {
            'latents': latents,
            'bboxes': bboxes,
            'depth_labels': depth_labels,
            'clip_embeddings': clip_embeddings,
            'dino_embeddings': dino_embeddings,
            # T5 embeddings
            't5_text_embeddings': t5_text_embeddings,
            't5_pooled_embeddings': t5_pooled_embeddings,
            # CLIP embeddings
            'clip_text_embeddings': clip_text_embeddings,
            'clip_pooled_embeddings': clip_pooled_embeddings,
            # Legacy keys (T5 text + CLIP pooled for FLUX)
            'text_embeddings': text_embeddings,
            'pooled_embeddings': pooled_embeddings,
            'num_refs': num_refs,
            'image_id': image_id,
            'global_caption': global_caption,
            'width': width,
            'height': height,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of samples from MoviePostProductionDataset
    
    Returns:
        Batched dict with all tensors properly stacked
    """
    latents = torch.stack([item['latents'] for item in batch])
    bboxes = torch.stack([item['bboxes'] for item in batch])
    clip_embeddings = torch.stack([item['clip_embeddings'] for item in batch])
    dino_embeddings = torch.stack([item['dino_embeddings'] for item in batch])
    
    # T5 embeddings
    t5_text_embeddings = torch.stack([item['t5_text_embeddings'] for item in batch])
    t5_pooled_embeddings = torch.stack([item['t5_pooled_embeddings'] for item in batch])
    
    # CLIP embeddings
    clip_text_embeddings = torch.stack([item['clip_text_embeddings'] for item in batch])
    clip_pooled_embeddings = torch.stack([item['clip_pooled_embeddings'] for item in batch])
    
    # Legacy keys (T5 text + CLIP pooled for FLUX)
    text_embeddings = torch.stack([item['text_embeddings'] for item in batch])
    pooled_embeddings = torch.stack([item['pooled_embeddings'] for item in batch])
    
    depth_labels = [item['depth_labels'] for item in batch]
    num_refs = [item['num_refs'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    global_captions = [item['global_caption'] for item in batch]
    widths = [item['width'] for item in batch]
    heights = [item['height'] for item in batch]
    
    return {
        'latents': latents,
        'bboxes': bboxes,
        'depth_labels': depth_labels,
        'clip_embeddings': clip_embeddings,
        'dino_embeddings': dino_embeddings,
        # T5 embeddings
        't5_text_embeddings': t5_text_embeddings,
        't5_pooled_embeddings': t5_pooled_embeddings,
        # CLIP embeddings
        'clip_text_embeddings': clip_text_embeddings,
        'clip_pooled_embeddings': clip_pooled_embeddings,
        # Legacy keys (T5 text + CLIP pooled for FLUX)
        'text_embeddings': text_embeddings,
        'pooled_embeddings': pooled_embeddings,
        'num_refs': num_refs,
        'image_ids': image_ids,
        'global_captions': global_captions,
        'widths': widths,
        'heights': heights,
    }


if __name__ == "__main__":
    print("Testing data_loader...")
    
    try:
        print("\n[1/3] Testing load_sample function...")
        sample = load_sample(sample_idx=0, output_dir='./output_data')
        print(f"✓ Loaded sample: {sample['image_id']}")
        print(f"  Image size: {sample['width']}×{sample['height']}")
        print(f"  Global caption: {sample['global_caption'][:50]}...")
        print(f"  Number of objects: {len(sample['objects'])}")
        
        print("\n[2/3] Testing MoviePostProductionDataset...")
        dataset = MoviePostProductionDataset(output_dir='./output_data', max_refs=50, split='train')
        print(f"✓ Created dataset with {len(dataset)} samples")
        
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Latents shape: {sample['latents'].shape}")
        print(f"  Bboxes shape: {sample['bboxes'].shape}")
        print(f"  CLIP embeddings shape: {sample['clip_embeddings'].shape}")
        print(f"  DINO embeddings shape: {sample['dino_embeddings'].shape}")
        print(f"  Num refs: {sample['num_refs']}")
        
        print("\n[3/3] Testing DataLoader with collate_fn...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(dataloader))
        print(f"✓ Created batch:")
        print(f"  Latents shape: {batch['latents'].shape}")
        print(f"  Bboxes shape: {batch['bboxes'].shape}")
        print(f"  CLIP embeddings shape: {batch['clip_embeddings'].shape}")
        print(f"  DINO embeddings shape: {batch['dino_embeddings'].shape}")
        print(f"  Batch size: {len(batch['image_ids'])}")
        
        print(f"\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Make sure output_data/ exists with preprocessed data")
        print(f"  Run prepare_synthetic_dataset.py first if needed")