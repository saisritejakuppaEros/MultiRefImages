from datasets import load_dataset
import os
import json
from PIL import Image
import torch
import clip
import numpy as np
from torchvision import transforms
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

def create_dataset(output_dir='./output_data', dataset_repo="FireRedTeam/DenseLayout", 
                   current_dir='./dataset', max_samples=5):
    """
    Create dataset with images and captions from DenseLayout dataset.
    """
    # Load the dataset
    ds = load_dataset(dataset_repo, split="test", cache_dir=current_dir)
    
    # Ensure output directories exist
    images_dir = os.path.join(output_dir, 'images')
    captions_dir = os.path.join(output_dir, 'captions')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(captions_dir, exist_ok=True)
    
    # Process samples and create dataset
    for sample_idx, sample in enumerate(ds):
        # Get image and metadata
        image = sample['image']
        image_id = sample.get('id', f'img{sample_idx}')
        prompt = sample.get('prompt', '')
        annos = sample.get('annos', [])
        
        # Flatten annos if it's a list of lists
        if annos and isinstance(annos[0], list):
            annos = [item for sublist in annos for item in sublist]
        
        # Save ground truth composited scene image
        image_filename = f"{image_id}.png"
        image_path = os.path.join(images_dir, image_filename)
        image.save(image_path)
        
        # Process objects: create isolated crops, extract bboxes, and assign depth labels
        objects = []
        for obj_idx, obj in enumerate(annos):
            if not isinstance(obj, dict):
                continue
                
            bbox = obj.get('bbox', [])
            if not bbox or len(bbox) < 4:
                continue
                
            category_name = obj.get('category_name', 'object')
            caption = obj.get('caption', '')
            
            # Extract bbox coordinates [xmin, ymin, xmax, ymax] in absolute pixels
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Ensure bbox is within image bounds
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(image.width, int(xmax))
            ymax = min(image.height, int(ymax))
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            # Normalize bbox coordinates to [0, 1] range based on image size
            normalized_bbox = [
                xmin / image.width,   # normalized xmin
                ymin / image.height,  # normalized ymin
                xmax / image.width,   # normalized xmax
                ymax / image.height   # normalized ymax
            ]
            
            # Estimate depth label based on bbox size and position (simple heuristic)
            # Objects closer to camera are typically larger and lower in image
            bbox_area = (xmax - xmin) * (ymax - ymin)
            image_area = image.width * image.height
            relative_size = bbox_area / image_area
            y_center = (ymin + ymax) / 2 / image.height
            
            if relative_size > 0.1 or y_center > 0.7:
                depth_label = "foreground"
            elif relative_size < 0.01 or y_center < 0.3:
                depth_label = "background"
            else:
                depth_label = "midground"
            
            objects.append({
                "object_id": obj_idx,
                "category_name": category_name,
                "bbox": normalized_bbox,
                "depth_label": depth_label,
                "caption": caption
            })
        
        # Create caption JSON structure
        caption_data = {
            "image_id": image_id,
            "image_path": image_path,
            "global_caption": prompt,
            "width": image.width,
            "height": image.height,
            "objects": objects
        }
        
        # Save caption JSON
        caption_filename = f"{image_id}.json"
        caption_path = os.path.join(captions_dir, caption_filename)
        with open(caption_path, 'w') as f:
            json.dump(caption_data, f, indent=2)
        
        print(f"Processed sample {sample_idx}: {image_id}")
        
        # Limit samples for testing
        if sample_idx >= max_samples - 1:
            break
    
    print(f"\nDataset created successfully!")
    print(f"Images saved to: {images_dir}")
    print(f"Captions saved to: {captions_dir}")


def generate_clip_embeddings_for_cropped_boxes(output_dir='./output_data'):
    """
    Generate CLIP embeddings for all cropped boxes from the dataset.
    Saves embeddings to embedding/clip_cropped_boxes directory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # Setup directories
    images_dir = os.path.join(output_dir, 'images')
    captions_dir = os.path.join(output_dir, 'captions')
    embeddings_dir = os.path.join(output_dir, 'embedding', 'clip_cropped_boxes')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = [f for f in os.listdir(captions_dir) if f.endswith('.json')]
    json_files.sort()
    
    print(f"Processing {len(json_files)} images...")
    
    for json_file in json_files:
        json_path = os.path.join(captions_dir, json_file)
        
        # Load caption data
        with open(json_path, 'r') as f:
            caption_data = json.load(f)
        
        image_id = caption_data['image_id']
        image_path = caption_data['image_path']
        width = caption_data['width']
        height = caption_data['height']
        objects = caption_data.get('objects', [])
        
        # Load the full image - handle relative paths correctly
        if not os.path.isabs(image_path):
            # Normalize the path - resolve relative to current working directory
            image_path = os.path.normpath(image_path)
            # If the path doesn't exist, try constructing it from images_dir
            if not os.path.exists(image_path):
                # Extract just the filename and construct from images_dir
                image_filename = os.path.basename(image_path)
                image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        image = Image.open(image_path).convert('RGB')
        
        # Process each object
        embeddings = []
        for obj in objects:
            obj_id = obj['object_id']
            bbox = obj['bbox']
            
            # Check if bbox is normalized (values between 0 and 1) or in pixels
            if max(bbox) <= 1.0:
                # Normalized coordinates, convert to pixels
                xmin = int(bbox[0] * width)
                ymin = int(bbox[1] * height)
                xmax = int(bbox[2] * width)
                ymax = int(bbox[3] * height)
            else:
                # Already in pixels
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Ensure bbox is within image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(image.width, xmax)
            ymax = min(image.height, ymax)
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            # Crop the image to get the isolated object box
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            
            # Preprocess and resize cropped box to CLIP format (224x224)
            preprocessed_image = preprocess(cropped_image).unsqueeze(0).to(device)
            
            # Generate CLIP embedding
            with torch.no_grad():
                image_features = model.encode_image(preprocessed_image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
            
            # Convert to numpy and save
            embedding_np = image_features.cpu().numpy().flatten()
            
            # Save embedding for this cropped box
            embedding_filename = f"{image_id}_obj{obj_id}.npy"
            embedding_path = os.path.join(embeddings_dir, embedding_filename)
            np.save(embedding_path, embedding_np)
            
            embeddings.append({
                "object_id": obj_id,
                "embedding_path": embedding_path,
                "bbox": bbox,
                "crop_size": (xmax - xmin, ymax - ymin)  # Store crop dimensions for verification
            })
        
        print(f"Generated {len(embeddings)} CLIP embeddings for cropped boxes in {image_id}")
    
    print(f"\nCLIP embeddings generated successfully!")
    print(f"Embeddings saved to: {embeddings_dir}")


def generate_dino_embeddings_for_cropped_boxes(output_dir='./output_data'):
    """
    Generate DINOv2 embeddings for all cropped boxes from the dataset.
    Uses DINOv2 ViT-L/14 model (1024-dim embeddings).
    Saves embeddings to embedding/dino_cropped_boxes directory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv2 ViT-L/14 model on {device}...")
    
    # Load the DINOv2 ViT-L/14 model (1024-dim embeddings)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.to(device)
    model.eval()
    
    # Standard ImageNet normalization transform
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Setup directories
    images_dir = os.path.join(output_dir, 'images')
    captions_dir = os.path.join(output_dir, 'captions')
    embeddings_dir = os.path.join(output_dir, 'embedding', 'dino_cropped_boxes')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = [f for f in os.listdir(captions_dir) if f.endswith('.json')]
    json_files.sort()
    
    print(f"Processing {len(json_files)} images...")
    
    for json_file in json_files:
        json_path = os.path.join(captions_dir, json_file)
        
        # Load caption data
        with open(json_path, 'r') as f:
            caption_data = json.load(f)
        
        image_id = caption_data['image_id']
        image_path = caption_data['image_path']
        width = caption_data['width']
        height = caption_data['height']
        objects = caption_data.get('objects', [])
        
        # Load the full image - handle relative paths correctly
        if not os.path.isabs(image_path):
            # Normalize the path - resolve relative to current working directory
            image_path = os.path.normpath(image_path)
            # If the path doesn't exist, try constructing it from images_dir
            if not os.path.exists(image_path):
                # Extract just the filename and construct from images_dir
                image_filename = os.path.basename(image_path)
                image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        image = Image.open(image_path).convert('RGB')
        
        # Process each object
        embeddings = []
        for obj in objects:
            obj_id = obj['object_id']
            bbox = obj['bbox']
            
            # Check if bbox is normalized (values between 0 and 1) or in pixels
            if max(bbox) <= 1.0:
                # Normalized coordinates, convert to pixels
                xmin = int(bbox[0] * width)
                ymin = int(bbox[1] * height)
                xmax = int(bbox[2] * width)
                ymax = int(bbox[3] * height)
            else:
                # Already in pixels
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Ensure bbox is within image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(image.width, xmax)
            ymax = min(image.height, ymax)
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            # Crop the image to get the isolated object box
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            
            # Preprocess cropped box for DINOv2
            input_tensor = transform(cropped_image).unsqueeze(0).to(device)  # Add batch dim
            
            # Extract embeddings (no_grad for inference)
            with torch.no_grad():
                features = model(input_tensor)  # Shape: [1, 1024] (CLS token embedding)
            
            # Global embedding: CLS token is already returned directly
            global_embedding = features  # Shape: [1, 1024]
            
            # Normalize L2
            global_embedding = torch.nn.functional.normalize(global_embedding, p=2, dim=-1)
            
            # Convert to numpy and save
            embedding_np = global_embedding.cpu().numpy().flatten()  # Shape: [1024]
            
            # Save embedding for this cropped box
            embedding_filename = f"{image_id}_obj{obj_id}.npy"
            embedding_path = os.path.join(embeddings_dir, embedding_filename)
            np.save(embedding_path, embedding_np)
            
            embeddings.append({
                "object_id": obj_id,
                "embedding_path": embedding_path,
                "bbox": bbox,
                "crop_size": (xmax - xmin, ymax - ymin)  # Store crop dimensions for verification
            })
        
        print(f"Generated {len(embeddings)} DINO embeddings for cropped boxes in {image_id}")
    
    print(f"\nDINO embeddings generated successfully!")
    print(f"Embeddings saved to: {embeddings_dir}")


def cache_vae_latents_for_full_images(output_dir='./output_data', vae_path="/path/to/cache/flux1-dev-vae"):
    """
    Cache VAE latents for all full images (not cropped boxes) from the dataset.
    Uses FLUX VAE to encode images to latents.
    Saves latents to latents/vae_full_images directory.
    
    Args:
        output_dir: Directory containing images and captions
        vae_path: Path to the FLUX VAE model directory
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"Loading VAE model from {vae_path} on {device}...")
    try:
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype).to(device)
        vae.eval()
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        print("Please ensure the VAE path is correct and the model is available.")
        return
    
    # Setup directories
    images_dir = os.path.join(output_dir, 'images')
    captions_dir = os.path.join(output_dir, 'captions')
    latents_dir = os.path.join(output_dir, 'latents', 'vae_full_images')
    os.makedirs(latents_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = [f for f in os.listdir(captions_dir) if f.endswith('.json')]
    json_files.sort()
    
    print(f"Processing {len(json_files)} full images...")
    
    for json_file in json_files:
        json_path = os.path.join(captions_dir, json_file)
        
        # Load caption data
        with open(json_path, 'r') as f:
            caption_data = json.load(f)
        
        image_id = caption_data['image_id']
        image_path = caption_data['image_path']
        
        # Load the full image - handle relative paths correctly
        if not os.path.isabs(image_path):
            # Normalize the path - resolve relative to current working directory
            image_path = os.path.normpath(image_path)
            # If the path doesn't exist, try constructing it from images_dir
            if not os.path.exists(image_path):
                # Extract just the filename and construct from images_dir
                image_filename = os.path.basename(image_path)
                image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 1024x1024 as recommended for FLUX VAE
        image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # Convert to tensor: [H, W, C] -> [C, H, W] -> [1, C, H, W]
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
        
        # Encode image to latents
        with torch.no_grad():
            latents = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
        
        # Save latents (on CPU for smaller file size)
        latents_filename = f"{image_id}.pt"
        latents_path = os.path.join(latents_dir, latents_filename)
        torch.save(latents.cpu(), latents_path)
        
        print(f"Cached VAE latents for {image_id}: shape {latents.shape}, saved to {latents_path}")
    
    print(f"\nVAE latents cached successfully!")
    print(f"Latents saved to: {latents_dir}")


def encode_text(captions, tokenizer, text_encoder, device):
    """Encode text captions to FLUX format."""
    tokens = tokenizer(
        captions,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        text_embeddings = text_encoder(tokens.input_ids).last_hidden_state
    
    # FLUX uses pooled projection (mean over sequence)
    pooled = text_embeddings.mean(dim=1)
    
    return text_embeddings, pooled


def create_position_ids(latent_h, latent_w, num_txt_tokens, device):
    """Create img_ids and txt_ids for RoPE."""
    # Image position IDs (H*W, 3)
    img_ids = torch.zeros(latent_h * latent_w, 3, device=device)
    for i in range(latent_h):
        for j in range(latent_w):
            idx = i * latent_w + j
            img_ids[idx, 1] = i
            img_ids[idx, 2] = j
    
    # Text position IDs (num_txt_tokens, 3)
    txt_ids = torch.zeros(num_txt_tokens, 3, device=device)
    for i in range(num_txt_tokens):
        txt_ids[i, 2] = i
    
    return img_ids, txt_ids


def pack_latents(latents):
    """Pack latents from (B, 16, H, W) to (B, H*W, 64)."""
    B, C, H, W = latents.shape
    # FLUX packs 2×2×4 = 16 channels into 64 dims
    latents = latents.view(B, 4, 2, 2, H, W)
    latents = latents.permute(0, 4, 5, 1, 2, 3)  # (B, H, W, 4, 2, 2)
    latents = latents.reshape(B, H * W, 64)
    return latents


def cache_text_embeddings_for_captions(output_dir='./output_data', text_encoder=None, tokenizer=None, clip_encoder=None, clip_tokenizer=None, device=None):
    """
    Cache T5 and CLIP text embeddings for all captions from the dataset.
    FLUX uses T5 for main text embeddings and CLIP for pooled projection.
    
    Args:
        output_dir: Directory containing images and captions
        text_encoder: T5EncoderModel instance (if None, will be loaded)
        tokenizer: T5Tokenizer instance (if None, will be loaded)
        clip_encoder: CLIPTextModel instance (if None, will be loaded)
        clip_tokenizer: CLIPTokenizer instance (if None, will be loaded)
        device: Device to use (if None, will be auto-detected)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load T5 text encoder and tokenizer if not provided
    if text_encoder is None or tokenizer is None:
        print(f"Loading T5 text encoder and tokenizer on {device}...")
        try:
            from transformers import T5EncoderModel, T5Tokenizer
            text_encoder = T5EncoderModel.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="text_encoder_2",
                torch_dtype=dtype,
            ).to(device)
            text_encoder.eval()
            for param in text_encoder.parameters():
                param.requires_grad = False
            
            tokenizer = T5Tokenizer.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="tokenizer_2",
            )
        except Exception as e:
            print(f"Error loading T5 text encoder: {e}")
            print("Please ensure the FLUX.1-dev model is available.")
            return
    
    # Load CLIP text encoder and tokenizer if not provided
    if clip_encoder is None or clip_tokenizer is None:
        print(f"Loading CLIP text encoder and tokenizer on {device}...")
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            clip_encoder = CLIPTextModel.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="text_encoder",
                torch_dtype=dtype,
            ).to(device)
            clip_encoder.eval()
            for param in clip_encoder.parameters():
                param.requires_grad = False
            
            clip_tokenizer = CLIPTokenizer.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                subfolder="tokenizer",
            )
        except Exception as e:
            print(f"Error loading CLIP text encoder: {e}")
            print("Please ensure the FLUX.1-dev model is available.")
            return
    
    # Setup directories
    captions_dir = os.path.join(output_dir, 'captions')
    text_embeddings_dir = os.path.join(output_dir, 'latents', 'text_embeddings')
    os.makedirs(text_embeddings_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = [f for f in os.listdir(captions_dir) if f.endswith('.json')]
    json_files.sort()
    
    print(f"Processing {len(json_files)} captions...")
    
    for json_file in json_files:
        json_path = os.path.join(captions_dir, json_file)
        
        # Load caption data
        with open(json_path, 'r') as f:
            caption_data = json.load(f)
        
        image_id = caption_data['image_id']
        global_caption = caption_data.get('global_caption', '')
        objects = caption_data.get('objects', [])
        
        # Encode global caption (main prompt for FLUX)
        if not global_caption:
            print(f"Warning: No global caption found for {image_id}, skipping...")
            continue
        
        # Encode global caption with T5 (main text embeddings)
        t5_tokens = tokenizer(
            [global_caption],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            t5_output = text_encoder(t5_tokens.input_ids)
            t5_text_embeddings = t5_output.last_hidden_state  # (1, seq_len, 4096)
            t5_pooled = t5_text_embeddings.mean(dim=1)  # (1, 4096) - mean pooling
        
        # Encode global caption with CLIP (for pooled projection)
        clip_tokens = clip_tokenizer(
            [global_caption],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            clip_output = clip_encoder(clip_tokens.input_ids)
            clip_text_embeddings = clip_output.last_hidden_state  # (1, 77, 768)
            clip_pooled = clip_output.pooler_output  # (1, 768) - [CLS] token
        
        # Save both T5 and CLIP embeddings separately
        # T5: (1, 512, 4096) text + (1, 4096) pooled
        # CLIP: (1, 77, 768) text + (1, 768) pooled
        embeddings_data = {
            # T5 embeddings (for FLUX encoder_hidden_states)
            't5_text_embeddings': t5_text_embeddings.cpu(),  # (1, seq_len, 4096)
            't5_pooled_embeddings': t5_pooled.cpu(),  # (1, 4096)
            
            # CLIP embeddings (for FLUX pooled_projections)
            'clip_text_embeddings': clip_text_embeddings.cpu(),  # (1, 77, 768)
            'clip_pooled_embeddings': clip_pooled.cpu(),  # (1, 768)
            
            # Legacy keys for backward compatibility (use T5 for text, CLIP for pooled)
            'text_embeddings': t5_text_embeddings.cpu(),  # T5 text
            'pooled_embeddings': clip_pooled.cpu(),  # CLIP pooled
            
            'global_caption': global_caption,
        }
        
        # Save embeddings
        embeddings_filename = f"{image_id}.pt"
        embeddings_path = os.path.join(text_embeddings_dir, embeddings_filename)
        torch.save(embeddings_data, embeddings_path)
        
        print(f"Cached text embeddings for {image_id}: "
              f"T5 text {t5_text_embeddings.shape}, T5 pooled {t5_pooled.shape}, "
              f"CLIP text {clip_text_embeddings.shape}, CLIP pooled {clip_pooled.shape}, "
              f"saved to {embeddings_path}")
    
    print(f"\nText embeddings cached successfully!")
    print(f"Text embeddings saved to: {text_embeddings_dir}")


if __name__ == "__main__":
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load T5 text encoder for FLUX
    print(f"Loading T5 text encoder on {device}...")
    from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
    
    text_encoder = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder_2",
        torch_dtype=dtype,
    ).to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    tokenizer = T5Tokenizer.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer_2",
    )
    print("✓ T5 text encoder loaded")
    
    # Load CLIP text encoder for FLUX
    print(f"Loading CLIP text encoder on {device}...")
    clip_encoder = CLIPTextModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).to(device)
    clip_encoder.eval()
    for param in clip_encoder.parameters():
        param.requires_grad = False

    clip_tokenizer = CLIPTokenizer.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="tokenizer",
    )
    print("✓ CLIP text encoder loaded")
    
    # Uncomment to create dataset first
    create_dataset()
    
    # Generate CLIP embeddings for all cropped boxes
    generate_clip_embeddings_for_cropped_boxes()

    # Generate DINO embeddings for all cropped boxes
    generate_dino_embeddings_for_cropped_boxes()
    
    # Cache VAE latents for all full images
    cache_vae_latents_for_full_images(vae_path="/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae")
    
    # Cache text embeddings for all captions
    cache_text_embeddings_for_captions(
        output_dir='./output_data',
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        clip_encoder=clip_encoder,
        clip_tokenizer=clip_tokenizer,
        device=device
    )


