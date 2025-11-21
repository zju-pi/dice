"""Script for calculating the CLIP Score."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import json
import click
import tqdm
import torch
import open_clip
from torchvision import transforms
from accelerate import Accelerator
from torch_utils import dataset

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate CLIP score.
    python clip_score.py calc --images=path/to/images
    or running on multiple GPUs (e.g., 4):
    torchrun --standalone --nproc_per_node=4 clip_score.py calc --images=path/to/images
    """

#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=250, show_default=True)
@click.option('--desc',                 help='A description string', metavar='str',                 type=str)

@torch.no_grad()
def calc(image_path, batch, desc=None, num_expected=None, seed=0, max_batch_size=64,
    num_workers=0, prefetch_factor=None, device=torch.device('cuda')):
    
    accelerator = Accelerator()
    device = accelerator.device

    # List images.
    accelerator.print(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)

    # Load prompts
    # COCO 2017 valiadation set - 5k
    prompt_path = "./assets/val2017_5k.json"
    accelerator.print(f"Loading MS-COCO 2017 valiadation captions from {prompt_path}...")
    sample_captions = list(json.load(open(prompt_path, 'r')).values())

    # Loading CLIP model
    accelerator.print(f'Loading CLIP-ViT-g-14 model...')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    tokenizer = open_clip.get_tokenizer('ViT-g-14')
    model.to(device)

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * accelerator.num_processes) + 1) * accelerator.num_processes
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[accelerator.process_index :: accelerator.num_processes]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    accelerator.print(f'Calculating statistics for {len(dataset_obj)} images...')
    avg_clip_score, batch_idx = 0, 0
    to_pil = transforms.ToPILImage()
    for images, _ in tqdm.tqdm(data_loader, unit='batch', disable=(accelerator.process_index != 0)):
        accelerator.wait_for_everyone()
        prompts = sample_captions[rank_batches[batch_idx][0]:rank_batches[batch_idx][-1]+1]

        images = torch.stack([preprocess(to_pil(img)) for img in images], dim=0).to(device)
        text = tokenizer(prompts).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        sd_clip_score = 100 * (image_features * text_features).sum(axis=-1)
        avg_clip_score += sd_clip_score.sum()
        batch_idx += 1
    
    avg_clip_score = accelerator.reduce(avg_clip_score, reduction="sum")
    avg_clip_score /= len(dataset_obj)
    accelerator.print(f"CLIP score: {avg_clip_score}")
    if accelerator.process_index == 0:
        assert desc is not None
        Note = open('results.txt', mode='a')
        Note.write(f'{desc}-CS: {avg_clip_score}\n')
        Note.close()
    accelerator.wait_for_everyone()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
