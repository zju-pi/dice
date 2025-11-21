"""Script for calculating the CLIP Score."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import json
import click
import tqdm
import torch
from torchvision import transforms
from accelerate import Accelerator
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
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
    
    # Load MLP
    class MLP(pl.LightningModule):
        def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
            super().__init__()
            self.input_size = input_size
            self.xcol = xcol
            self.ycol = ycol
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.layers(x)

        def training_step(self, batch, batch_idx):
                x = batch[self.xcol]
                y = batch[self.ycol].reshape(-1, 1)
                x_hat = self.layers(x)
                loss = F.mse_loss(x_hat, y)
                return loss
        
        def validation_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer

    def normalized(a, axis=-1, order=2):
        import numpy as np  # pylint: disable=import-outside-toplevel

        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    # download from https://github.com/christophschuhmann/improved-aesthetic-predictor
    s = torch.load("/Path/to/sac+logos+ava1-l14-linearMSE.pth")   
    model.load_state_dict(s)
    model.to(device)
    model.eval()

    # Loading CLIP model
    accelerator.print(f'Loading CLIP-ViT-H-14 model...')
    version = "openai/clip-vit-large-patch14"
    model2 = CLIPModel.from_pretrained(version).to(device)
    preprocess = CLIPProcessor.from_pretrained(version)

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * accelerator.num_processes) + 1) * accelerator.num_processes
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[accelerator.process_index :: accelerator.num_processes]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    accelerator.print(f'Calculating statistics for {len(dataset_obj)} images...')
    avg_aes_score, batch_idx = 0, 0
    to_pil = transforms.ToPILImage()
    for images, _ in tqdm.tqdm(data_loader, unit='batch', disable=(accelerator.process_index != 0)):
        accelerator.wait_for_everyone()
        prompts = sample_captions[rank_batches[batch_idx][0]:rank_batches[batch_idx][-1]+1]

        for i in range(len(prompts)):
            img = to_pil(images[i])
            inputs = preprocess(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model2.get_image_features(**inputs).unsqueeze(1)
            im_emb_arr = normalized(image_features.cpu().detach().numpy() )
            scores = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
            avg_aes_score += scores.sum()

        batch_idx += 1
    
    avg_aes_score = accelerator.reduce(avg_aes_score, reduction="sum")
    avg_aes_score /= len(dataset_obj)
    accelerator.print(f"AES score: {avg_aes_score}")
    if accelerator.process_index == 0:
        assert desc is not None
        Note = open('results.txt', mode='a')
        Note.write(f'{desc}-AES: {avg_aes_score}\n')
        Note.close()
    accelerator.wait_for_everyone()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
