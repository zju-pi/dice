# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
from accelerate import Accelerator  # Can also be PartialState or AcceleratorState
from torch_utils import dataset
from torch_utils.util import open_url

#----------------------------------------------------------------------------

def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=0, prefetch_factor=None, device=torch.device('cuda'), accelerator=None
):
    assert accelerator is not None

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    accelerator.print('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    with open_url(detector_url, verbose=(accelerator.process_index == 0)) as f:
        detector_net = pickle.load(f).to(device)
    
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048

    # List images.
    accelerator.print(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * accelerator.num_processes) + 1) * accelerator.num_processes
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[accelerator.process_index :: accelerator.num_processes]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    accelerator.print(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(accelerator.process_index != 0)):
        accelerator.wait_for_everyone()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    mu = accelerator.reduce(mu, reduction="sum")
    sigma = accelerator.reduce(sigma, reduction="sum")
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate Frechet Inception Distance (FID).

    Examples:

    \b
    # Generate 50000 images and save them as fid-tmp/*/*.png
    torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Calculate FID
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

    \b
    # Compute dataset reference statistics
    python fid.py ref --data=datasets/my-dataset.zip --dest=fid-refs/my-dataset.npz
    """

#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--desc',                 help='A description string', metavar='str',                 type=str)

def calc(image_path, ref_path, num_expected, seed, batch, desc=None):
    """Calculate FID for a given set of images."""

    accelerator = Accelerator()
    device = accelerator.device

    accelerator.print(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if accelerator.process_index == 0:
        with open_url(ref_path) as f:
            ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch, device=device, accelerator=accelerator)
    accelerator.print('Calculating FID...')
    if accelerator.process_index == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        print(f'{fid:g}')
        Note = open('results.txt', mode='a')
        Note.write(f'{desc}-FID: {fid}\n') if desc is not None else Note.write('{} {} {}\n'.format(image_path.split('/')[-2], image_path.split('/')[-1], fid))
        Note.close()
    accelerator.wait_for_everyone()

#----------------------------------------------------------------------------

@main.command()
@click.option('--data', 'dataset_path', help='Path to the dataset', metavar='PATH|ZIP', type=str, required=True)
@click.option('--dest', 'dest_path',    help='Destination .npz file', metavar='NPZ',    type=str, required=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',       type=click.IntRange(min=1), default=500, show_default=True)

def ref(dataset_path, dest_path, batch):
    """Calculate dataset reference statistics needed by 'calc'."""

    accelerator = Accelerator()
    device = accelerator.device

    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch, device=device, accelerator=accelerator)
    accelerator.print(f'Saving dataset reference statistics to "{dest_path}"...')
    if accelerator.process_index == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    accelerator.wait_for_everyone()
    accelerator.print('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
