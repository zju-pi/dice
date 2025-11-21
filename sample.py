import re
import os
import json
import argparse
import pickle
import tqdm
import random
import numpy as np
import torch
from pipelines.pipeline_stable_diffusion import StableDiffusionPipeline
from pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler, AutoencoderKL
from accelerate import Accelerator
from torch_utils.util import open_url

parser = argparse.ArgumentParser(description=globals()["__doc__"])
parser.add_argument('--outdir',         help='Where to save images', type=str, default='./samples')
parser.add_argument('--batch_gpu',      help='Batch size per gpu', type=int, default=4)
parser.add_argument('--seed',           help='Random seed', type=int, default=0)
parser.add_argument('--steps',          help='Sampling steps',  type=int, default=20)
parser.add_argument('--model_name',     help='Model name', type=str, default='sd15')
parser.add_argument('--cfg_scale',      help='Guidance scale, will be set to 1.0 if DICE sharpener is specified', type=float, default=5.0)
parser.add_argument('--sharpener_path', help='Path or expiemnt id of the DICE sharpener', type=str)
parser.add_argument('--sharpener_id',   help='Id of the sharpener, will be used as folder name to save images', type=str)
parser.add_argument('--alpha',          help='Strength for sharpener', type=float, default=1.0)
args = parser.parse_args()

assert args.model_name in ['sd15', 'sdxl']

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


# Enable parallel computing
accelerator = Accelerator()
device = accelerator.device
accelerator.print('--------------settings--------------')
accelerator.print(args)
accelerator.print('------------------------------------')

# Basic hyperparameters
seed = args.seed
seeds = parse_int_list('0-4999')

# Load model
if args.model_name == 'sd15':
    # pipe = StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained('Lykon/DreamShaper', torch_dtype=torch.float16)
elif args.model_name == 'sdxl':
    pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to(device)

# Pipe configuration
pipe.set_progress_bar_config(disable=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load sampled COCO 2017 valiadation set - 5k prompts
prompt_path = "./assets/val2017_5k.json"
accelerator.print(f"Loading MS-COCO 2017 valiadation captions from {prompt_path}...")
sample_captions = list(json.load(open(prompt_path, 'r')).values())
accelerator.print('Finish, num of prompts:', len(sample_captions))

# Load DICE sharpener
if args.sharpener_path is not None:
    sharpener_path = args.sharpener_path
    if not sharpener_path.endswith('pkl'):      # load by experiment number
        # find the directory with distilled models
        predictor_path_str = '0' * (5 - len(sharpener_path)) + sharpener_path
        for file_name in os.listdir("./exps"):
            if file_name.split('-')[0] == predictor_path_str:
                sharpener_path = os.path.join('./exps', file_name, f'checkpoint-{file_name.split("-")[2]}/sharpener-snapshot.pkl')
                break
    accelerator.print(f'Loading embedding model from "{sharpener_path}"...')
    with open_url(sharpener_path, verbose=(accelerator.process_index == 0)) as f:
        sharpener = pickle.load(f)['model'].to(device)
    sharpener.eval()
    pipe.sharpener = sharpener
    pipe.sharpener_alpha = args.alpha
    
# Generate images
if args.sharpener_path is not None:
    outdir_img = os.path.join(args.outdir, f"{args.model_name}_{args.sharpener_id}_steps{args.steps}_alpha{args.alpha}")
else:
    outdir_img = os.path.join(args.outdir, f"{args.model_name}_base_steps{args.steps}_cfg{args.cfg_scale}")
seed_everything(seed+accelerator.process_index)
generator = torch.Generator().manual_seed(seed+accelerator.process_index)
num_batches = ((len(seeds) - 1) // (args.batch_gpu * accelerator.num_processes) + 1) * accelerator.num_processes
all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
rank_batches = all_batches[accelerator.process_index :: accelerator.num_processes]
for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(accelerator.process_index != 0)):
    accelerator.wait_for_everyone()
    prompts = sample_captions[batch_seeds[0]:batch_seeds[-1]+1]
    with torch.no_grad():
        images = pipe(
            prompts, 
            generator=generator, 
            num_images_per_prompt=1, 
            num_inference_steps=args.steps, 
            guidance_scale=args.cfg_scale,
        ).images

    # Save images
    for seed, image in zip(batch_seeds, images):
        image_dir = os.path.join(outdir_img, f'{seed-seed%1000:06d}')
        os.makedirs(image_dir, exist_ok=True)
        image.save(os.path.join(image_dir, f'{seed:06d}.png'))

