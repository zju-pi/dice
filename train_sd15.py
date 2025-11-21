#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import copy
import os
import shutil
import random
from contextlib import nullcontext

import accelerate
import datasets
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import re
import pickle
from dice_sharpener import Sharpener
from torchvision.utils import make_grid, save_image
from pipelines.pipeline_stable_diffusion import StableDiffusionPipeline
from torch_utils.arg_parser import parse_args
from torch_utils.dataset import COCODataset, InfiniteSampler
from datasets import load_dataset

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")

#----------------------------------------------------------------------------

class RandomGenerator:
    def __init__(self, seed=42):
        random.seed(seed)

    def rand(self):
        while True:
            yield random.random()
    
    def randint(self, a, b):
        while True:
            yield random.randint(a, b)

#----------------------------------------------------------------------------

def main():
    args = parse_args()

    # Description string.
    total_batch_size = args.train_batch_size * args.n_gpus * args.gradient_accumulation_steps
    desc = f'sd15-{args.max_train_steps}-batch{total_batch_size}-cfg{args.guidance_scale}'
    prev_run_dirs = []
    if os.path.isdir(args.output_dir):
        prev_run_dirs = [x for x in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.project_dir = os.path.join(args.output_dir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(args.project_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.project_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Handle the repository creation
    if args.project_dir is not None:
        os.makedirs(args.project_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    accelerator.wait_for_everyone()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.project_dir, "train.log")),
        ]
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet"
        )

    sharpener = Sharpener(input_dim=768, output_dim=768, inner_dim=512, nhead=8)
    sharpener.train().requires_grad_(True)
    # Check model parameters
    total_params = 0
    for param in sharpener.parameters():
        total_params += param.numel()

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
            foreach=args.foreach_ema,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                # model.save_pretrained(os.path.join(output_dir, "unet"))
                data = dict(model=accelerator.unwrap_model(sharpener))

                with open(os.path.join(output_dir, f'sharpener-snapshot.pkl'), 'wb') as f:
                    pickle.dump(data, f)

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel, foreach=args.foreach_ema
                )
                ema_unet.load_state_dict(load_model.state_dict())
                if args.offload_ema:
                    ema_unet.pin_memory()
                else:
                    ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        sharpener.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    dataset_obj = COCODataset('../data/COCO/train2017', annPath='../data/COCO/annotations_trainval2017/captions_train2017.json', res=512)
    dataset_sampler = InfiniteSampler(dataset=dataset_obj, rank=accelerator.process_index, num_replicas=accelerator.num_processes, seed=args.seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers))

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(dataset_obj) / (accelerator.num_processes * args.train_batch_size))
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding * args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    sharpener, optimizer, dataset_iterator, lr_scheduler = accelerator.prepare(
        sharpener, optimizer, dataset_iterator, lr_scheduler
    )

    if args.use_ema:
        if args.offload_ema:
            ema_unet.pin_memory()
        else:
            ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move modules to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    sharpener.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataset_obj) * args.gradient_accumulation_steps / (accelerator.num_processes * args.train_batch_size))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'dataset_obj' after 'accelerator.prepare' ({len(dataset_obj)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    # Logger(file_name=os.path.join(args.project_dir, 'log.txt'), file_mode='a', should_flush=True)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset_obj)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total parameters of DICE sharpener = {total_params}")
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.project_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.project_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Get the null text embedding
    with torch.no_grad():
        uc = text_encoder(tokenize_captions([''] * args.train_batch_size).to(accelerator.device), return_dict=False)[0]
    
    rg = RandomGenerator()
    
    if accelerator.is_main_process:
        with torch.no_grad():
            log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, sharpener, global_step)

    # Training loop
    train_loss = 0.0
    while True:
        images, captions = next(dataset_iterator)
        images = images.to(accelerator.device) / 127.5 - 1
        batch = {'pixel_values': images, 'input_ids': tokenize_captions(captions).to(accelerator.device)}

        # Sample a random timestep for each step
        if global_step % args.gradient_accumulation_steps == 0:
            timesteps = torch.tensor(
                next(rg.randint(0, noise_scheduler.config.num_train_timesteps-1)), 
                device=accelerator.device
            ).reshape(-1,).repeat(args.train_batch_size,)
            timesteps = timesteps.long()

        with accelerator.accumulate(sharpener):
            with torch.no_grad():
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                c = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Predict the noise residual and compute loss
                with torch.autocast('cuda'):
                    target = unet(torch.cat([noisy_latents] * 2), torch.cat([timesteps] * 2), torch.cat([uc, c]), return_dict=False)[0]
                    target_uncond, target_text = target.chunk(2)
                    target = target_uncond + args.guidance_scale * (target_text - target_uncond)

            # Get the enhanced text embedding
            c_learn = c + sharpener(c, uc)
            c_learn[:,0] = c[:,0]       # Keep the <SOS> token unchanged

            with torch.autocast('cuda'):
                # Compute loss
                model_pred = unet(noisy_latents, timesteps, c_learn, return_dict=False)[0]
                loss = (model_pred - target).abs().sum().mul(1 / total_batch_size)
                # loss = ((model_pred - target)**2).sum().mul(1 / total_batch_size)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

            optimizer.zero_grad()

        if accelerator.is_main_process:
            logger.info(f"Global: {global_step+1}/{args.max_train_steps} | Loss: {avg_loss.item():8.4f}")
        
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if args.use_ema:
                if args.offload_ema:
                    ema_unet.to(device="cuda", non_blocking=True)
                ema_unet.step(unet.parameters())
                if args.offload_ema:
                    ema_unet.to(device="cpu", non_blocking=True)
            progress_bar.update(1)
            accelerator.log({"train_loss": train_loss}, step=global_step)
            global_step += 1
            train_loss = 0.0

            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.project_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.project_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.project_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

                with torch.no_grad():
                    log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, sharpener, global_step)
        
        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()

#----------------------------------------------------------------------------

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, sharpener, global_step):
    sharpener.eval()
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    # Save images
    args.validation_prompts = [
        'Photo portrait of a girl, sunshine',
        'A beautiful landscape, mountain, cloud, lake',
        'A corgi wearing sunglasses on the beach',
        'A sports car running on the road'
    ]
    generator = torch.Generator(device=accelerator.device).manual_seed(0)
    os.makedirs(os.path.join(args.project_dir, 'images'), exist_ok=True)

    with torch.autocast('cuda'):
        if global_step == 0:
            images = pipeline(args.validation_prompts, num_inference_steps=10, num_images_per_prompt=2, generator=generator, guidance_scale=args.guidance_scale).images
            images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
            images = make_grid(images, nrow=4, padding=0)
            save_image(images, os.path.join(args.project_dir, 'images', f"grid_cfg{args.guidance_scale}.png"))

            images = pipeline(args.validation_prompts, num_inference_steps=10, num_images_per_prompt=2, generator=generator, guidance_scale=1).images
            images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
            images = make_grid(images, nrow=4, padding=0)
            save_image(images, os.path.join(args.project_dir, 'images', f"grid_cfg1.png"))
        else:
            pipeline.sharpener = sharpener
            pipeline.sharpener_alpha = 1
            images = pipeline(args.validation_prompts, num_inference_steps=10, num_images_per_prompt=2, generator=generator).images
            images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
            images = make_grid(images, nrow=4, padding=0)
            save_image(images, os.path.join(args.project_dir, 'images', f"grid_dice_{global_step}.png"))

    del pipeline
    torch.cuda.empty_cache()

    sharpener.train()
    return images


if __name__ == "__main__":
    main()