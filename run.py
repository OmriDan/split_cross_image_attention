# Import necessary modules
import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed

# Add current and parent directories to system path for module import
sys.path.append(".")
sys.path.append("..")


# Import custom modules for model and utilities
from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images


# Configure the script to use configuration objects defined by Pyrallis
@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)


# Main function to set up and execute the appearance transfer
def run(cfg: RunConfig) -> List[Image.Image]:
    # Save the configuration to a YAML file in the output directory
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    # Set the random seed for reproducibility
    set_seed(cfg.seed)
    # Initialize the appearance transfer model with the configuration
    model = AppearanceTransferModel(cfg)
    # Load or compute latents and structural noises as per the model requirements
    latents_app1, latents_app2, latents_struct, input_app1, input_app2, noise_struct =\
        load_latents_or_invert_images(model=model, cfg=cfg)
    # Set model latents and noise for appearance transfer
    model.set_latents(latents_app1, latents_app2, latents_struct)
    model.set_noise(input_app1, input_app2, noise_struct)
    print("Running appearance transfer...")
    # Execute the appearance transfer process
    images = run_appearance_transfer(model=model, cfg=cfg)
    print("Done.")
    return images


# Function to conduct the appearance transfer using the configured model
def run_appearance_transfer(model: AppearanceTransferModel, cfg: RunConfig) -> List[Image.Image]:
    # Retrieve initial latents and noise configurations
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    # Configure the scheduler with the number of time steps for the transfer
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    # Enable editing features for cross-image effects
    model.enable_edit = True
    # Determine the start and end steps for cross-image attention
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
    # Perform the model's appearance transfer process
    images = model.pipe(
        prompt=[cfg.prompt] * 4,
        latents=init_latents,
        guidance_scale=1.0,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator('cuda').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
    ).images
    # Save output images individually and then join and save them as one image
    images[0].save(cfg.output_path / f"out_transfer---seed_{cfg.seed}.png")
    images[1].save(cfg.output_path / f"out_style1---seed_{cfg.seed}.png")
    images[2].save(cfg.output_path / f"out_style2---seed_{cfg.seed}.png")
    images[3].save(cfg.output_path / f"out_struct---seed_{cfg.seed}.png")
    joined_images = np.concatenate(images[::-1], axis=1)
    Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}.png")
    return images


# Entry point check to run the main function
if __name__ == '__main__':
    main()
