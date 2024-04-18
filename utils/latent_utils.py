from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion import invert


def load_latents_or_invert_images(model: AppearanceTransferModel, cfg: RunConfig):
    if cfg.load_latents and cfg.app1_latent_save_path.exists() and cfg.struct_latent_save_path.exists():
        print("Loading existing latents...")
        latents_app1, latents_app2, latents_struct = load_latents(cfg.app1_latent_save_path, cfg.app2_latent_save_path,
                                                                  cfg.struct_latent_save_path)
        noise_app1, noise_app2, noise_struct = load_noise(cfg.app1_latent_save_path, cfg.app2_latent_save_path,
                                                          cfg.struct_latent_save_path)
        print("Done.")
    else:
        print("Inverting images...")
        app1_image, app2_image, struct_image = image_utils.load_images(cfg=cfg, save_path=cfg.output_path)
        model.enable_edit = False  # Deactivate the cross-image attention layers
        latents_app1, latents_app2, latents_struct, noise_app1, noise_app2, noise_struct = invert_images(
            app1_image=app1_image,
            app2_image=app2_image,
            struct_image=struct_image,
            sd_model=model.pipe,
            cfg=cfg)
        model.enable_edit = True
        print("Done.")
    return latents_app1, latents_app2, latents_struct, noise_app1, noise_app2, noise_struct


def load_latents(app1_latent_save_path: Path, app2_latent_save_path: Path, struct_latent_save_path: Path) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    latents_app1 = torch.load(app1_latent_save_path)
    latents_app2 = torch.load(app2_latent_save_path)
    latents_struct = torch.load(struct_latent_save_path)
    if type(latents_struct) == list:
        latents_app1 = [l.to("cuda") for l in latents_app1]
        latents_app2 = [l.to("cuda") for l in latents_app2]
        latents_struct = [l.to("cuda") for l in latents_struct]
    else:
        latents_app1 = latents_app1.to("cuda")
        latents_app2 = latents_app2.to("cuda")
        latents_struct = latents_struct.to("cuda")
    return latents_app1, latents_app2, latents_struct


def load_noise(app1_latent_save_path: Path, app2_latent_save_path: Path, struct_latent_save_path: Path) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    latents_app1 = torch.load(app1_latent_save_path.parent / (app1_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_app2 = torch.load(app2_latent_save_path.parent / (app2_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_struct = torch.load(struct_latent_save_path.parent / (struct_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_app1 = latents_app1.to("cuda")
    latents_app2 = latents_app2.to("cuda")
    latents_struct = latents_struct.to("cuda")
    return latents_app1, latents_app2, latents_struct


def invert_images(sd_model: AppearanceTransferModel, app1_image: Image.Image, app2_image: Image.Image,
                  struct_image: Image.Image, cfg: RunConfig):
    input_app1 = torch.from_numpy(np.array(app1_image)).float() / 127.5 - 1.0
    input_app2 = torch.from_numpy(np.array(app2_image)).float() / 127.5 - 1.0
    input_struct = torch.from_numpy(np.array(struct_image)).float() / 127.5 - 1.0
    zs_app1, latents_app1 = invert(x0=input_app1.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                   pipe=sd_model,
                                   prompt_src=cfg.prompt,
                                   num_diffusion_steps=cfg.num_timesteps,
                                   cfg_scale_src=3.5)
    zs_app2, latents_app2 = invert(x0=input_app2.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                   pipe=sd_model,
                                   prompt_src=cfg.prompt,
                                   num_diffusion_steps=cfg.num_timesteps,
                                   cfg_scale_src=3.5)
    zs_struct, latents_struct = invert(x0=input_struct.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                       pipe=sd_model,
                                       prompt_src=cfg.prompt,
                                       num_diffusion_steps=cfg.num_timesteps,
                                       cfg_scale_src=3.5)
    # Save the inverted latents and noises
    torch.save(latents_app1, cfg.latents_path / f"{cfg.app1_image_path.stem}.pt")
    torch.save(latents_app2, cfg.latents_path / f"{cfg.app2_image_path.stem}.pt")
    torch.save(latents_struct, cfg.latents_path / f"{cfg.struct_image_path.stem}.pt")
    torch.save(zs_app1, cfg.latents_path / f"{cfg.app1_image_path.stem}_ddpm_noise.pt")
    torch.save(zs_app2, cfg.latents_path / f"{cfg.app2_image_path.stem}_ddpm_noise.pt")
    torch.save(zs_struct, cfg.latents_path / f"{cfg.struct_image_path.stem}_ddpm_noise.pt")
    return latents_app1, latents_app2, latents_struct, zs_app1, zs_app2, zs_struct


def get_init_latents_and_noises(model: AppearanceTransferModel, cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    if model.latents_struct.dim() == 4 and model.latents_app1.dim() == 4 and model.latents_app1.shape[0] > 1:
        model.latents_struct = model.latents_struct[cfg.skip_steps]
        model.latents_app1 = model.latents_app1[cfg.skip_steps]
        model.latents_app2 = model.latents_app2[cfg.skip_steps]
    init_latents = torch.stack([model.latents_struct, model.latents_app1, model.latents_app2, model.latents_struct])
    init_zs = [model.zs_struct[cfg.skip_steps:], model.zs_app1[cfg.skip_steps:], model.zs_app2[cfg.skip_steps:],
               model.zs_struct[cfg.skip_steps:]]
    return init_latents, init_zs
