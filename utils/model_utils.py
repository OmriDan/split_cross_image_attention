import torch
from diffusers import DDIMScheduler

from models.stable_diffusion import CrossImageAttentionStableDiffusionPipeline
from models.unet_2d_condition import FreeUUNet2DConditionModel


def get_stable_diffusion_model() -> CrossImageAttentionStableDiffusionPipeline:
    print("Loading Stable Diffusion model...")
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = CrossImageAttentionStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                      safety_checker=None,resume_download=True).to(device)
    pipe.unet = FreeUUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", resume_download=True).to(device)
    pipe.scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler", resume_download=True)
    #pipe.enable_model_cpu_offload() # For Model offloading
    #pipe.enable_sequential_cpu_offload() # For CPU offloading
    #pipe.enable_vae_slicing() # For Sliced  VAE
    #pipe.enable_xformers_memory_efficient_attention() # For Sliced VAE (optional) OR by itself
    print("Done.")
    return pipe
