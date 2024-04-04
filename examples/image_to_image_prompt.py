import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
from matplotlib import pyplot as plt

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()


# prepare image
init_image_path = "/home/omridan/msc/cross-image-attention/prompt_creations/Zebra in times square, 8k_seed12_guidance_scale4_5.png"
init_image = load_image(init_image_path)

prompt = "Zebra in the african savanna, photo"
generator = torch.Generator(device="cuda").manual_seed(10)

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.6, num_inference_steps=1000, generator=generator).images[0]

fig, axes = plt.subplots(1, 2)

axes[0].imshow(init_image)
axes[0].set_title('orig')
axes[0].axis('off')  # Turn off axis
axes[1].imshow(image)
axes[1].set_title('new background')
axes[1].axis('off')

plt.show()
