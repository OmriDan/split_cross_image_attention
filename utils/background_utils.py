import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from matplotlib import pyplot as plt

def generate_initial_image(prompt, seed=12):
    text2image_pipeline = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = text2image_pipeline(prompt, generator=generator, guidance_scale=4.5).images[0]
    return image

def modify_image(initial_image, modification_prompt, seed=10):
    image2image_pipeline = AutoPipelineForImage2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    image2image_pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    modified_image = image2image_pipeline(
        modification_prompt, image=initial_image, strength=0.75, num_inference_steps=1000, generator=generator).images[0]
    return modified_image

def display_images(initial_image, modified_image):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(initial_image)
    axes[0].set_title('Initial Image')
    axes[0].axis('off')
    axes[1].imshow(modified_image)
    axes[1].set_title('Modified Image')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

'''

# Using the functions to generate and modify an image
initial_prompt = "A scenic view of the mountains at sunset"
modification_prompt = "a clearer sky"

initial_image = generate_initial_image(initial_prompt)
modified_image = modify_image(initial_image, modification_prompt)

display_images(initial_image, modified_image)
'''