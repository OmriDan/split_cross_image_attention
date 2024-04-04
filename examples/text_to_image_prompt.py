from diffusers import AutoPipelineForText2Image
from matplotlib import pyplot as plt
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

generator = torch.Generator(device="cuda").manual_seed(12)
image = pipeline(
"Zebra in times square, 8k",generator=generator, guidance_scale=4.5).images[0]
plt.imshow(image)
plt.show()
