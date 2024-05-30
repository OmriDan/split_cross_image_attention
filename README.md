# Cross-Image Attention for Zero-Shot Appearance Transfer

> Omri Dan, Gilad Sivan
> Mentor: Dana Cohen
> Tel Aviv University  

>
> PROJECT ABSTRACT HERE

<a href="https://arxiv.org/abs/2311.03335"><img src="https://img.shields.io/badge/arXiv-2311.03335-b31b1b.svg" height=22.5></a>
<a href="https://garibida.github.io/cross-image-attention/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=20.5></a>
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yuvalalaluf/cross-image-attention)

<p align="center">
<img src="docs/teaser.jpg" width="90%"/>  
<br>
Given two images depicting a source structure and a target appearance, our method generates an image merging the structure of one image with the appearance of the other in a zero-shot manner.
</p>


## Description  
Official implementation of our SplitAttention mechanism.


## Environment
Our code builds on the requirement of the `diffusers` library. To set up their environment, please run:
```
conda env create -f environment/environment.yaml
conda activate cross_image
```

## Usage  
<p align="center">
<img src="docs/general_results.jpg" width="90%"/>  
<br>
Sample appearance transfer results obtained by our cross-image attention technique.
</p>

To generate an image, you can simply run the `run.py` script. For example,
```
python run.py \
--app1_image_path /path/to/appearance1/image.png \
--app2_image_path /path/to/appearance1/image.png \
--struct_image_path /path/to/structure/image.png \
--output_path /path/to/output/images.png \
--domain_name [domain the objects are taken from (e.g., animal, building)] \
--use_masked_adain True \
--contrast_strength 1.67 \
--swap_guidance_scale 3.5 \
```
Notes:
- To perform the inversion, if no prompt is specified explicitly, we will use the prompt `"A photo of a [domain_name]"`
- If `--use_masked_adain` is set to `True` (its default value), then `--domain_name` must be given in order 
  to compute the masks using the self-segmentation technique.
  - In cases where the domains are not well-defined, you can also set `--use_masked_adain` to `False` and 
    no `domain_name` is required.
- You can set `--load_latents` to `True` to load the latents from a file instead of inverting the input images every time. 
  - This is useful if you want to generate multiple images with the same structure but different appearances.


## Acknowledgements 
This code builds on the code of 
This code was a part of a Deep Learning course final project, under the guidance of Prof. Raja Geris.
We'd like to thank Dana Cohen, Daniel Garibi and Romario Zarik for their contribution.
This code builds on the code from the Cross-Image Attention for Zero-Shot Appearance Transfer paper [cross_image_attention]([https://github.com/huggingface/diffusers](https://github.com/garibida/cross-image-attention)).
```
