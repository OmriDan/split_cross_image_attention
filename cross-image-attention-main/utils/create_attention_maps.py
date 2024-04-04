import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch
"""
Source:
https://github.com/yuval-alaluf/Attend-and-Excite/blob/163efdfd341bf3590df3c0c2b582935fbc8e8343/utils/vis_utils.py
"""

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def create_maps(attn_weight):
    ma = attn_weight[0].mean(dim=0)
    point = (20, 17)
    out_path = 'notebooks/animal/app=monkey---struct=koala/out_transfer---seed_42.png'
    style_img_path ='notebooks/inputs/monkey.png'
    struct_img_path = 'notebooks/inputs/koala.png'
    ma = ma[32 * point[1] + point[0]].reshape(32, 32)
    ma = (ma - torch.min(ma)) / (torch.max(ma) - torch.min(ma))
    image = show_image_relevance(ma, Image.open(out_path).convert('RGB'))
    Image.fromarray(image).save('out.png')
    ma = attn_weight[1].mean(dim=0)
    ma = ma[32 * point[1] + point[0]].reshape(32, 32)
    ma = (ma - torch.min(ma)) / (torch.max(ma) - torch.min(ma))
    image = show_image_relevance(ma, Image.open(style_img_path).convert('RGB'), relevnace_res=32)
    Image.fromarray(image).save('style.png')
    ma = attn_weight[2].mean(dim=0)
    ma = ma[32 * point[1] + point[0]].reshape(32, 32)
    ma = (ma - torch.min(ma)) / (torch.max(ma) - torch.min(ma))
    image = show_image_relevance(ma, Image.open(struct_img_path).convert('RGB'))
    Image.fromarray(image).save('struct.png')
    print('finished!')
