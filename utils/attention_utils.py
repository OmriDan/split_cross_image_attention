import math
import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt

from constants import *


def should_mix_keys_and_values(model, hidden_states: torch.Tensor) -> bool:
    """ Verify whether we should perform the mixing in the current timestep. """
    is_in_32_timestep_range = (
            model.config.cross_attn_32_range.start <= model.step < model.config.cross_attn_32_range.end
    )
    is_in_64_timestep_range = (
            model.config.cross_attn_64_range.start <= model.step < model.config.cross_attn_64_range.end
    )
    is_hidden_states_32_square = (hidden_states.shape[1] == 32 ** 2)
    is_hidden_states_64_square = (hidden_states.shape[1] == 64 ** 2)
    should_mix = (is_in_32_timestep_range and is_hidden_states_32_square) or \
                 (is_in_64_timestep_range and is_hidden_states_64_square)
    return should_mix


def compute_scaled_dot_product_attention(Q, K, V, edit_map=False, is_cross=False, contrast_strength=1.0):
    """ Compute the scale dot product attention, potentially with our contrasting operation. """
    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    if edit_map and not is_cross:
        attn_weight[OUT_INDEX] = torch.stack([
            torch.clip(enhance_tensor(attn_weight[OUT_INDEX][head_idx], contrast_factor=contrast_strength),
                       min=0.0, max=1.0)
            for head_idx in range(attn_weight.shape[1])
        ])
    return attn_weight @ V, attn_weight


def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)
    return adjusted_tensor


def compute_attention(Q, K, V, is_cross, split_attn, edit_map, model_self):
    if not split_attn:
        hidden_states, attn_weight = compute_scaled_dot_product_attention(Q, K, V, edit_map=edit_map,
                                                                          is_cross=is_cross, contrast_strength=model_self.config.contrast_strength)
    else:
        hidden_states, attn_weight = split_attention(Q,K,V,masks=load_masks(res=Q.shape[2]))
    return hidden_states, attn_weight



def split_attention(query, key, value, masks, edit_map=False, is_cross=False, contrast_strength=1.0):
    # calculate all but OUT_INDEX attention as usual
    query_without_out, key_without_out, value_without_out = query[OUT_INDEX+1:], key[OUT_INDEX+1:], value[OUT_INDEX+1:]
    attn_weight_without_out = torch.softmax((query_without_out @ key_without_out.transpose(-2, -1) / math.sqrt(query_without_out.size(-1))), dim=-1)
    hidden_state_without_out = attn_weight_without_out @ value_without_out


    # Get the binary masks of the objects in the image
    (binary_struct_masks, inv_binary_struct_masks, binary_mask_appearance1, binary_mask_appearance2,
     inv_binary_mask_appearance1, inv_binary_mask_appearance2) = masks

    query_out, key_out, v_out = query[OUT_INDEX], key[OUT_INDEX], value[OUT_INDEX]
    query_out, key_out, v_out = query_out.unsqueeze(0), key_out.unsqueeze(0), v_out.unsqueeze(0)
    query_object1 = torch.zeros_like(query_out)
    query_object2 = torch.zeros_like(query_out)
    struct_mask1, struct_mask2 = binary_struct_masks
    inv_struct_mask1, inv_struct_mask2 = inv_binary_struct_masks

    if query_out.ndim == 4:
        mid_index = query_out.shape[2] // 2
    else:
        raise Exception

    # Splitting the query and the binary masks to 2 objects
    query_object1 = query_out * inv_struct_mask2
    query_object1[query_object1 == 0] = -float("Inf")
    query_object2 = query_out * inv_struct_mask1 # Taking all the query vals: # query_out[:, :, mid_index:, :]
    query_object2[query_object2 == 0] = -float("Inf")

    # Using k,v from style 1 on object 1 might be key[mask] = OUT/VAL
    key_out1 = key[STYLE1_INDEX]  # adding k of style1 maybe unsqueeze?
    value_out1 = value[STYLE1_INDEX] # adding v of style1
    # Using k,v from style 2 on object 2
    key_out2 = key[STYLE2_INDEX]  # adding k of style2
    value_out2 = value[STYLE2_INDEX]  # adding v of style2
    # Attention calculation for object 1
    attn_weight1 = torch.softmax(
        (query_object1 @ key_out1.transpose(-2, -1) / math.sqrt(query_object1.size(-1))), dim=-1)
    hidden_state1 = attn_weight1 @ value_out1
    # Attention calculation for object 2
    attn_weight2 = torch.softmax(
        (query_object2 @ key_out2.transpose(-2, -1) / math.sqrt(query_object2.size(-1))), dim=-1)
    hidden_state2 = attn_weight2 @ value_out2

    attn_weight_out = attn_weight1 + attn_weight2
    hidden_state_out = hidden_state1 + hidden_state2
    hidden_states = torch.cat((hidden_state_out, hidden_state_without_out), dim=0)
    attn_weight = torch.cat((attn_weight_out, attn_weight_without_out), dim=0)

    if edit_map and not is_cross:
        attn_weight[OUT_INDEX] = torch.stack([
            torch.clip(enhance_tensor(attn_weight[OUT_INDEX][head_idx], contrast_factor=contrast_strength),
                       min=0.0, max=1.0)
            for head_idx in range(attn_weight.shape[1])
        ])

    return hidden_states, attn_weight


def load_masks(res):
    convert_tensor = transforms.ToTensor()
    mask_style1_32 = convert_tensor(np.load("masks/cross_attention_style1_mask_resolution_32.npy"))
    mask_style2_32 = convert_tensor(np.load("masks/cross_attention_style2_mask_resolution_32.npy"))
    mask_struct_32 = convert_tensor(np.load("masks/cross_attention_structural_mask_resolution_32.npy"))
    mask_style1_64 = convert_tensor(np.load("masks/cross_attention_style1_mask_resolution_64.npy"))
    mask_style2_64 = convert_tensor(np.load("masks/cross_attention_style2_mask_resolution_64.npy"))
    mask_struct_64 = convert_tensor(np.load("masks/cross_attention_structural_mask_resolution_64.npy"))
    if res == 32 ** 2:
        binary_mask_struct, binary_mask_appearance1, binary_mask_appearance2 =\
            mask_struct_32.squeeze(), mask_style1_32.squeeze(), mask_style2_32.squeeze()
    elif res == 64 ** 2:
        binary_mask_struct, binary_mask_appearance1, binary_mask_appearance2 =\
            mask_struct_64.squeeze(), mask_style1_64.squeeze(), mask_style2_64.squeeze()
    else:
        return
    mid_struct_idx = int(np.sqrt(res) // 2)

    inv_binary_mask_struct = (~binary_mask_struct + 2)

    struct_mask1 = torch.zeros_like(binary_mask_struct)
    struct_mask2 = torch.zeros_like(binary_mask_struct)
    struct_mask1[:,:mid_struct_idx] = binary_mask_struct[:,:mid_struct_idx]
    struct_mask2[:,mid_struct_idx:] = binary_mask_struct[:,mid_struct_idx:]

    struct_mask1 = struct_mask1.flatten().view(-1,1).to('cuda')
    struct_mask2 = struct_mask2.flatten().view(-1,1).to('cuda')
    binary_mask_appearance1 = binary_mask_appearance1.squeeze().flatten().view(-1, 1).to('cuda')
    binary_mask_appearance2 = binary_mask_appearance2.squeeze().flatten().view(-1, 1).to('cuda')

    inv_struct_mask1 = (~struct_mask1 + 2).flatten().view(-1, 1).to('cuda')
    inv_struct_mask2 = (~struct_mask2 + 2).flatten().view(-1, 1).to('cuda')
    inv_binary_mask_appearance1 = (~binary_mask_appearance1 + 2).view(-1, 1).to('cuda')
    inv_binary_mask_appearance2 = (~binary_mask_appearance2 + 2).view(-1, 1).to('cuda')
    binary_struct_masks = [struct_mask1, struct_mask2]

    inv_binary_struct_masks = [inv_struct_mask1, inv_struct_mask2]
    return binary_struct_masks, inv_binary_struct_masks, binary_mask_appearance1, binary_mask_appearance2, inv_binary_mask_appearance1, inv_binary_mask_appearance2
