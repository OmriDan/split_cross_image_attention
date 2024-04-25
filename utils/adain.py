import torch


def masked_adain_half_mask(content_feat, style1_feat, style2_feat, content_mask, style1_mask, style2_mask):
    assert ((content_feat.size()[:2] == style1_feat.size()[:2]) and (content_feat.size()[:2] == style2_feat.size()[:2]))
    size = content_feat.size()

    # Calculate the mean and standard deviation for style and content features with respective masks
    style1_mean, style1_std = calc_mean_std(style1_feat, mask=style1_mask)
    style2_mean, style2_std = calc_mean_std(style2_feat, mask=style2_mask)
    content_mean, content_std = calc_mean_std(content_feat, mask=content_mask)

    # Normalize the content features and apply style statistics
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    style1_normalized_feat = normalized_feat * style1_std.expand(size) + style1_mean.expand(size)
    style2_normalized_feat = normalized_feat * style2_std.expand(size) + style2_mean.expand(size)

    # Split the content_mask into left and right sides
    mid_index = content_mask.shape[1] // 2
    left_half = content_mask[:, :mid_index]
    right_half = content_mask[:, mid_index:]

    # Create new masks with zero padding to fill up to the original dimensions
    left_mask = torch.cat((left_half, torch.zeros_like(left_half)), dim=1)
    right_mask = torch.cat((torch.zeros_like(right_half), right_half), dim=1)

    # Combine the features based on the original mask
    combined_feat = content_feat * (
                1 - content_mask) + style1_normalized_feat * left_mask + style2_normalized_feat * right_mask
    return combined_feat


def masked_adain(content_feat, style1_feat, style2_feat, content_mask, style1_mask, style2_mask):
    assert ((content_feat.size()[:2] == style1_feat.size()[:2]) and (content_feat.size()[:2] == style2_feat.size()[:2]))
    size = content_feat.size()
    style1_mean, style1_std = calc_mean_std(style1_feat, mask=style1_mask)
    style2_mean, style2_std = calc_mean_std(style2_feat, mask=style2_mask)
    content_mean, content_std = calc_mean_std(content_feat, mask=content_mask)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    style1_normalized_feat = normalized_feat * style1_std.expand(size) + style1_mean.expand(size)
    style2_normalized_feat = normalized_feat * style2_std.expand(size) + style2_mean.expand(size)
    #print(content_mask)
    #print(f"content_mask shape: {content_mask.shape} ")
    #print(f"content_mask type: {type(content_mask)} ")
    return content_feat * (1 - content_mask) + style1_normalized_feat * content_mask  #  style2_normalized_feat

def adain(content_feat, style1_feat, style2_feat):
    assert ((content_feat.size()[:2] == style1_feat.size()[:2]) and (content_feat.size()[:2] == style2_feat.size()[:2]))
    size = content_feat.size()
    style1_mean, style1_std = calc_mean_std(style1_feat)
    style2_mean, style2_std = calc_mean_std(style2_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style1_std.expand(size) + style1_mean.expand(size)# + normalized_feat * style2_std.expand(size)  + style2_mean.expand(size)# TODO: add style2


def calc_mean_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    if len(size) == 2:
        return calc_mean_std_2d(feat, eps, mask)

    assert (len(size) == 3)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)

    return feat_mean, feat_std


def calc_mean_std_2d(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 2)
    C = size[0]
    if mask is not None:
        feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1)
    else:
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(C, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)

    return feat_mean, feat_std
