# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
from torch import unique
import torch.nn.functional as F


def gaussian_blur(input_tensor, kernel_size, sigma, channels):
    padding = kernel_size // 2

    # Create Gaussian kernel
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Shape kernel for convolution
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    gaussian_kernel = gaussian_kernel.to(input_tensor.device)
    return F.conv2d(input_tensor, gaussian_kernel, padding=padding, groups=channels)


def compute_mode(tensor_2d):
    flattened_tensor = tensor_2d.view(-1)
    unique_values, counts = unique(flattened_tensor, return_counts=True)
    mode_index = counts.argmax()
    mode = unique_values[mode_index]
    return mode


def smooth(img, w, mean=None):
    mask = (img < 0.95) & (img > 0.001)
    if mean:
        img_mean = mean
    else:
        img_mean = torch.mean(img[mask])
    new_image = (1 - w) * img + w * img_mean
    return new_image


def compute_cdf(img):
    num_bins = 1024
    hist_img = torch.histc(img, bins=num_bins, min=0, max=1)
    total_pixels = img.numel()
    pixels_per_bin = total_pixels / num_bins
    cdf_equalized = torch.cumsum(hist_img, dim=0)
    cdf_equalized = cdf_equalized - (pixels_per_bin / 2)  # Shift to the middle of each bin
    cdf_equalized = cdf_equalized / total_pixels  # Normalize to [0, 1]
    return cdf_equalized


def equalize_hist(img, cdf):
    img_equalized = cdf[torch.floor(img * 1024).long()]
    img_equalized = torch.clamp(img_equalized, 0, 1)
    img_equalized = img_equalized.view(img.shape)
    return img_equalized


def enhance(img, w, img_ref=None):
    if img_ref:
        cdf = compute_cdf(img_ref)
    else:
        cdf = compute_cdf(img)
    img_equalized = equalize_hist(img, cdf)
    wp = max(0, w - 0.5)
    img_equalized = wp * img_equalized + (1 - wp) * img
    return img_equalized


def fill_lowest_values(img, x, min=None):
    if min:
        img_min = min
    else:
        img_min = torch.min(img)
    threshold = img_min + x
    mask = img < threshold
    img[mask] = threshold
    img = torch.clamp(img, 0, 1)
    return img


def flatten_peaks(img, x, max=None):
    if max:
        img_max = max
    else:
        img_max = torch.max(img)
    threshold = img_max - abs(x)
    mask = img > threshold
    img[mask] = threshold
    img = torch.clamp(img, 0, 1)
    return img

def offset(img, x, max=None):
    if max:
        img_max = max
    else:
        img_max = torch.mean(img)
    if x < 0:
        x *= img_max
    img += x
    img = torch.clamp(img, 0, 1)
    return img

def multiply(parameter_map, x, mean=None):
    # multiplies to even the average value to the value coming from the slider from -1 to 1
    if mean:
        mean_value = mean
    else:
        mean_value = torch.mean(parameter_map)

    if x >= 0.0:
        parameter_map *= (1.0 + x*(1.0/mean_value))
    else:
        parameter_map /= (1.0 + abs(x)*(1.0/mean_value))
    parameter_map = torch.clamp(parameter_map, 0.0, 1.0)
    return parameter_map


def multiply_raw(parameter_map, x):
    parameter_map *= x
    parameter_map = torch.clamp(parameter_map, 0.0, 1.0)
    return parameter_map


def blur(parameter_map, value):
    kernel_size = np.round(3.0 + 10 * value)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = int(kernel_size)
    sigma = 0.01 + value * 5.0
    input_tensor_4d = parameter_map.unsqueeze(0).unsqueeze(0)
    output_tensor_4d = gaussian_blur(input_tensor_4d, kernel_size=kernel_size, sigma=sigma, channels=1)
    parameter_map = output_tensor_4d.squeeze(0).squeeze(0)
    return parameter_map


def apply_operator(parameter_map, operation, value, parameter=None):
    if operation == "multiply":
        parameter_map = multiply(parameter_map, value, parameter.mean() if parameter else None)
    elif operation == "multiply_raw":
        parameter_map = multiply_raw(parameter_map, value)
    elif operation == "smooth":
        parameter_map = smooth(parameter_map, value, parameter.mean_masked() if parameter else None)
    elif operation == "enhance":
        parameter_map = enhance(parameter_map, value, parameter.map_edited if parameter else None)
    elif operation == "fill_valleys":
        parameter_map = fill_lowest_values(parameter_map, value, parameter.min() if parameter else None)
    elif operation == "flatten_peaks":
        parameter_map = flatten_peaks(parameter_map, value, parameter.max() if parameter else None)
    elif operation == "offset":
        parameter_map = offset(parameter_map, value, parameter.mean_masked() if parameter else None)
    elif operation == "blurr":
        parameter_map = blur(parameter_map, value)
    else:
        print("Operator unknown!")
    return parameter_map
        