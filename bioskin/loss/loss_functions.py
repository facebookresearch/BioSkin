# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def MSE(output, target, weight):
    loss = torch.mean((output - target)**2)
    return loss


def L1(output, target, weight):
    loss = torch.mean(torch.abs(output - target))
    return loss


def Luv_approx(output, target, weight):
    output_r = output[:, 0]
    output_g = output[:, 1]
    output_b = output[:, 2]

    target_r = target[:, 0]
    target_g = target[:, 1]
    target_b = target[:, 2]

    rmean = 0.5 * (output_r + target_r)
    deltaR = output_r - target_r
    deltaG = output_g - target_g
    deltaB = output_b - target_b
    distances = (((2 + rmean) * deltaR * deltaR + 4 * deltaG * deltaG + (2 + (1 - rmean)) * deltaB * deltaB) / 10.0) ** 0.5
    loss = torch.mean(distances)
    return loss


def Luv_approx_regularized(output, target, weight):

    output_r = output[:, 0]
    output_g = output[:, 1]
    output_b = output[:, 2]

    target_r = target[:, 0]
    target_g = target[:, 1]
    target_b = target[:, 2]

    rmean = 0.5 * (output_r + target_r)
    deltaR = output_r - target_r
    deltaG = output_g - target_g
    deltaB = output_b - target_b

    distances = (((2 + rmean) * deltaR * deltaR + 4 * deltaG * deltaG
                  + (2 + (1 - rmean)) * deltaB * deltaB) / 10.0) ** 0.5
    loss = torch.mean(distances)
    loss += MSE(output, target, weight)
    return loss


def weighted_RGB_euclidean(output, target, weight):
    output_r = output[:, 0]
    output_g = output[:, 1]
    output_b = output[:, 2]

    target_r = target[:, 0]
    target_g = target[:, 1]
    target_b = target[:, 2]

    deltaR = output_r - target_r
    deltaG = output_g - target_g
    deltaB = output_b - target_b
    distances = ((3*deltaR*deltaR + 4*deltaG*deltaG + 2*deltaB*deltaB)/9.0) ** 0.5
    loss = torch.mean(distances)
    return loss


def mixed_luv_l1(output, target, weight):
    l1_loss = torch.mean(torch.abs(output - target))
    return weight * Luv_approx(output, target, weight) + (1.0 - weight) * l1_loss


def mixed_luv_l2(output, target, weight):
    l2_loss = torch.mean((output - target)**2)
    return weight * Luv_approx(output, target, weight) + (1.0 - weight) * l2_loss


def sam_spectral_loss(output, target, weight=1.0):
    numerator = torch.sum(output * target, 1)
    denominator = torch.sqrt(torch.sum(torch.square(output), 1) * torch.sum(torch.square(target), 1))
    return torch.mean(torch.arccos(numerator / denominator))


def sam_spectral_loss2(output, target, weight=1.0):
    numerator = torch.sum(output * target, 1)
    denominator = torch.sqrt(torch.sum(torch.square(output), 1)) * torch.sqrt(torch.sum(torch.square(target), 1))
    return torch.mean(torch.arccos(numerator / denominator))


# sequential version
def sam_spectral_distance(output, target, weight=1.0):
    dot_prod = torch.dot(output, target)
    norms = torch.mul(torch.norm(output), torch.norm(target))
    x = torch.div(dot_prod, norms)
    return torch.arccos(x)


def sam_spectral_loss_sequential(output, target, weight=1.0):
    sam_distances = torch.zeros(output.shape[0])
    for i in range(0, output.shape[0]):
        sam_distances = sam_spectral_distance(output[i, :], target[i, :])
    loss = torch.mean(sam_distances)
    return loss


def perceptual_luminance(rgb):
    R = rgb[:, 0]
    G = rgb[:, 1]
    B = rgb[:, 2]
    luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return luminance
