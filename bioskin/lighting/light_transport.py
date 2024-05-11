# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import numpy as np
import bioskin.spectrum.color_spectrum as color_spectrum


def random_hemisphere_samples(num_samples, local_x, local_y, local_z):
    # Generate random angles
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    phi = np.random.uniform(0, np.pi / 2, num_samples)

    # Spherical to Cartesian
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Transform to local frame
    samples = np.array([x, y, z]).T
    transform_matrix = np.array([local_x, local_y, local_z]).T
    transformed_samples = samples @ transform_matrix

    return transformed_samples


def compute_irradiance(surface_normals, environment, num_env_samples=64):
    irradiance = np.zeros_like(surface_normals)
    h = environment.shape[0]
    w = environment.shape[1]

    for i in range(surface_normals.shape[0] - 1):
        for j in range(surface_normals.shape[1] - 1):
            surface_normal = 2.0 * surface_normals[i, j] - 1.0
            zenith = (0.5 * np.pi + np.arcsin(-surface_normal[0]))
            azimuth = np.pi + np.arctan2(surface_normal[1], surface_normal[2])
            if np.isnan(zenith):
                zenith = 0.0
            if np.isnan(azimuth):
                azimuth = 0.0

            zenith = np.clip(zenith, 0.0, np.pi) / np.pi
            azimuth = np.clip(azimuth, 0.0, 2 * np.pi) / (2 * np.pi)

            x = int(np.minimum(h-1, np.round(h * zenith)))
            y = int(np.minimum(w-1, np.round(w * azimuth)))

            irradiance[i, j] = environment[x, y]
    return irradiance


def fresnel_schlick(cos_theta, R0):
    return R0 + (1 - R0) * (1 - cos_theta).pow(5)


def integrate_fresnel(R0, num_samples):
    # Stratified sampling the zenith angle in the hemisphere of directions
    u1 = torch.linspace(0.0, 1.0 - 1 / num_samples, steps=num_samples)
    u1 += torch.rand(num_samples) / num_samples  # jitter
    cos_theta = 1 - u1
    sin_theta = torch.sqrt(1 - cos_theta * cos_theta)
    specular_reflectance = fresnel_schlick(cos_theta, torch.tensor([R0]))
    total_specular_reflectance = torch.sum(specular_reflectance * cos_theta * sin_theta)  # foreshortening & solid angle

    diffuse_reflectance = 1.0 - specular_reflectance
    diffuse_reflectance[diffuse_reflectance < 0.0] = 0.0
    total_diffuse_reflectance = torch.sum(diffuse_reflectance * cos_theta * sin_theta)

    total_reflectance = total_specular_reflectance + total_diffuse_reflectance
    total_specular_reflectance /= total_reflectance
    total_diffuse_reflectance /= total_reflectance
    return total_specular_reflectance, total_diffuse_reflectance


def add_specular_reflectance(spectrums, R0, num_hemisphere_samples, device):
    specular_reflectance, diffuse_reflectance = integrate_fresnel(R0, num_hemisphere_samples)
    spectrums_with_specular = diffuse_reflectance*spectrums + specular_reflectance

    Spectrum_handler = color_spectrum.ColorSpectrum(device, spectrums_with_specular.shape[1])
    skin_tones_with_specular = Spectrum_handler.spectrum_to_rgb(spectrums_with_specular)
    skin_tones_with_specular = torch.clamp(skin_tones_with_specular, min=0, max=1)

    return spectrums_with_specular, skin_tones_with_specular