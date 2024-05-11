# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bioskin.parameters.parameter import MIN_PROP_VALUES
from bioskin.parameters.parameter import MAX_PROP_VALUES
from bioskin.utils.plotting import cm_parula
import bioskin.utils.io as io
from bioskin.parameters.parameter import SKIN_PROPS


thick_min = MIN_PROP_VALUES[2]
thick_max = MAX_PROP_VALUES[2]
melanin_exp = 3
hemoglobin_exp = 4


def save_parameter_maps(parameters, row, col, path, save_plot=True):
    # assumes linear skin properties as input
    parameters = warp_parameter_maps(parameters)
    for param in range(0, parameters.shape[1]):
        p = parameters[:, param]
        p = p.reshape(row, col)
        parameter_values = np.zeros((row, col, 3))
        parameter_values[:, :, 0] = p
        parameter_values[:, :, 1] = p
        parameter_values[:, :, 2] = p
        filename = path + '_p' + str(param) + '_' + SKIN_PROPS[param]
        parameter_image = parameter_values.astype("float32")
        cv2.imwrite(filename + '.exr', parameter_image)
        io.save_jpeg(filename, parameter_image, linear_input=True)
        if save_plot:
            max_value = 1.0
            if param == 2:
                max_value = thick_max
            plot_map(parameter_image, max_value, filename + '.jpg')


def plot_map(image, max_value, filename):

    map = image[:, :, 0]
    map = np.clip(map, 0.0, max_value)
    plt.imshow(map, cmap=cm_parula)  # , vmin=0.0, vmax=1.0)

    cbar = plt.colorbar(orientation='vertical', location='right')
    cbar.ax.tick_params(labelsize=28)  # Set font size for colorbar labels
    plt.axis('off')

    plt.savefig(filename, format='jpeg', dpi=300)
    plt.close()

    print("> Plot saved to " + filename)


def warp_parameter_maps(parameters):
    if torch.is_tensor(parameters):
        parameters_w = torch.clone(parameters)
    else:
        parameters_w = parameters.copy()
    parameters_w[:, 0] = parameters[:, 0] ** melanin_exp
    parameters_w[:, 1] = parameters[:, 1] ** hemoglobin_exp
    parameters_w[:, 2] = thick_min + parameters[:, 2] * (thick_max - thick_min)
    return parameters_w


def unwarp_parameter_maps(parameters):
    if torch.is_tensor(parameters):
        parameters_w = torch.clone(parameters)
    else:
        parameters_w = parameters.copy()
    parameters_w[:, 0] = torch.pow(parameters[:, 0], 1/melanin_exp)
    parameters_w[:, 1] = torch.pow(parameters[:, 1], 1/hemoglobin_exp)
    parameters_w[:, 2] = (- thick_min + parameters[:, 2]) / (thick_max - thick_min)
    return parameters_w


def remap_parameters_tensors(parameters_train, parameters_test):
    return unwarp_parameter_maps(parameters_train), unwarp_parameter_maps(parameters_test)
