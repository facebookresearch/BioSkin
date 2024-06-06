# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import cv2
import torch
import numpy as np
from bioskin.parameters.parameter import Parameter
from bioskin.parameters.parameter import SKIN_PROPS
import bioskin.utils.io as io
import bioskin.parameters.params_io as params_io
import bioskin.bioskin as bioskin
from skimage.exposure import match_histograms


class Character:
    def __init__(self, name, path, device, extension='.exr'):
        self.device = device
        self.name = name
        self.path = path
        self.extension = extension
        self.masked = False
        self.parameters = []

    def reset(self):
        self.albedo_map_edited = self.albedo_map.copy()
        self.ir_map_edited = self.ir_map.copy()
        self.reset_params()

    def reset_param(self, parameter_str):
        i = self.param_index(parameter_str)
        self.parameters[i].reset()

    def reset_params(self):
        for param in self.parameters:
            param.reset()

    def apply_operations_stack(self):
        for param in self.parameters:
            param.apply_operations_stack()

    def load_input_albedo(self, max_width=0):
        self.albedo_map = io.load_image(self.path + self.name + self.extension, max_width=max_width)
        self.albedo_map_edited = self.albedo_map.copy()
        self.albedo_map_edited_last = self.albedo_map.copy()
        self.has_speculars = False
        self.masked = False
        if os.path.exists(self.path + self.name + "_mask" + self.extension):
            self.mask = io.load_image(self.path + self.name + "_mask" + self.extension, max_width=max_width)
            self.masked = True
        if os.path.exists(self.path + self.name + "_spec" + self.extension):
            self.specular_map = io.load_image(self.path + self.name + "_spec" + self.extension, max_width=max_width)
            self.has_speculars = True

    def param_index(self, param_name):
        for i in range(len(SKIN_PROPS)):
            if SKIN_PROPS[i] == param_name:
                return i
        print("ERROR: Unknown skin parameter name!\n")
        return -1

    def num_params(self):
        return len(self.parameters)

    def get_param(self, param_name):
        index = self.param_index(param_name)
        return self.parameters[index].map

    def get_param_by_index(self, index):
        return self.parameters[index].map

    def get_param_edited(self, param_name):
        index = self.param_index(param_name)
        return self.parameters[index].map_edited_last

    def apply_mask(self):
        self.albedo_map_edited = (1.0 - self.mask) * self.albedo_map + self.mask * self.albedo_map_edited
        self.albedo_map_edited_last = self.albedo_map_edited.copy()
        if hasattr(self, 'ir_map') and self.ir_map is not None:
            self.ir_map_edited = (1.0 - self.mask) * self.ir_map + self.mask * self.ir_map_edited
            self.ir_map_edited_last = self.ir_map_edited.copy()

    def pack_parameters(self):
        # linearize skin properties to feed the network
        skin_props = []
        for i in range(0, len(self.parameters)):
            skin_prop = io.vectorize_image(self.parameters[i].map_edited_last, device=self.device, monochrome=True)
            skin_props.append(skin_prop)
        packed_skin_props = torch.stack(skin_props, dim=1)
        packed_skin_props = params_io.unwarp_parameter_maps(packed_skin_props)
        return packed_skin_props

    def unpack_parameters(self, skin_props):
        # output linear skin properties back to their original values
        skin_props = skin_props.cpu().detach().numpy()
        row = self.albedo_map.shape[0]
        col = self.albedo_map.shape[1]
        self.parameters = []
        warped_skin_props = params_io.warp_parameter_maps(skin_props)
        for i in range(0, skin_props.shape[1]):
            p = warped_skin_props[:, i]
            p = p.reshape(row, col)
            parameter_values = np.zeros((row, col, 3))
            parameter_values[:, :, 0] = p
            parameter_values[:, :, 1] = p
            parameter_values[:, :, 2] = p
            param_image = parameter_values.astype("float32")
            param = Parameter(SKIN_PROPS[i], param_image)
            self.parameters.append(param)
        return 0

    def estimate_skin_props(self, bio_skin):
        input_reflectance = io.vectorize_image(self.albedo_map, device=self.device)
        skin_props = bio_skin.reflectance_to_skin_props(input_reflectance)
        self.unpack_parameters(skin_props)
        return skin_props

    def reconstruct_albedo_from_skin_props(self, bio_skin, skin_props, original=False):
        ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg = bio_skin.skin_props_to_reflectance(skin_props)
        self.albedo_map_edited = io.tensor_to_image(ref_vis_rgb, self.albedo_map.shape, channels=3)
        self.albedo_map_edited_last = self.albedo_map_edited.copy()
        if original:
            self.albedo_map = self.albedo_map_edited.copy()

        if bio_skin.model_params.color_mode != 'rgb':
            ref_ir_avg = ref_ir_avg.unsqueeze(1)
            ref_ir_avg = ref_ir_avg.expand(-1, 3)
            self.ir_map_edited = io.tensor_to_image(ref_ir_avg, self.albedo_map.shape, channels=3)
            self.ir_map_edited_last = self.ir_map_edited.copy()
            if original:
                self.ir_map = self.ir_map_edited.copy()

        if self.masked:
            self.apply_mask()


    def reconstruct_albedo(self, bio_skin, original=False):
        skin_props = self.pack_parameters()
        self.reconstruct_albedo_from_skin_props(bio_skin, skin_props, original)

    def edit(self, param_name, operation, value):
        index = self.param_index(param_name)
        self.parameters[index].apply_operation(operation, value)

    def apply_operators_external_skin_props(self, skin_props):
        for index, parameter in enumerate(self.parameters):
            skin_props[:, index] = parameter.apply_operations_stack_external_map(skin_props[:, index])
        return skin_props

    def match_histogram(self, param_name, target_map):
        index = self.param_index(param_name)
        parameter = self.parameters[index]
        matched = match_histograms(parameter.map, target_map,  channel_axis=-1)
        parameter.map = matched.copy()
        parameter.map_edited = matched.copy()
        parameter.map_edited_last = matched.copy()

    def match_reference_hist(self, target_character):
        for i in range(0, min(len(self.parameters), len(target_character.parameters))):
            input_image = self.parameters[i].map_edited_last
            reference_image = target_character.parameters[i].map_edited_last
            matched = match_histograms(input_image, reference_image, channel_axis=-1)
            self.parameters[i].map_edited_last = matched

    def match_reference(self, target_character):
        for i in range(0, min(len(self.parameters), len(target_character.parameters))):
            value = np.median(target_character.get_param(SKIN_PROPS[i])) / np.median(self.get_param(SKIN_PROPS[i]))
            self.edit(SKIN_PROPS[i], "multiply_raw", value)

    def save_edited_albedo(self, path):
        cv2.imwrite(path, self.albedo_map_edited)

    def save_edited_albedo_jpeg(self, path):
        io.save_jpeg(path, self.albedo_map_edited)

    def save_param(self, str, path):
        index = self.param_index(str)
        cv2.imwrite(path, self.parameters[index].map)

    def save_param_edited(self, str, path):
        index = self.param_index(str)
        cv2.imwrite(path, self.parameters[index].map_edited_last)

    def save_parameters(self, path):
        for index in range(len(SKIN_PROPS)):
            filename_p = self.path + "_p" + str(index) + ".exr"
            cv2.imwrite(filename_p, self.parameters[index].map_edited_last)

    def save_reconstruction(self, bio_skin, path, save_spectrum=False, save_skin_props=True, save_error=True):
        skin_props = self.pack_parameters()
        input_image = self.albedo_map
        ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg = bio_skin.skin_props_to_reflectance(skin_props)
        input_reflectance = io.vectorize_image(input_image, self.device)
        reconstruction_error = abs(ref_vis_rgb - input_reflectance)
        bioskin.save_reconstruction(bio_skin.model_params, path, self.name, input_image, skin_props, ref_vis,
                                    ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error, save_spectrum=save_spectrum,
                                    save_skin_props=save_skin_props, save_error=save_error, device=bio_skin.device,
                                    save_plot=True)
