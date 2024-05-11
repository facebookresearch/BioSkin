# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import json
from argparse import Namespace
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import numpy as np
from scipy.stats import qmc
from itertools import product
import pandas as pd
import bioskin.spectrum.color_spectrum as color
import bioskin.parameters.params_io as params_io
import bioskin.utils.io as io
from bioskin.parameters.parameter import REGULAR_SKIN_MIN_VALUES
from bioskin.parameters.parameter import REGULAR_SKIN_MAX_VALUES
from bioskin.parameters.parameter import SKIN_MIN_VALUES
from bioskin.parameters.parameter import SKIN_MAX_VALUES
from bioskin.parameters.parameter import MIN_PROP_VALUES
from bioskin.parameters.parameter import MAX_PROP_VALUES


def get_unit_data_size(bio_skin):
    unit_data_size = bio_skin.model_params.D_rgb + bio_skin.latent_size
    if bio_skin.model_params.color_mode == 'spectral':
        unit_data_size += bio_skin.model_params.D_spec
        unit_data_size += bio_skin.model_params.D_rgb  # normalized reflectance
        unit_data_size += 1  # IR
        unit_data_size += 1  # occlusion
    return unit_data_size


class Skin2RgbNN(nn.Module):
    """ RGB reflectance -> skin properties -> RGB reflectance """
    def __init__(self, D_in, H, H_2, D_out):
        super().__init__()
        self.l1 = nn.Linear(D_in, H)
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(H, D_out)
        self.sigmoid = nn.Sigmoid()
        self.l3 = nn.Linear(D_out, H_2)
        self.l4 = nn.Linear(H_2, D_in)
        self.l1_1 = nn.Linear(H, H)
        self.l1_2 = nn.Linear(H, H)
        self.l3_1 = nn.Linear(H, H)
        self.l3_2 = nn.Linear(H, H)

    def forward1(self, x):
        return self.sigmoid(self.l2(self.tanh(self.l1_2(self.tanh(self.l1_1(self.tanh(self.l1(x))))))))

    def forward2(self, x):
        return self.sigmoid(self.l4(self.tanh(self.l3_2(self.tanh(self.l3_1(self.tanh(self.l3(x))))))))

    def forward(self, data, mode):
        if mode == 'skin2reflectance':
            return self.forward1(data)
        if mode == 'reflectance2skin':
            return self.forward2(data)
        else:
            raise ValueError("|mode| is invalid")


class Skin2ReflectanceNN(nn.Module):
    """ RGB reflectance -> skin properties -> Spectral reflectance """
    def __init__(self, H_enc, H_dec, D_rgb, D_skin, D_spec):
        super().__init__()

        self.act = nn.Tanh()
        self.act_final = nn.Sigmoid()

        self.fc_enc_in = nn.Linear(D_rgb, H_enc)
        self.fc_enc = nn.Linear(H_enc, H_enc)
        self.fc_enc_out = nn.Linear(H_enc, D_skin)

        self.fc_dec_in = nn.Linear(D_skin, H_dec)
        self.fc_dec = nn.Linear(H_dec, H_dec)
        self.fc_dec_out = nn.Linear(H_dec, D_spec)

    def decoder(self, x):
        hidden1 = self.act(self.fc_dec_in(x))
        hidden2 = self.act(self.fc_dec(hidden1))
        output = self.act_final(self.fc_dec_out(hidden2))
        return output

    def encoder(self, x):
        hidden1 = self.act(self.fc_enc_in(x))
        hidden2 = self.act(self.fc_enc(hidden1))
        output = self.act_final(self.fc_enc_out(hidden2))
        return output

    def forward(self, data, mode):
        if mode == 'skin2reflectance':
            return self.decoder(data)
        if mode == 'reflectance2skin':
            return self.encoder(data)
        else:
            raise ValueError("|mode| is invalid")


class Skin2ReflectanceOcclusionNN(nn.Module):
    """ Reflectance -> skin properties + occlusion -> Spectral reflectance """
    """ Latent code has skin parameters and an occlusion value, both connected to input and output in encoder decoder"""

    def __init__(self, H_enc, H_dec, D_rgb, D_skin, D_spec):
        super().__init__()

        self.act = nn.Tanh()
        self.act_final = nn.Sigmoid()

        self.fc_enc_in = nn.Linear(D_rgb, H_enc)
        self.fc_enc = nn.Linear(H_enc, H_enc)
        self.fc_enc_out = nn.Linear(H_enc, D_skin + 1)

        self.fc_dec_in = nn.Linear(D_skin + 1, H_dec)
        self.fc_dec = nn.Linear(H_dec, H_dec)
        self.fc_dec_out = nn.Linear(H_dec, D_spec)

    def decoder(self, x):
        hidden1 = self.act(self.fc_dec_in(x))
        hidden2 = self.act(self.fc_dec(hidden1))
        output = self.act_final(self.fc_dec_out(hidden2))
        return output

    def encoder(self, x):
        hidden1 = self.act(self.fc_enc_in(x))
        hidden2 = self.act(self.fc_enc(hidden1))
        output = self.act_final(self.fc_enc_out(hidden2))
        return output

    def forward(self, data, mode):
        if mode == 'skin2reflectance':
            return self.decoder(data)
        if mode == 'reflectance2skin':
            return self.encoder(data)
        else:
            raise ValueError("|mode| is invalid")


def save_training_parameters(args):
    print(args)
    args_dict = vars(args)
    json_filename = args.file_info
    with open(json_filename, "w") as f:
        json.dump(args_dict, f)
        f.close()
    print('Saved Training Parameters')


def load_training_parameters(json_file):
    print('Loading Training Parameters')
    with open(json_file) as f:
        args = json.load(f)
        print(args)
    return args


def load_model(model_params, device=torch.device('cuda')):
    if device == torch.device('cuda'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")
    else:
        device = torch.device('cpu')
        print("Device: " + str(device))

    if model_params.color_mode == 'rgb':
        model = Skin2RgbNN(model_params.D_skin, model_params.H_enc, model_params.H_enc, model_params.D_rgb)
    elif model_params.color_mode == 'spectral':
        if not hasattr(model_params, 'exposure_aware'):
            model = Skin2ReflectanceNN(model_params.H_enc, model_params.H_dec, model_params.D_rgb,
                                       model_params.D_skin, model_params.D_spec)
        else:
            model = Skin2ReflectanceOcclusionNN(model_params.H_enc, model_params.H_dec,
                                                model_params.D_rgb, model_params.D_skin, model_params.D_spec)
    else:
        print('ERROR: Choose "rgb" or "spectral" color mode')
        exit(0)

    if device == torch.device('cuda'):
        if torch.cuda.device_count() >= 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model)

    model.to(device)
    return model, device


def load_trained_weights(device, model, path):
    state_dict = torch.load(path, map_location=device)

    if device == torch.device('cpu') or isinstance(model, torch.nn.DataParallel) and (not torch.cuda.is_available()
                                                                                      or torch.cuda.device_count() < 1):
        # remove 'module.' from key
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    if device == torch.device('cuda') and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # if it does not have module prefix, add 'module.' from key
        if not any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict['module.' + key] = value
                state_dict = new_state_dict.copy()

    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


def load_model_and_trained_weights(path, device):
    print('Loading model in ' + path)
    model_params = load_training_parameters(path + '.json')
    model_params = Namespace(**model_params)

    if os.path.exists(path + '.pt'):
        model, device = load_model(model_params, device)
        model = load_trained_weights(device, model, path + '.pt')
        flag = True
    else:
        print('WARNING: Model not found, skipping...')
        model = []
        model_params = []
        flag = False

    return model, model_params, flag


def save_reconstruction(model_params, path, name, input_image, skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg,
                        reconstruction_error, device=torch.device('cuda'), save_spectrum=False,
                        save_spectrums_csv=False, save_skin_props=True, save_plot=False, save_error=True):

    if device == torch.device('cuda'):
        cpu = False
    else:
        cpu = True

    path = path + name
    # save skin properties
    if hasattr(model_params, "exposure_aware"):
        occlusion = skin_props[:, model_params.D_skin]
        skin_props = skin_props[:, 0:model_params.D_skin]
        normalized_reflectance = ref_vis_rgb / occlusion.unsqueeze(1)

        io.save_tensor_to_image(path + "_reconstruction_occlusion", occlusion, input_image.shape, channels=1, cpu=cpu)
        io.save_tensor_to_image(path + "_reconstruction_vis_normalized", normalized_reflectance, input_image.shape,
                                channels=3, cpu=cpu)

    if save_skin_props:
        if not cpu:
            skin_props = skin_props.cpu().detach()
        shape = input_image.shape
        params_io.save_parameter_maps(skin_props.numpy(), shape[0], shape[1], path, save_plot=save_plot)

    io.save_jpeg(path + "_input", input_image, linear_input=True)
    io.save_tensor_to_image(path + "_reconstruction_vis", ref_vis_rgb, input_image.shape, channels=3, cpu=cpu)
    if save_error:
        io.save_tensor_to_image(path + "_reconstruction_diff", reconstruction_error, input_image.shape, channels=3,
                                cpu=cpu)
    if model_params.color_mode == "spectral":
        io.save_tensor_to_image(path + "_reconstruction_IR", ref_ir_avg, input_image.shape, channels=1, cpu=cpu)

        if save_spectrum:
            print("\nSaving spectrums to .npy...")
            if not cpu:
                ref_vis = ref_vis.cpu().detach()
            ref_vis = ref_vis.numpy()
            np.save(path + '_reconstruction_vis_spectral.npy', ref_vis)
            if not cpu:
                ref_ir = ref_ir.cpu().detach()
            ref_ir = ref_ir.numpy()
            np.save(path + '_reconstruction_ir_spectral.npy', ref_ir)
            print("DONE!")

            if save_spectrums_csv:
                print("\nSaving spectrums to CSV, will take a while...")
                ref_vis_df = pd.DataFrame(ref_vis)
                ref_vis_df.to_csv(path + '_reconstruction_vis_spectral.csv', index=False)
                ref_ir_df = pd.DataFrame(ref_ir)
                ref_ir_df.to_csv(path + '_reconstruction_ir_spectral.csv', index=False)
            print("DONE!")


def export_spectral_bands(out_filename, device, spectrums, input_image, power_spectra=None):
    if device == torch.device('cuda'):
        cpu = False
    else:
        cpu = True

    if not power_spectra:
        power_spectra = np.array([4.98E-128, 2.82E-111, 1.17E-09, 1.41E-08, 1.45E-07, 1.27E-06, 9.47E-06, 6.01E-05,
                                  0.0003246213982, 0.001493535282, 0.005851690826, 0.01952426981, 0.0554748034,
                                  0.1342283606, 0.2765797266, 0.4853152552, 0.7251954753, 0.9228131434, 1, 0.9228131434,
                                  0.7251954753, 0.4853152552, 0.2765797266, 0.1342283606, 0.0554748034, 0.01952426981,
                                  0.005851690826])
        power_spectra = power_spectra[::-1]

    new_size = spectrums.shape[1]
    power_spectra_interp = torch.from_numpy(np.interp(np.linspace(0, len(power_spectra) - 1, new_size),
                                     np.arange(len(power_spectra)), power_spectra))

    spectrums_weighted_average = torch.zeros_like(spectrums[:, 0])
    for i in range(0, spectrums.shape[1]):
        # io.save_tensor_to_image(out_filename + "_norm_" + str(i) + ".exr", spectrums[:, i],
        #                         input_image.shape, channels=1, cpu=cpu)

        weighted_spectrum = spectrums[:, i] * power_spectra_interp[i]
        if torch.sum(weighted_spectrum) > 0.0001:
            spectrums_weighted_average += weighted_spectrum
            # io.save_tensor_to_image(out_filename + "_light_weighted_" + str(i) + ".exr",
            #                         weighted_spectrum,
            #                         input_image.shape, channels=1, cpu=cpu)

    spectrums_weighted_average /= torch.sum(power_spectra_interp)

    io.save_tensor_to_image(out_filename + "_light_weighted_avg.exr",
                            spectrums_weighted_average,
                            input_image.shape, channels=1, cpu=cpu)


class BioSkinInference:
    def __init__(self, model_json_path, device=torch.device('cpu'), batch_size=65536):
        super().__init__()
        self.device = device
        self.model_path = model_json_path
        self.model, self.model_params, self.loaded = load_model_and_trained_weights(self.model_path, device=device)
        if not self.loaded:
            print("ERROR loading the pre trained bio skin model")
            exit(0)
        self.visible_range_limit = int((400 / 620) * self.model_params.D_spec)
        self.Spectrum = color.ColorSpectrum(device, self.visible_range_limit)
        self.batch_size = batch_size
        self.exposure_aware = False
        self.latent_size = self.model_params.D_skin
        if hasattr(self.model_params, "exposure_aware"):
            self.latent_size = self.latent_size + 1
        if self.model_params.color_mode == 'rgb':
            self.reflectance_size = self.model_params.D_rgb
        elif self.model_params.color_mode == 'spectral':
            self.reflectance_size = self.model_params.D_spec
        else:
            print('ERROR color_mode has to be rgb or spectral')
            exit(0)

    def reflectance_to_skin_props(self, input_reflectance):
        length = len(input_reflectance)
        batch_num = int(np.ceil(length / self.batch_size))
        skin_params = torch.zeros((length, self.latent_size), device=self.device, dtype=torch.float32)

        #  Estimate skin parameters from reflectance input
        for index in range(0, batch_num):
            i1 = index * self.batch_size
            i2 = min(length, (index + 1) * self.batch_size)

            input_reflectance_batch = input_reflectance[i1:i2, :].to(self.device)
            skin_params_batch = self.model.forward(input_reflectance_batch, 'reflectance2skin')
            skin_params[i1:i2, :] = skin_params_batch
        return skin_params

    def skin_props_to_reflectance(self, skin_params):
        length = len(skin_params)
        batch_num = int(np.ceil(length / self.batch_size))
        reflectances = torch.zeros((skin_params.shape[0], self.reflectance_size), device=self.device,
                                   dtype=torch.float32)

        for index in range(0, batch_num):
            i1 = index * self.batch_size
            i2 = min(length, (index + 1) * self.batch_size)
            skin_params_batch = skin_params[i1:i2, :].to(self.device)
            reflectance_batch = self.model.forward(skin_params_batch, 'skin2reflectance')
            reflectances[i1:i2, :] = reflectance_batch

        if self.model_params.color_mode == 'spectral':
            reflectances_visible = reflectances[:, :self.visible_range_limit]
            reflectances_visible_rgb = self.Spectrum.spectrum_to_rgb(reflectances_visible)

            reflectances_ir = reflectances[:, self.visible_range_limit + 1:self.model_params.D_spec]
            reflectances_ir_avg = self.Spectrum.spectrum_to_infrared(reflectances_ir)
        else:
            reflectances_visible = reflectances
            reflectances_visible_rgb = reflectances_visible
            reflectances_ir = 0
            reflectances_ir_avg = 0
        return reflectances_visible, reflectances_visible_rgb, reflectances_ir, reflectances_ir_avg

    def reconstruct(self, input_reflectance):
        skin_props = self.reflectance_to_skin_props(input_reflectance)
        ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg = self.skin_props_to_reflectance(skin_props)
        reconstruction_error = abs(ref_vis_rgb.to(self.device) - input_reflectance.to(self.device))
        return skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error

    def get_limits(self, ranges="RegularSkin"):
        if ranges == "RegularSkin":
            lower_limits = torch.tensor(REGULAR_SKIN_MIN_VALUES[0:self.model_params.D_skin])
            upper_limits = torch.tensor(REGULAR_SKIN_MAX_VALUES[0:self.model_params.D_skin])
        elif ranges == "AllSkin":
            lower_limits = torch.tensor(SKIN_MIN_VALUES[0:self.model_params.D_skin])
            upper_limits = torch.tensor(SKIN_MAX_VALUES[0:self.model_params.D_skin])
        elif ranges == "FullRange":
            lower_limits = torch.tensor(MIN_PROP_VALUES[0:self.model_params.D_skin])
            upper_limits = torch.tensor(MAX_PROP_VALUES[0:self.model_params.D_skin])
        else:
            print("Parameter 'Ranges' unknown type, use 'RegularSkin', 'AllSkin', or 'FullRange'")
            exit(0)
        return lower_limits, upper_limits

    def sample_skintones(self, num_samples, mode='Random', ranges="RegularSkin"):

        params_sample_0to1 = torch.rand(num_samples, self.model_params.D_skin)

        if mode == "QuasiRandom":
            sampler = qmc.Halton(d=5, scramble=True)
            params_sample_0to1 = torch.tensor(sampler.random(n=num_samples))

        params_sample = params_io.warp_parameter_maps(params_sample_0to1)

        lower_limits, upper_limits = self.get_limits(ranges)

        lower_limits = lower_limits.expand(num_samples, self.model_params.D_skin)
        upper_limits = upper_limits.expand(num_samples,  self.model_params.D_skin)

        mask_lower = params_sample.ge(lower_limits)
        mask_upper = params_sample.le(upper_limits)

        mask = torch.logical_and(mask_lower, mask_upper)
        mask_count_trues_per_tuple = mask.sum(1)
        mask_flat_filtered = mask_count_trues_per_tuple.ge(self.model_params.D_skin)  # all parameters within range
        mask_flat_filtered = mask_flat_filtered.resize(num_samples, 1)  # torch.t(mask_flat_filtered)

        mask_filtered_parameters = mask_flat_filtered.expand(num_samples, self.model_params.D_skin)
        params_sample_filtered = torch.masked_select(params_sample, mask_filtered_parameters)

        new_size = int(np.floor(params_sample_filtered.shape[0] / self.model_params.D_skin))
        params_sample_filtered = params_sample_filtered.resize(new_size, self.model_params.D_skin)
        params_sample = params_sample_filtered

        params_sample = params_sample.clone().detach().to(self.device).type(torch.float32)
        # params_sample = params_sample.clone().detach()
        if hasattr(self.model_params, "exposure_aware"):
            ones = torch.ones((params_sample.size(dim=0), 1), device=self.device, dtype=torch.float32)
            params_sample = torch.cat((params_sample, ones), dim=1)
            params_sample = params_sample.to(self.device)
        with torch.no_grad():
            params_sample = params_io.unwarp_parameter_maps(params_sample)
            reflectances_visible, reflectances_visible_rgb, \
            reflectances_ir, reflectances_ir_avg = self.skin_props_to_reflectance(params_sample)
            params_sample = params_io.warp_parameter_maps(params_sample)
        return reflectances_visible, reflectances_visible_rgb, reflectances_ir, reflectances_ir_avg, params_sample

    def sample_skintones_rgb(self, num_samples, mode, ranges):
        reflectances_visible, reflectances_visible_rgb, reflectances_ir, reflectances_ir_avg, params_sample = \
            self.sample_skintones(num_samples, mode, ranges)
        return reflectances_visible, reflectances_visible_rgb, reflectances_ir, reflectances_ir_avg, params_sample

    def sample_skintones_by_props(self, ranges="RegularSkin", num_samples=[4, 3, 3, 3, 3]):
        lower_limits, upper_limits = self.get_limits(ranges)
        ranges = [(lower_limits[0], upper_limits[0]),
                  (lower_limits[1], upper_limits[1]),
                  (lower_limits[2], upper_limits[2]),
                  (lower_limits[3], upper_limits[3]),
                  (lower_limits[4], upper_limits[4])]
        tensors = [torch.linspace(start, end, num) for (start, end), num in zip(ranges, num_samples)]
        combinations = list(product(*tensors))
        params_sample = torch.tensor(combinations, device=self.device, dtype=torch.float32)

        if hasattr(self.model_params, "exposure_aware"):
            ones = torch.ones((params_sample.size(dim=0), 1), device=self.device, dtype=torch.float32)
            params_sample = torch.cat((params_sample, ones), dim=1)
            params_sample = params_sample.to(self.device)
        with torch.no_grad():
            reflectances_visible, reflectances_visible_rgb, reflectances_ir, reflectances_ir_avg = \
                self.skin_props_to_reflectance(params_io.unwarp_parameter_maps(params_sample))
        return reflectances_visible, reflectances_visible_rgb, reflectances_ir, reflectances_ir_avg, params_sample




