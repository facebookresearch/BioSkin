# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import argparse
import os
import shutil
import json
from argparse import Namespace
import BioSkin
from parameters.params_io import remap_parameters_tensors
import color.color_spectrum as color
import loss.loss_functions as loss_functions
from loss.log_losses import save_log_losses_exp, plot_losses

writer = SummaryWriter()


class Losses:
    loss_parameters = 0
    loss_reflectance = 0
    loss_reflectance_full = 0
    loss_mode = 0
    weight = 0


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Supervised Encoder Decoder to map Reflectance->SkinProps->Reflectance')

    # modes
    parser.add_argument('--color_mode', nargs='?',
                        type=str, default='spectral',
                        help='Color mode: rgb|spectral')

    parser.add_argument('--color_space', nargs='?',
                        # type=str, default='RGB_nonlinear',
                        type=str, default='RGB_linear',
                        help='Color linearity: RGB_nonlinear|RGB_linear|Lab')

    parser.add_argument('--param_mode', nargs='?',
                        type=str, default='linear',
                        help='Parameter mode: linear|nonlinear')

    parser.add_argument('--exposure_aware',
                        action='store_true',
                        help='Enable exposure awareness')

    #losses
    parser.add_argument('--loss_mode', nargs='?',
                        type=str, default='visible_range',
                        help='Loss mode: visible_range | full_spectrum')
    parser.add_argument('--loss_params', nargs='?',
                        type=str, default='MSE',
                        help='Parameter loss: MSE')
    parser.add_argument('--loss_color', nargs='?',
                        type=str, default='sam',
                        help='Reflectance loss: l1|luv|MSE|sam|mixed')
    parser.add_argument('--loss_color_full_cycle',  nargs='?',
                        type=str, default='l1',
                        help='Reflectance loss (full cycle): l1|luv|MSE|sam|mixed')

    parser.add_argument('--w_p', nargs='?',
                        type=float, default=float(1),
                        help='Weight Loss Parameters (encoder)')
    parser.add_argument('--w_e', nargs='?',
                        type=float, default=float(1),
                        help='Weight Loss Exposures (encoder)')
    parser.add_argument('--w_a', nargs='?',
                        type=float, default=float(1),
                        help='Weight Loss Albedo (decoder)')
    parser.add_argument('--w_full', nargs='?',
                        type=float, default=float(1),
                        help='Weight Loss Full cycle albedo-parameters-albedo (encoder-decoder)')

    # optimization
    parser.add_argument('--n_epochs', nargs='?',
                        type=int, default=1,
                        help='Max number of epochs')
    parser.add_argument('--learning_rate', nargs='?',
                        type=float, default=0.0001,
                        help='Learning Rate')
    parser.add_argument('--learning_rate_update', nargs='?',
                        type=int, default=100,
                        help='Number of epochs whether check and update Learning Rate')
    parser.add_argument('--batch_size', nargs='?',
                        type=int, default=256,
                        help='Batch Size')
    parser.add_argument('--writing_rate', nargs='?',
                        type=int, default=1,
                        help='Model save every <writing_rate> epochs')

    # data and network size
    parser.add_argument('--D_rgb', nargs='?',
                        type=int, default=3,
                        help='Size input RGB')
    parser.add_argument('--H_enc', nargs='?',
                        type=int, default=70,
                        help='Hidden layers encoder albedo-->skin')
    parser.add_argument('--H_dec', nargs='?',
                        type=int, default=512,
                        help='Hidden layers decoder skin --> albedo')

    # input dataset for training
    parser.add_argument('--dataset_folder', nargs='?',
                        type=str, default='../datasets/',
                        help='Input Dataset Folder')
    parser.add_argument('--dataset_filename', nargs='?',
                        type=str, default='dataset_wide_thickness_wide_spectrum_380_1000nm_2nm_1M_photons_uniform_0.01_1.0_samples_14',
                        help='Input Dataset Filename')

    # output trained model
    parser.add_argument('--output_model_folder', nargs='?',
                        type=str, default='../models/',  # default='Z:/Users/Carlos/skin_models/',
                        help='Trained Neural Model Folder')
    parser.add_argument('--output_model_name', nargs='?',
                        type=str, default='',
                        help='Trained Neural Model Filename')

    args = parser.parse_args()

    args.exposure_aware = True
    return args


def process_arguments(args):

    loss_types = Losses()

    if args.loss_params == 'MSE':
        loss_types.loss_parameters = torch.nn.MSELoss()
    else:
        loss_types.loss_parameters = torch.nn.MSELoss()

    loss_types.loss_mode = args.loss_mode
    args.weight = 0.0
    if args.loss_color == 'l1':
        loss_types.loss_reflectance = loss_functions.L1  # L1 loss (which maybe is better/more perceptual for colors)

    elif args.loss_color == 'MSE':
        loss_types.loss_reflectance = loss_functions.MSE

    # custom
    elif args.loss_color == 'luv':
        loss_types.loss_reflectance = loss_functions.Luv_approx

    elif args.loss_color == 'wluv':
        loss_types.loss_reflectance = loss_functions.Luv_approx_regularized

    elif args.loss_color == 'wRGB' or args.loss_color == 'wrgb':
        loss_types.loss_reflectance = loss_functions.weighted_RGB_euclidean

    elif args.loss_color == 'mixed_l1':
        loss_types.loss_reflectance = loss_functions.mixed_luv_l1

    elif args.loss_color == 'mixed_l2':
        loss_types.loss_reflectance = loss_functions.mixed_luv_l2

    elif args.loss_color == 'sam' or args.loss_color == 'sam_weighted':
        if args.color_mode != 'spectral':
            print("ERROR: SAM metric ONLY applies to SPECTRAL")
            exit(0)
        else:
            loss_types.loss_reflectance = loss_functions.sam_spectral_loss
    else:
        if args.color_mode == 'rgb':    # default rgb loss is L1
            loss_types.loss_reflectance = loss_functions.L1
        else:
            loss_types.loss_reflectance = loss_functions.sam_spectral_loss


    # loss colors full cycle loss
    if args.loss_color_full_cycle == 'l1':
        # L1 loss (which maybe is better/more perceptual for colors)
        loss_types.loss_reflectance_full = loss_functions.L1

    elif args.loss_color_full_cycle == 'MSE':
        loss_types.loss_reflectance_full = loss_functions.MSE

    elif args.loss_color_full_cycle == 'sam' or args.loss_color_full_cycle == 'sam_weighted':
        if args.color_mode != 'spectral':
            print("ERROR: SAM metric ONLY applies to SPECTRAL")
            exit(0)
        else:
            loss_types.loss_reflectance_full = loss_functions.sam_spectral_loss
    else:
        if args.color_mode == 'rgb':    # default rgb loss is L1
            loss_types.loss_reflectance_full = loss_functions.L1
        else:
            loss_types.loss_reflectance_full = loss_functions.sam_spectral_loss

    return args, loss_types


def set_dirs(args):
    # output models
    x = datetime.datetime.now()
    datetime_prefix = str(x.year) + '_' + str(x.month) + '_' + str(x.day) + '_' + str(x.hour) + \
                      '_' + str(x.minute) + "_" + str(x.microsecond) + "__"

    if args.output_model_name == '':
        args.output_model_name = args.dataset_filename

    model_name = datetime_prefix + args.output_model_name + \
                 '_' + args.color_mode + '_h_' + str(args.H_enc) + '_' + str(args.H_dec)


    args.output_model_folder += model_name

    args.already_trained = False
    if not os.path.exists(args.output_model_folder + '/'):
        os.makedirs(args.output_model_folder + '/')
    else:
        args.already_trained = True

    args.file_info = args.output_model_folder + ".json"
    args.log_filename = args.output_model_folder + ".txt"
    args.loss_graph_path = args.output_model_folder + "_loss.png"
    return args


def load_dataset_specs(args):
    # input datasets
    args.foldername = args.dataset_folder + args.dataset_filename + '/'
    args.foldername_test = args.foldername + 'test/tensors/'
    args.foldername_train = args.foldername + 'train/tensors/'

    dataset_specs_json = args.foldername + args.dataset_filename + '.json'
    print('Loading Dataset Specs from ' + str(dataset_specs_json))
    with open(dataset_specs_json) as f:
        dataset_specs = json.load(f)
        print(dataset_specs)
    dataset_specs = Namespace(**dataset_specs)

    args.wavelength_begin = dataset_specs.wavelength_begin
    args.wavelength_end = dataset_specs.wavelength_end
    args.spectral_resolution = dataset_specs.spectral_resolution

    args.D_skin = dataset_specs.skin_params
    args.D_spec = int((dataset_specs.wavelength_end - dataset_specs.wavelength_begin)
                      / dataset_specs.spectral_resolution)

    #TODO:
    args.visible_range_limit = int((780 - 380) / dataset_specs.spectral_resolution)

    if args.H_dec < args.D_spec:
        args.H_dec = args.D_spec
        print('>>>>WARNING: hidden layer is smaller than output spectrum, setting it up to same size, ' + str(args.H_dec))
    return args


def load_dataset_tensors(args):

    parameters_train = torch.load(args.foldername_train + args.dataset_filename + '_params_train.pt')
    parameters_test = torch.load(args.foldername_test + args.dataset_filename + '_params_test.pt')
    parameters_train, parameters_test = remap_parameters_tensors(parameters_train, parameters_test)

    spec_albedos_train = torch.load(args.foldername_train + args.dataset_filename + '_spec_train.pt')
    spec_albedos_test = torch.load(args.foldername_test + args.dataset_filename + '_spec_test.pt')

    args.D_spec = spec_albedos_train.shape[1]

    rgb_albedos_train = torch.load(args.foldername_train + args.dataset_filename + '_rgb_train.pt')
    rgb_albedos_test = torch.load(args.foldername_test + args.dataset_filename + '_rgb_test.pt')

    if args.color_space == 'RGB_linear':
        rgb_albedos_train = torch.tensor(rgb_albedos_train.numpy()**(1/2.2))
        rgb_albedos_test = torch.tensor(rgb_albedos_test.numpy()**(1/2.2))
    elif args.color_space == 'RGB_nonlinear':
        #TODO
        print("Color Space: RGB_nonlinear\n")
    else:
        print('Wrong Color Space')
        exit(0)

    exposures_train = torch.load(args.foldername_train + args.dataset_filename + '_exp_train.pt')
    exposures_test = torch.load(args.foldername_test + args.dataset_filename + '_exp_test.pt')

    return parameters_train, spec_albedos_train, rgb_albedos_train, exposures_train,\
           parameters_test, spec_albedos_test, rgb_albedos_test, exposures_test


def spectral_distance(args, spec_albedo, spec_albedo_gt, exposure, rgb_albedo_gt, loss_types, color_spectrum):

    spec_albedo = spec_albedo[:, :args.visible_range_limit]
    spec_albedo_gt = spec_albedo_gt[:, :args.visible_range_limit]
    loss_hue = loss_types.loss_reflectance(spec_albedo, spec_albedo_gt, loss_types.weight)

    rgb_albedo = exposure * color_spectrum.spectrum_to_rgb(spec_albedo)

    perceptual_luminance = loss_functions.perceptual_luminance(rgb_albedo)
    perceptual_luminance_gt = loss_functions.perceptual_luminance(rgb_albedo_gt)

    loss_decoder_value = loss_functions.L1(perceptual_luminance, perceptual_luminance_gt, loss_types.weight)

    return 0.5 * (loss_hue + loss_decoder_value)


def forward_pass_and_compute_loss(args, model, loss_types, w_albedo, w_params, w_exposure, w_albedo_full_cycle,
                                  parameters, spec_albedos, rgb_albedos, exposures, color_spectrum):
    # enconder
    parameters_exp_pred = model.forward(rgb_albedos, 'reflectance2skin')  # encoder pass, GT reflectance and exposure

    parameters_pred = parameters_exp_pred[:, :parameters.size(1)]
    exposures_pred = parameters_exp_pred[:, parameters.size(1):]

    parameters_exp = torch.cat((parameters, exposures), dim=1)

    # decoder
    spec_albedo_pred_params_known = model.forward(parameters_exp, 'skin2reflectance')  # decoder pass, GT skin params

    # encoder-decoder
    spec_albedo_pred_params_pred = model.forward(parameters_exp_pred, 'skin2reflectance')  # encoder-decoder pass

    # loss encoder
    loss_encoder = loss_types.loss_parameters(parameters_pred, parameters)  # MSE
    loss_encoder_exp = loss_types.loss_parameters(exposures_pred, exposures)  # MSE

    # loss_decoder
    loss_decoder = loss_types.loss_reflectance(spec_albedo_pred_params_known, spec_albedos, loss_types.weight)

    # full cycle loss
    if args.loss_color_full_cycle == 'sam':
        if args.loss_mode == 'visible_range':
            loss_full_cycle = loss_types.loss_reflectance_full(
                spec_albedo_pred_params_pred[:, :args.visible_range_limit],
                spec_albedos[:, :args.visible_range_limit],
                loss_types.weight)
        elif args.loss_mode == 'full_spectrum':
            loss_full_cycle = loss_types.loss_reflectance_full(spec_albedo_pred_params_pred, spec_albedos,
                                                               loss_types.weight)
        else:
            print('ERROR: select a valid loss_mode: visible_range | full_spectrum')
            exit(0)

    elif args.loss_color_full_cycle == 'luv' or args.loss_color_full_cycle == 'mse' or args.loss_color_full_cycle == 'l1':
        if args.loss_mode == 'visible_range':

            rgb_pred = color_spectrum.spectrum_to_rgb(spec_albedo_pred_params_pred[:, :args.visible_range_limit])
            perceptual_luminance_pred = loss_functions.perceptual_luminance(rgb_pred)
            perceptual_luminance_gt = loss_functions.perceptual_luminance(rgb_albedos)

            loss_full_cycle = loss_functions.L1(perceptual_luminance_pred, perceptual_luminance_gt, loss_types.weight) + \
                              loss_types.loss_reflectance(spec_albedo_pred_params_pred, spec_albedos, loss_types.weight)

        elif args.loss_mode == 'full_spectrum':
            loss_full_cycle = loss_types.loss_reflectance_full(spec_albedo_pred_params_pred, spec_albedos ,
                                                               loss_types.weight)
        else:
            print('ERROR: select a valid loss_mode: visible_range | full_spectrum')
            exit(0)

    elif args.loss_color_full_cycle == 'l1_spectral':
        # TODO
        print("Loss to be implemented!")
        exit(0)
    else:
        print('ERROR: select sam | rgb | mse | l1_spectral | mse_spectral as loss_color_full_cycle')
        exit(0)

    loss = w_albedo * loss_decoder + w_params * loss_encoder + w_exposure * loss_encoder_exp \
           + w_albedo_full_cycle * loss_full_cycle

    loss /= w_albedo + w_params + w_albedo_full_cycle + w_exposure
    return loss, loss_encoder, loss_encoder_exp, loss_decoder, loss_full_cycle


def make_train_step(args, loss_types, w_params, w_albedo, w_exposure, w_albedo_full_cycle, model, optimizer,
                    color_spectrum):
    # Builds function that performs a step in the train loop
    def train_step_spectral(parameters, spec_albedos, rgb_albedos, exposures):
        # Sets model to TRAIN mode
        model.train()

        loss, loss_encoder, loss_encoder_exp, loss_decoder, loss_full_cycle = \
            forward_pass_and_compute_loss(args, model, loss_types, w_albedo, w_params, w_exposure, w_albedo_full_cycle,
                                          parameters, spec_albedos, rgb_albedos, exposures, color_spectrum)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), loss_encoder.item(), loss_encoder_exp.item(), loss_decoder.item(), loss_full_cycle.item()

    # Builds function that performs a step in the train loop
    def train_step_rgb(parameters, spec_albedos, rgb_albedos):
        # Sets model to TRAIN mode
        model.train()

        # Forward pass
        params_pred = model.forward(rgb_albedos, 'reflectance2skin')
        albedo_pred_params_pred = model.forward(params_pred, 'skin2reflectance')
        albedo_pred_params_known = model.forward(parameters, 'skin2reflectance')

        # computes loss
        loss_param = loss_types.loss_parameters(params_pred, parameters)
        loss_albedo = loss_types.loss_reflectance(albedo_pred_params_known, rgb_albedos, loss_types.weight)
        loss_albedo_full_cycle = loss_types.loss_reflectance_full(albedo_pred_params_pred, rgb_albedos, loss_types.weight)

        loss = w_albedo * loss_albedo + w_params * loss_param + w_albedo_full_cycle * loss_albedo_full_cycle
        loss /= w_albedo + w_params + w_albedo_full_cycle

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), loss_param.item(), loss_albedo.item(), loss_albedo_full_cycle.item()

    # Returns the function that will be called inside the train loop
    if args.color_mode == 'rgb':
        train_step_func = train_step_rgb
    elif args.color_mode == 'spectral':
        train_step_func = train_step_spectral
    else:
        print("ERROR in making the train step, select correct mode rgb|spectral")
    return train_step_func


def make_test_step(args, loss_types, w_params, w_albedo, w_exposure, w_albedo_full_cycle, model, color_spectrum):

    # Builds function that performs a step in the test loop
    def test_step_spectral(parameters, spec_albedos, rgb_albedos, exposures):
        # Sets model to TEST mode
        model.eval()
        loss, loss_encoder, loss_encoder_exp, loss_decoder, loss_full_cycle = \
            forward_pass_and_compute_loss(args, model, loss_types, w_albedo, w_params, w_exposure, w_albedo_full_cycle,
                                          parameters, spec_albedos, rgb_albedos, exposures, color_spectrum)
        # Returns the loss
        return loss.item(), loss_encoder.item(), loss_encoder_exp.item(), loss_decoder.item(), loss_full_cycle.item()

    # Builds function that performs a step in the test loop
    def test_step_rgb(parameters, spec_albedos, rgb_albedos):
        # Sets model to TRAIN mode
        model.train()

        # Forward pass
        params_pred = model.forward(rgb_albedos, 'reflectance2skin')
        albedo_pred_params_pred = model.forward(params_pred, 'skin2reflectance')
        albedo_pred_params_known = model.forward(parameters, 'skin2reflectance')

        # computes loss
        loss_param = loss_types.loss_parameters(params_pred, parameters)
        loss_albedo = loss_types.loss_reflectance(albedo_pred_params_known, rgb_albedos, loss_types.weight)
        loss_albedo_full_cycle = loss_types.loss_reflectance_full(albedo_pred_params_pred, rgb_albedos,
                                                                  loss_types.weight)

        loss = w_albedo * loss_albedo + w_params * loss_param + w_albedo_full_cycle * loss_albedo_full_cycle
        loss /= w_albedo + w_params + w_albedo_full_cycle

        return loss.item(), loss_param.item(), loss_albedo.item(), loss_albedo_full_cycle.item()

    # Returns the function that will be called inside the train loop
    if args.color_mode == 'rgb':
        test_step_func = test_step_rgb
    elif args.color_mode == 'spectral':
        test_step_func = test_step_spectral
    else:
        print("ERROR in making the train step, select correct mode rgb|spectral")
    return test_step_func


def run(args):
    args, loss_types = process_arguments(args)
    args = load_dataset_specs(args)
    args = set_dirs(args)

    if args.already_trained:
        print('>>>>WARNING: Already trained, skipping ' + args.output_model_folder)
        return 0
    
    log_file = open(args.log_filename, "a")

    parameters_train, spec_albedos_train, rgb_albedos_train, exposures_train,\
    parameters_test, spec_albedos_test, rgb_albedos_test, exposures_test = load_dataset_tensors(args)

    # save training and model specs
    BioSkin.save_training_parameters(args)

    # load nn model
    model, device = BioSkin.load_model(args)

    # load datasets
    train_dataset = TensorDataset(parameters_train, spec_albedos_train, rgb_albedos_train, exposures_train)
    val_dataset = TensorDataset(parameters_test, spec_albedos_test, rgb_albedos_test, exposures_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset))  # batch_size=len(val_dataset))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # color spectrum functions to device
    spectrum_size = args.D_spec
    if args.loss_mode == 'visible_range':
        spectrum_size = args.visible_range_limit

    color_spectrum = color.ColorSpectrum(device, spectrum_size)

    # make train and test steps
    train_step = make_train_step(args, loss_types, args.w_p, args.w_a, args.w_e, args.w_full, model, optimizer,
                                 color_spectrum)

    test_step = make_test_step(args, loss_types, args.w_p, args.w_a, args.w_e, args.w_full, model, color_spectrum)

    train_losses, train_losses_param, train_losses_exp = [], [], []
    train_losses_albedo, train_losses_albedo_full_cycle = [], []

    test_losses, test_losses_param, test_losses_exp = [], [], []
    test_losses_albedo, test_losses_albedo_full_cycle = [], []

    best_test_loss = 10000
    best_epoch = 0

    print("starting training...\n")
    for epoch in range(0, args.n_epochs):
        # batch losses
        b_train_loss, b_train_loss_param, b_train_loss_exp = 0.0, 0.0, 0.0
        b_train_loss_albedo, b_train_loss_albedo_full_cycle = 0.0, 0.0

        for params_batch, spec_albedos_batch, rgb_albedos_batch, exposures_batch in train_loader:
            params_batch = params_batch.to(device)  # parameters
            rgb_albedos_batch = rgb_albedos_batch.to(device)  # albedos rgb
            spec_albedos_batch = spec_albedos_batch.to(device)  # albedos spectral
            exposures_batch = exposures_batch.to(device)     # exposures

            train_loss, train_loss_param, train_loss_exp, train_loss_albedo, train_loss_albedo_full_cycle \
                = train_step(params_batch, spec_albedos_batch, rgb_albedos_batch, exposures_batch)

            b_train_loss += train_loss
            b_train_loss_param += train_loss_param
            b_train_loss_exp += train_loss_exp
            b_train_loss_albedo += train_loss_albedo
            b_train_loss_albedo_full_cycle += train_loss_albedo_full_cycle

        b_train_loss /= len(train_loader)
        b_train_loss_param /= len(train_loader)
        b_train_loss_exp /= len(train_loader)
        b_train_loss_albedo /= len(train_loader)
        b_train_loss_albedo_full_cycle /= len(train_loader)

        train_losses.append(b_train_loss)
        train_losses_param.append(b_train_loss_param)
        train_losses_exp.append(b_train_loss_exp)
        train_losses_albedo.append(b_train_loss_albedo)
        train_losses_albedo_full_cycle.append(b_train_loss_albedo_full_cycle)

        # validation
        with torch.no_grad():
            # batch losses
            b_test_loss, b_test_loss_param, b_test_loss_exp = 0.0, 0.0,  0.0
            b_test_loss_albedo, b_test_loss_albedo_full_cycle = 0.0, 0.0

            for params_batch, spec_albedos_batch, rgb_albedos_batch, exposures_batch in val_loader:
                params_batch = params_batch.to(device)
                rgb_albedos_batch = rgb_albedos_batch.to(device)
                spec_albedos_batch = spec_albedos_batch.to(device)
                exposures_batch = exposures_batch.to(device)

                test_loss, test_loss_param, test_loss_exp, test_loss_albedo, test_loss_albedo_full_cycle \
                    = test_step(params_batch, spec_albedos_batch, rgb_albedos_batch, exposures_batch)

                b_test_loss += test_loss
                b_test_loss_param += test_loss_param
                b_test_loss_exp += test_loss_exp
                b_test_loss_albedo += test_loss_albedo
                b_test_loss_albedo_full_cycle += test_loss_albedo_full_cycle

            b_test_loss /= len(val_loader)
            b_test_loss_param /= len(val_loader)
            b_test_loss_exp /= len(val_loader)
            b_test_loss_albedo /= len(val_loader)
            b_test_loss_albedo_full_cycle /= len(val_loader)

            test_losses.append(b_test_loss)
            test_losses_param.append(b_test_loss_param)
            test_losses_exp.append(b_test_loss_exp)
            test_losses_albedo.append(b_test_loss_albedo)
            test_losses_albedo_full_cycle.append(b_test_loss_albedo_full_cycle)

        # save logs
        save_log_losses_exp(optimizer.param_groups[0]['lr'], args.writing_rate, epoch, log_file, writer,
                        train_losses, train_losses_param, train_losses_exp,
                        train_losses_albedo, train_losses_albedo_full_cycle,
                        test_losses, test_losses_param, test_losses_exp,
                        test_losses_albedo, test_losses_albedo_full_cycle)

        plot_losses(epoch, args.writing_rate, args.loss_graph_path, train_losses, test_losses)

        # save model
        if epoch % args.writing_rate == 0:
            torch.save(model.module.state_dict(), args.output_model_folder + '/epoch' + str(epoch) + '.pt')

            if b_test_loss < best_test_loss:
                best_epoch = epoch

        # EARLY STOP / LR UPDATE
        if epoch > args.learning_rate_update:
            testlosscheck = test_losses[epoch - args.learning_rate_update:epoch]
            if abs(testlosscheck[0] - np.mean(testlosscheck)) < 0.001:
                shutil.copy(args.output_model_folder + '/epoch' + str(best_epoch) + '.pt',
                            args.output_model_folder + '.pt')
                writer.close()
                log_file.close()
                return

    shutil.copy(args.output_model_folder + '/epoch' + str(best_epoch) + '.pt', args.output_model_folder + '.pt')
    writer.close()
    log_file.close()
    return 1


if __name__ == '__main__':
    args = parse_arguments()
    run(args)

