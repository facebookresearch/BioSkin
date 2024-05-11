# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import numpy as np
import argparse
import json
import bioskin.utils.io as io
import bioskin.bioskin as bioskin
from bioskin.character import Character
from bioskin.parameters.parameter import SKIN_PROPS
from bioskin.parameters.parameter import REGULAR_SKIN_MIN_VALUES
from bioskin.parameters.parameter import REGULAR_SKIN_MAX_VALUES


def parse_arguments():
    parser = argparse.ArgumentParser(description='Augment face dataset by varying estimated skin properties')

    parser.add_argument('--json_model',  nargs='?',
                        type=str, default='../pretrained_models/BioSkinAO',
                        # type=str, default='../pretrained_models/BioSkin',
                        help='Json file with trained nn model description and params')

    # folder to reconstruct
    parser.add_argument('--input_folder',  nargs='?',
                        type=str, default='../input_images/',
                        help='Input Folder with diffuse albedos to reconstruct')

    parser.add_argument('--output_folder',  nargs='?',
                        type=str, default='../results/augmented_albedos/',
                        help='Output folder with results of augmentation')

    # skin props sampling parameters
    parser.add_argument('--nM',  nargs='?',
                        type=int, default=6,
                        help='Number of samples (melanin)')

    parser.add_argument('--nB',  nargs='?',
                        type=int, default=6,
                        help='Number of samples (hemoglobin)')

    parser.add_argument('--nT',  nargs='?',
                        type=int, default=4,
                        help='Number of samples (thickness)')

    parser.add_argument('--nE',  nargs='?',
                        type=int, default=3,
                        help='Number of samples (eumelanin/pheomelanin ratio)')

    parser.add_argument('--nO',  nargs='?',
                        type=int, default=3,
                        help='Number of samples (blood oxygenation)')

    parser.add_argument("--ranges",
                        nargs=10,
                        metavar=('Mmin', 'Mmax', 'Bmin', 'Bmax', 'Tmin', 'Tmax', 'Omin', 'Omax', 'Emin', 'Emax'),
                        help="input parameters as floats (min, max) for Melanin, Bood, Thickness, Oxygenation, Eumelanin",
                        type=float,
                        default=None)

    parser.add_argument('--max_width', nargs='?',
                        type=int, default=2048,
                        help='Maximum image width for interactive demo. '
                             'Edited images can still be stored at the original resolution')

    args = parser.parse_args()
    return args


def sample_linear(x, min, max):
    return min + x * (max - min)


def sample_exponential(x, min, max, b):
    return min + (max - min) * np.power(x, b)


def build_parameter_combinations(component_resolutions, output_path):
    # build the combinatorial samples
    melanins = np.linspace(REGULAR_SKIN_MIN_VALUES[0], REGULAR_SKIN_MAX_VALUES[0], component_resolutions[0])
    haes = np.linspace(REGULAR_SKIN_MIN_VALUES[1], REGULAR_SKIN_MAX_VALUES[1], component_resolutions[1])
    thicknesses = np.linspace(REGULAR_SKIN_MIN_VALUES[2], REGULAR_SKIN_MAX_VALUES[2], component_resolutions[2])
    oxys = np.linspace(REGULAR_SKIN_MIN_VALUES[3], REGULAR_SKIN_MAX_VALUES[3], component_resolutions[3])
    melratios = np.linspace(REGULAR_SKIN_MIN_VALUES[4], REGULAR_SKIN_MAX_VALUES[4], component_resolutions[4])

    parameter_combinations = []

    for mr in melratios:
        for oxy in oxys:
            for th in thicknesses:
                for hae in haes:
                    for mel in melanins:
                        parameter_tuple = [mel, hae, th, oxy, mr]
                        if parameter_tuple not in parameter_combinations:
                            parameter_combinations.append(parameter_tuple)

    with open(output_path + '.json', 'w') as json_file:
        json.dump(parameter_combinations, json_file)

    return np.array(parameter_combinations)


def augment_dataset(device, bio_skin_model, component_resolutions, input_folder, output_folder, max_width):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print('Generating new variations...')
    character_names, extensions = io.get_file_list(input_folder)
    print("Found {} characters in the input directory. Estimating skin properties...".format(len(character_names)))

    for character_name, extension in zip(character_names, extensions):
            print("Loading character {}".format(character_name))
            subject = Character(character_name, input_folder, device, extension)
            subject.load_input_albedo(max_width=max_width)
            skin_props = subject.estimate_skin_props(bio_skin_model)
            subject.reconstruct_albedo_from_skin_props(bio_skin_model, skin_props, original=True)

            parameter_combinations = build_parameter_combinations(component_resolutions, output_folder + character_name)

            for i in range(0, parameter_combinations.shape[0]):
                # define values depending on the average level
                c_index = 0
                print('Combination ' + str(i) + ' of ' + str(parameter_combinations.shape[0]))
                subject.reset_params()
                for param_index in range(0, parameter_combinations.shape[1]):
                    target_value = parameter_combinations[i, param_index]
                    parameter_map = subject.get_param_by_index(param_index)
                    mode = np.median(parameter_map)
                    value = target_value / mode
                    print("mode " + str(mode) + ", variation " + str(target_value) + ", value " + str(value))
                    # value = SKIN_MIN_VALUES[param_index] + variation * SKIN_MAX_VALUES[param_index]
                    subject.edit(SKIN_PROPS[param_index], "multiply_raw", value)
                    c_index += 1

                subject.reconstruct_albedo(bio_skin_model, original=False)

                filename = output_folder + character_name + '_' + str(i) \
                           + '_m' + "{:.2f}".format(parameter_combinations[i, 0])\
                           + '_b' + "{:.2f}".format(parameter_combinations[i, 1])\
                           + '_t' + "{:.2f}".format(parameter_combinations[i, 2])\
                           + '_o' + "{:.2f}".format(parameter_combinations[i, 3])\
                           + '_e' + "{:.2f}".format(parameter_combinations[i, 4]) + '.exr'
                subject.save_edited_albedo(filename)
                subject.save_edited_albedo_jpeg(filename)
                # subject.save_reconstruction(bio_skin_model, filename)


if __name__ == '__main__':

    args = parse_arguments()

    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")

    bio_skin_model = bioskin.BioSkinInference(args.json_model, device=device)

    component_resolutions = [args.nM, args.nB, args.nT, args.nE, args.nO]
    augment_dataset(device, bio_skin_model, component_resolutions, args.input_folder, args.output_folder, args.max_width)
