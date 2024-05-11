# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import argparse
import bioskin.utils.io as io
import bioskin.bioskin as bioskin
from bioskin.character import Character


def parse_arguments():
    parser = argparse.ArgumentParser(description='Optimizing a color correction matrix to minimize reconstruction error '
                                                 'of Skin property estimator assuming D65 uniform diffuse lighting')

    parser.add_argument('--json_model',  nargs='?',
                        type=str, default='../pretrained_models/BioSkinAO',
                        help='Json file with trained nn model description and params')

    parser.add_argument('--input_folder',  nargs='?',
                        type=str, default='../input_images/',
                        help='Input folder with diffuse albedos to be adapted/homogeneized')

    parser.add_argument('--output_folder',  nargs='?',
                        type=str, default='../results/homogenize_images/',
                        help='Output folder of homogeneized skin textures')

    parser.add_argument('--batch_size',  nargs='?',
                        type=int, default=65536,
                        help='Batch size to eval the network')

    parser.add_argument('--mode',  nargs='?',
                        type=str, default='means',
                        # type=str, default='hist',
                        help='Match skin property map through a) histogram matching or b) equalizing means')


    args = parser.parse_args()
    return args


def homogenize_skins(bio_skin_model, input_folder, output_folder, max_width, mode="means"):
    """ Matches skin tone of a set of albedo textures to the latest albedo texture found in input_folder"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    characters = []
    character_names, extensions = io.get_file_list(input_folder)
    print("Found {} characters in the input directory. Estimating skin properties...".format(len(character_names)))

    for character_name, extension in zip(character_names, extensions):
        print("Loading character {}".format(character_name))
        character = Character(character_name, input_folder, bio_skin_model.device, extension)
        character.load_input_albedo(max_width=max_width)
        character.estimate_skin_props(bio_skin_model)
        character.reconstruct_albedo(bio_skin_model, original=True)
        characters.append(character)

    reference_character = characters[len(characters)-1]

    for index, character in enumerate(characters):
        if index < len(characters) - 1:
            character.reset()
            character.match_reference(reference_character) if mode == "means" \
                else character.match_reference_hist(reference_character)
            character.reconstruct_albedo(bio_skin_model, original=False)
            character.save_reconstruction(bio_skin_model, output_folder + character.name + '_matching_' +
                                          reference_character.name)
            io.save_jpeg(output_folder + character.name + '_target', reference_character.albedo_map)


if __name__ == '__main__':
    args = parse_arguments()

    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")

    bio_skin_model = bioskin.BioSkinInference(args.json_model, device=device)

    homogenize_skins(bio_skin_model, args.input_folder, args.output_folder, args.max_width, args.mode)
