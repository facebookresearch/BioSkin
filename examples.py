# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import argparse
from argparse import RawTextHelpFormatter
from bioskin.bioskin import BioSkinInference
from bioskin.bioskin import get_unit_data_size
from bioskin.bioskin import save_reconstruction
import bioskin.utils.io as io
from bioskin.apps.homogenize_skin import homogenize_skins
from bioskin.apps.optimize_color_correction import color_correction
from bioskin.apps.dataset_augment_skin_props import augment_dataset
from bioskin.apps.optimize_biomaps import optimize_bio_maps
import bioskin.apps.manifold_sampling as ms
import bioskin.bioskin_gui as gui


options = ["GUI",  # Interactive skin property estimation and editing
           "Reconstruction",  # Estimates Skin properties from Diffuse Albedo Textures
           "Sampler",  # Skin Tones Sampler
           "Augment",  # Augment Dataset with Skin Properties
           "OptimizeCCM",  # Optimize Color Correction Matrix
           "Homogenize",  # Homogenize Skins: match all to a target (the last one loaded)
           "OptimizeProps"]  # Optimize Bio Maps using the decoder only (skin properties --> reflectance spectra)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Reconstruct skin props using learned map Reflectance->SkinProps->Reflectance',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('--path_to_model', nargs='?',
                        type=str, default='pretrained_models/',
                        help='Folder to Json files with pretrained nn models')

    parser.add_argument('--json_model', nargs='?',
                        type=str, default='BioSkinAO',
                        help='Json file with trained nn model. Options:\n'
                             'BioSkinAO : <rgb> to <skin properties, occlusion> to <spectrum>\n'
                             'BioSkin: <rgb> to <skin properties> to <spectrum>\n'
                             'BioSkinRGB: <rgb> to <skin properties> to <rgb>\n')

    parser.add_argument('--input_folder',  nargs='?',
                        type=str, default='test_images/',
                        help='Input Folder with diffuse albedos to reconstruct')

    parser.add_argument('--output_folder',  nargs='?',
                        type=str, default='results/',
                        help='Output folder with skin properties and reconstructed albedos')

    parser.add_argument('--mode',  nargs='?',
                        type=str, choices=['GUI', 'Reconstruction', 'Sampler', 'Augment', 'OptimizeCCM',
                                           'Homogenize', 'OptimizeProps', 'AddSpecularReflection'],
                        default='Reconstruction',
                        help='GUI: Interactive skin property estimation and editing\n'
                             'Reconstruction: Estimates Skin properties from Diffuse Albedo Textures\n'
                             'Sampler: Skin Tones Sampler\n'
                             'Augment: Augment Dataset with Skin Properties\n'
                             'OptimizeCCM: Optimize Color Correction Matrix\n'
                             'Homogenize: Homogenize Skins: match all to a target (the last one loaded)\n'
                             'OptimizeProps: Optimize Bio Maps using the decoder only '
                             '(skin properties --> reflectance spectra)\n')

    parser.add_argument('--max_width', nargs='?',
                        type=int, default=800,
                        help='Maximum image width. '
                             'In GUI mode, edited images can still be stored at the original resolution')

    parser.add_argument('--batch_size', nargs='?', type=int, default=512000)

    parser.add_argument('--save_spectrum',
                        action='store_true',
                        default=False,
                        help='Save reflectance spectra to .npy')

    # sampler arguments
    parser.add_argument('--color_space',  nargs='?', type=str, choices=['RGB', 'LAB'], default='LAB')
    parser.add_argument('--num_samples',  nargs='?', type=int, default=10000000)
    parser.add_argument('--sampling_scheme',  nargs='?', type=str, choices=['Random', 'QuasiRandom'], default='Random')
    parser.add_argument('--sampling_mode',  nargs='?', type=str,
                        choices=['UniformColor', 'SkinProps', 'LabSlices'], default='UniformColor',
                        help='UniformColor: Interactive skin property estimation and editing\n'
                             'SkinProps: Estimates Skin properties from Diffuse Albedo Textures\n'
                             'LabSlices: Skin Tones Sampler\n')
    parser.add_argument('--ranges',  nargs='?',
                        type=str, choices=['RegularSkin', 'AllSkin', 'FullRange'], default='RegularSkin',
                        help='RegularSkin: Ranges for plausible skin tones, leaving aside imperfections\n'
                             'AllSkin: Ranges for skin, including melanin spots, rushes, or thinner areas (lips)\n'
                             'FullRange: Ranges beyond regular skin to recover facial hairs and other outliers\n')

    # skin props sampling parameters for data augmentation
    parser.add_argument('--nM',  nargs='?', type=int, default=5, help='Number of samples (melanin)')
    parser.add_argument('--nB',  nargs='?', type=int, default=4, help='Number of samples (hemoglobin)')
    parser.add_argument('--nT',  nargs='?', type=int, default=3, help='Number of samples (thickness)')
    parser.add_argument('--nE',  nargs='?', type=int, default=2, help='Number of samples (eumelanin/pheomelanin ratio)')
    parser.add_argument('--nO',  nargs='?', type=int, default=2, help='Number of samples (blood oxygenation)')

    # color correction arguments
    parser.add_argument('--regularization',  nargs='?', type=float, default=0.2, help='Regularization weight')
    parser.add_argument('--resizing_factor',  nargs='?', type=float, default=0.1,
                        help='Resizing factor of input calibration images')
    parser.add_argument('--exposure_correction',  nargs='?', type=float, default=1.0,
                        help='Correcting exposure of calibration images')

    #  optimize bio maps arguments
    parser.add_argument('--num_epochs', nargs='?',
                        type=int, default=200,
                        help='Number of optimization epochs')

    parser.add_argument('--output_ratio', nargs='?',
                        type=int, default=20,
                        help='Number of epochs to save results')
    args = parser.parse_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    return args


def main():
    args = parse_arguments()

    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")

    bio_skin = BioSkinInference(args.path_to_model + args.json_model, device=device, batch_size=args.batch_size)

    # Launching GUI
    if args.mode == options[0]:
        print(options[0])
        gui.run_gui(args)

    # Example of skin property estimation and reflectance up-sampling
    elif args.mode == options[1]:
        print(options[1])
        filename_list_no_ext, extensions = io.get_file_list(args.input_folder)
        for index, filename_no_ext in enumerate(filename_list_no_ext, start=0):
            print('-- Loading ' + filename_no_ext + ', ' + str(index+1) + ' of ' + str(len(filename_list_no_ext)))
            path_to_image = os.path.join(args.input_folder, filename_no_ext + extensions[index])
            input_image = io.load_image(path_to_image, max_width=args.max_width)
            input_reflectance = io.vectorize_image(input_image, device=device)

            required_memory = input_reflectance.shape[0] * get_unit_data_size(bio_skin) * 4
            available_memory = torch.cuda.get_device_properties(device).total_memory
            if required_memory > available_memory:
                print("*** Switching to CPU! Required memory of " + str(str(required_memory / 1024 ** 3))
                      + " GB (max. available " + str(available_memory / 1024 ** 3) + ' GB)')
                device = torch.device('cpu')
                bio_skin = BioSkinInference(args.json_model, device=device, batch_size=args.batch_size)

            # running encoder (reflectance --> skin properties)
            skin_props = bio_skin.reflectance_to_skin_props(input_reflectance)
            # running decoder (skin properties --> reflectance)
            ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg = bio_skin.skin_props_to_reflectance(skin_props)
            reconstruction_error = abs(ref_vis_rgb - input_reflectance)

            output_folder = args.output_folder + 'reconstruction/' + \
                            os.path.basename(os.path.normpath(args.json_model)) + '/'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            save_reconstruction(bio_skin.model_params, output_folder + filename_no_ext, "",
                                input_image, skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error,
                                save_spectrum=args.save_spectrum, save_plot=True)

    # Example of generating skin tones by randomly sampling skin properties and running decoder network
    elif args.mode == options[2]:
        print(options[2])
        if args.json_model == "BioSkinRGB":
            print("ERROR: Manifold visualization only for spectral models (BioSkin, BioSkinAO)")
            exit(0)
        if args.sampling_mode == "UniformColor":
            skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg = \
                ms.manifold_sample(args.json_model, bio_skin, args.output_folder, args.sampling_scheme, args.ranges,
                                args.num_samples, args.color_space, device, plotting=False)
        elif args.sampling_mode == "SkinProps":
            skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg = \
                ms.skin_properties_sample(args.json_model, bio_skin, args.output_folder, args.ranges, args.num_samples,
                                       args.color_space)
        elif args.sampling_mode == "LabSlices":
            skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg = \
                ms.manifold_sample_Lab_slices(args.json_model, bio_skin, args.output_folder, args.sampling_scheme,
                                              args.ranges, args.num_samples, args.color_space, device)
        else:
            print("Choose sampling mode: 'UniformColor|SkinProps'")
            exit(0)

        ms.add_specular_reflection_spectral(args.json_model, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, skin_props,
                                            args.output_folder, device, num_hemispheric_samples=90, R0_Schick=0.04)

    # Augment Dataset with Skin Properties
    elif args.mode == options[3]:
        print(options[3])
        component_resolutions = [args.nM, args.nB, args.nT, args.nE, args.nO]
        augment_dataset(device, bio_skin, component_resolutions, args.input_folder,
                        args.output_folder + 'augmented_skins/', args.max_width)

    # Optimize Color Correction Matrix
    elif args.mode == options[4]:
        print(options[4])
        output_folder = args.output_folder + 'optimize_ccm/' + \
                        os.path.basename(os.path.normpath(args.json_model)) + '/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        color_correction(bio_skin, device, args.input_folder, output_folder, args.batch_size, args.regularization,
                         args.resizing_factor, args.exposure_correction)

    # Homogenize Skins
    elif args.mode == options[5]:
        print(options[5])
        output_folder = args.output_folder + 'homogenize_skins/' + \
                        os.path.basename(os.path.normpath(args.json_model)) + '/'
        homogenize_skins(bio_skin, args.input_folder, output_folder, args.max_width)

    # Optimize Bio Maps
    elif args.mode == options[6]:
        print(options[6])
        output_folder = args.output_folder + 'optimize_bio_maps/' + \
                        os.path.basename(os.path.normpath(args.json_model)) + '/'
        optimize_bio_maps(bio_skin, device, args.input_folder, output_folder, args.regularization, args.num_epochs,
                          args.max_width, args.output_ratio)

    else:
        print("Choose an option with --mode <Option>")
        print(options)


if __name__ == '__main__':
    main()
