# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import numpy as np
import argparse
import math
import bioskin.bioskin as BioSkin
import bioskin.utils.io as io


def parse_arguments():
    parser = argparse.ArgumentParser(description='Reconstruct skin props using learned map Reflectance->SkinProps->Reflectance')

    parser.add_argument('--path_to_model',  nargs='?',
                        type=str, default='pretrained_models/',
                        help='Json file with trained nn model description and params')

    parser.add_argument('--json_model',  nargs='?',
                        type=str, default='BioSkinAO',
                        help='Json file with trained nn model description and params')

    parser.add_argument('--input_folder',  nargs='?',
                        type=str, default='input_images/',
                        help='Input Folder with diffuse albedos to reconstruct')

    parser.add_argument('--output_folder',  nargs='?',
                        type=str, default='reconstruction_tests/',
                        help='Output folder with skin properties and reconstructed albedos')

    parser.add_argument('--prefix',  nargs='?',
                        type=str, default='',
                        help='Prefix to run remotely (dgx)')

    parser.add_argument('--batch_size',  nargs='?',
                        type=int, default=1048576,  # 1048576
                        help='Batch size to eval the network')

    parser.add_argument("--export_spectrums", default=False, action="store_true",
                        help="Export per pixel spectral reflectance to CSV (slow)")

    parser.add_argument("--export_spectral_bands", default=False, action="store_true",
                        help="Export per pixel spectral reflectance to CSV (slow)")

    parser.add_argument('--max_width', nargs='?',
                        type=int, default=2048,
                        help='Maximum image width for interactive demo. '
                             'Edited images can still be stored at the original resolution')

    args = parser.parse_args()
    return args


def reconstruct_image_batch(args):
    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")

    bio_skin = BioSkin.BioSkinInference(args.path_to_model + args.json_model, device=device, batch_size=args.batch_size)

    model_params_dict = vars(bio_skin.model_params)
    stats = model_params_dict.copy()
    error_average = np.double(0.0)
    error_max = 0.0
    error_min = 10000
    file_log = 0

    i = 0
    filename_list_no_ext, extensions = io.get_file_list(args.input_folder)
    if len(filename_list_no_ext) == 0:
        print("Empty input folder!")
        exit(0)

    for index, filename_no_ext in enumerate(filename_list_no_ext):
        print('\n\n--> Processing texture ' + str(i) + ' of ' + str(len(filename_list_no_ext)))
        print(filename_no_ext)

        path_to_image = os.path.join(args.input_folder, filename_no_ext + extensions[index])
        input_image = io.load_image(path_to_image, max_width=args.max_width)
        input_reflectance = io.vectorize_image(input_image, device=device)

        skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error = \
            bio_skin.reconstruct(input_reflectance)

        path = args.output_folder + args.json_model + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        BioSkin.save_reconstruction(bio_skin.model_params, path + filename_no_ext, "", input_image,
                            skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error,
                            save_spectrum=args.export_spectrums)

        if args.export_spectral_bands:
            BioSkin.export_spectral_bands(path + filename_no_ext + '_IR_', device, ref_ir, input_image)

        if device == torch.device('cuda'):
            torch.cuda.empty_cache()

        print("\rProgress: ", 100.0, "%", end="\n")
        if not file_log:
            file_log = open(path + 'LOG.txt', 'a')

        length = reconstruction_error.size()[0]
        current_error = np.double(torch.sum(reconstruction_error)) / np.double(length)
        print("====== MAE: " + filename_no_ext + " " + str(current_error))
        file_log.write("====== MAE: " + filename_no_ext + " " + str(current_error))

        if math.isnan(current_error):
            print('WARNING: Nan in reconstruction diff')
        else:
            error_average += current_error

        if current_error >= error_max:
            error_max = current_error
        if current_error < error_min:
            error_min = current_error
        i += 1

    error_average /= float(len(filename_list_no_ext))

    print("\n\n====== AVERAGE STATS: Mean Abs Error: " + str(error_average))
    file_log.write("====== AVERAGE STATS: Mean Abs Error: " + str(error_average))
    file_log.write("\n")

    if math.isnan(error_average):
        print('WARNING: Nan')
    stats["model"] = args.path_to_model + args.json_model
    stats["error_max"] = float(error_max)
    stats["error_min"] = float(error_min)
    file_log.close()
    return stats


if __name__ == '__main__':
    args = parse_arguments()
    reconstruct_image_batch(args)


