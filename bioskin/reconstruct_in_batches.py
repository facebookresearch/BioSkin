# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import os
import sys
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import cv2
import numpy as np
import argparse
import math
import shutil
import bioskin.bioskin as bioskin
import bioskin.utils.io as io
from bioskin.parameters.parameter import SKIN_PROPS
import bioskin.parameters.params_io as params_io


def parse_arguments():
    # Instantiate the parser
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

    parser.add_argument('--device',  nargs='?',
                        type=str, default='cuda',
                        help='cuda | cpu')

    parser.add_argument('--batch_size',  nargs='?',
                        type=int, default=512,
                        help='Batch size to eval the network')

    parser.add_argument("--export_spectrums", default=False, action="store_true",
                        help="Export per pixel spectral reflectance")

    args = parser.parse_args()
    return args


def split_image_into_batches(image, image_path, extension, batch_size):
    print("Splitting Image into batches...")
    batches = []
    rows = 0
    total_batches = math.ceil(image.shape[0] / batch_size) * math.ceil(image.shape[1] / batch_size)
    current_batch = 0
    for i in range(0, image.shape[0], batch_size):
        columns = 0
        for j in range(0, image.shape[1], batch_size):
            batch = image[i:i + batch_size, j:j + batch_size]
            batches.append(batch)
            folder_name = os.path.basename(os.path.normpath(image_path))
            path = image_path + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            filename = path + folder_name + "_batch" + str(rows) + str(columns) + extension
            io.save_image(batch, filename, verbose=False)
            columns += 1
            current_batch += 1
            sys.stdout.write(f"\rCreating batch {current_batch} of {total_batches}")
            sys.stdout.flush()
        rows += 1
    sys.stdout.write(f"\nDONE ({path}{folder_name}_batchXX)\n")
    sys.stdout.flush()
    return rows, columns, path, folder_name


def recompose_image(bio_skin, folder, filename, rows, columns, original_image_shape, num_props):
    print("Stitching reconstructed batches...")
    recomposed_image = np.zeros(original_image_shape)
    result_types = ['_reconstruction_vis', '_reconstruction_diff']
    for i in range(0, bio_skin.model_params.D_skin):
        result_types.append('_p' + str(i) + '_' + SKIN_PROPS[i])
    if bio_skin.model_params.color_mode == 'spectral':
        result_types.append('_reconstruction_IR')
        if hasattr(bio_skin.model_params, "exposure_aware"):
            result_types.append('_reconstruction_occlusion')
            result_types.append('_reconstruction_vis_normalized')

    extensions = ['.exr', '.jpeg']
    for index, result_type in enumerate(result_types):
        for index_ext, extension in enumerate(extensions):
            current_batch = 0
            total_batches = rows * columns
            for i in range(0, rows):
                for j in range(0, columns):
                    path = folder + filename + "_batch" + str(i) + str(j) + result_type + extension
                    batch = io.load_image(path, verbose=False)
                    if current_batch == 0:
                        batch_size_x, batch_size_y = batch.shape[0], batch.shape[1]
                        residual_batch_size_x, residual_batch_size_y = batch_size_x, batch_size_y
                    else:
                        residual_batch_size_x, residual_batch_size_y = batch.shape[0], batch.shape[1]

                    if len(batch.shape) == 2:
                        batch = np.repeat(batch[:, :, np.newaxis], 3, axis=2)

                    recomposed_image[i*batch_size_x: i*batch_size_x + residual_batch_size_x,
                                        j*batch_size_y: j*batch_size_y + residual_batch_size_y, :] = batch

                    current_batch += 1
                    sys.stdout.write(f"\rStitching batch {current_batch} of {total_batches} ({index} / {len(result_types)} result)")
                    sys.stdout.flush()
            folder_parent = os.path.dirname(folder)
            cv2.imwrite(folder_parent + result_type + extension, np.float32(recomposed_image))
            io.save_jpeg(folder_parent + result_type, np.float32(recomposed_image), linear_input=True)
    sys.stdout.write(f"\nDONE ({folder_parent})\n")
    sys.stdout.flush()


def reconstruct_in_batches(device, path_to_model, json_model, bio_skin, input_folder, output_folder, batch_size,
                           export_spectrums):

    model_params_dict = vars(bio_skin.model_params)
    stats = model_params_dict.copy()
    error_average = np.double(0.0)
    error_max = 0.0
    error_min = 10000
    file_log = 0

    i = 0
    filename_list_no_ext, extensions = io.get_file_list(input_folder)
    if len(filename_list_no_ext) == 0:
        print("Empty input folder!")
        exit(0)

    for filename_no_ext in filename_list_no_ext:
        print('\n\n--> Processing texture ' + str(i+1) + ' of ' + str(len(filename_list_no_ext)))
        print(filename_no_ext)

        path_to_image = os.path.join(input_folder, filename_no_ext + extensions[i])
        input_image = io.load_image(path_to_image, verbose=False)

        path_no_extension = os.path.join(input_folder, filename_no_ext)
        rows, columns, path_input_batches, input_name = split_image_into_batches(input_image, path_no_extension,
                                                                                 extensions[i], batch_size)

        path = output_folder + json_model + '/' + input_name + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        print("Reconstructing batches...")
        current_batch = 1
        total_batches = rows * columns
        for i_batch in range(0, rows):
            for j_batch in range(0, columns):
                filename = path_input_batches + input_name + "_batch" + str(i_batch) + str(j_batch) + extensions[i]
                input_image_batch = io.load_image(filename, verbose=False)
                input_reflectance = io.vectorize_image(input_image_batch, device=device)

                skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error = \
                    bio_skin.reconstruct(input_reflectance)

                filename_out = filename_no_ext + "_batch" + str(i_batch) + str(j_batch)
                bioskin.save_reconstruction(bio_skin.model_params, path + filename_out, "", input_image_batch,
                                    skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error,
                                    save_spectrum=export_spectrums)
                current_batch += 1
                sys.stdout.write(f"\rReconstructing batch {current_batch} of {total_batches}")
                sys.stdout.flush()

        sys.stdout.write(f"\nDONE ({path + filename_no_ext})\n")
        sys.stdout.flush()

        recompose_image(bio_skin, path, filename_no_ext, extensions[i], rows, columns, input_image.shape,
                        bio_skin.model_params.D_skin)

        shutil.rmtree(path_input_batches)
        shutil.rmtree(path)

        print("\rProgress: ", 100.0, "%", end="\n")
        if not file_log:
            file_log = open(output_folder + json_model + '/' + 'LOG.txt', 'a')

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

    print("====== AVERAGE STATS: Mean Abs Error: " + str(error_average))
    file_log.write("====== AVERAGE STATS: Mean Abs Error: " + str(error_average))
    file_log.write("\n")

    if math.isnan(error_average):
        print('WARNING: Nan')
    stats["model"] = path_to_model + json_model
    stats["error_max"] = float(error_max)
    stats["error_min"] = float(error_min)
    file_log.close()
    return stats


def reconstruct_in_batches_with_ops(device, path_to_model, json_model, bio_skin, input_folder, output_folder,
                                    batch_size, export_spectrums, characters=None, character_name=None,
                                    progress_bar=None):

    model_params_dict = vars(bio_skin.model_params)
    stats = model_params_dict.copy()
    error_average = np.double(0.0)
    error_max = 0.0
    error_min = 10000
    file_log = 0

    i = 0
    filename_list_no_ext, extensions = io.get_file_list(input_folder)
    if len(filename_list_no_ext) == 0:
        print("Empty input folder!")
        exit(0)
    elif character_name:
        index = filename_list_no_ext.index(character_name)
        filename_list_no_ext = [filename_list_no_ext[index]]
        extensions = [extensions[index]]

    if progress_bar is not None:
        progress_bar.setValue(0)

    for index, filename_no_ext in enumerate(filename_list_no_ext):
        print('\n\n--> Processing texture ' + str(i+1) + ' of ' + str(len(filename_list_no_ext)))
        print(filename_no_ext)

        ch = characters[character_name if character_name else index]

        path_to_image = os.path.join(input_folder, filename_no_ext + extensions[i])
        input_image = io.load_image(path_to_image, verbose=False)

        path_no_extension = os.path.join(input_folder, filename_no_ext)
        rows, columns, path_input_batches, input_name = split_image_into_batches(input_image, path_no_extension,
                                                                                 extensions[i], batch_size)
        if progress_bar is not None:
            progress_bar.setValue(10)

        if json_model is not None:
            path = output_folder + json_model + '/' + input_name + '/'
        else:
            path = output_folder + input_name + '/'

        if not os.path.exists(path):
            os.makedirs(path)

        print("Reconstructing batches...")
        current_batch = 1
        total_batches = rows * columns
        for i_batch in range(0, rows):
            for j_batch in range(0, columns):
                filename = path_input_batches + input_name + "_batch" + str(i_batch) + str(j_batch) + extensions[i]
                input_image_batch = io.load_image(filename, verbose=False)
                input_reflectance = io.vectorize_image(input_image_batch, device=device)

                skin_props = bio_skin.reflectance_to_skin_props(input_reflectance)
                if characters:
                    skin_props = params_io.warp_parameter_maps(skin_props)
                    skin_props = ch.apply_operators_external_skin_props(skin_props)
                    skin_props = params_io.unwarp_parameter_maps(skin_props)
                ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg = bio_skin.skin_props_to_reflectance(skin_props)
                reconstruction_error = abs(ref_vis_rgb.to(bio_skin.device) - input_reflectance.to(bio_skin.device))

                filename_out = filename_no_ext + "_batch" + str(i_batch) + str(j_batch)
                bioskin.save_reconstruction(bio_skin.model_params, path + filename_out, "", input_image_batch,
                                    skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error,
                                    save_spectrum=export_spectrums)
                current_batch += 1
                sys.stdout.write(f"\rReconstructing batch {current_batch} of {total_batches}")
                sys.stdout.flush()

                if progress_bar is not None:
                    value = int(np.round(10 + 70.0*(current_batch/total_batches)))
                    progress_bar.setValue(value)

        sys.stdout.write(f"\nDONE ({path + filename_no_ext})\n")
        sys.stdout.flush()

        if progress_bar is not None:
            progress_bar.setValue(80)

        # stitch reconstructed patches into the final image
        recompose_image(bio_skin, path, filename_no_ext, rows, columns, input_image.shape, bio_skin.model_params.D_skin)

        if progress_bar is not None:
            progress_bar.setValue(100)

        shutil.rmtree(path_input_batches)
        shutil.rmtree(path)

        print("\rProgress: ", 100.0, "%", end="\n")
        if not file_log:
            if json_model is not None:
                log_path = output_folder + json_model + '/' + 'LOG.txt'
            else:
                log_path = output_folder + 'LOG.txt'
            file_log = open(log_path, 'a')

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

        if character_name:
            break

    error_average /= float(len(filename_list_no_ext))

    print("====== AVERAGE STATS: Mean Abs Error: " + str(error_average))
    file_log.write("====== AVERAGE STATS: Mean Abs Error: " + str(error_average))
    file_log.write("\n")

    if math.isnan(error_average):
        print('WARNING: Nan')
    model_text = path_to_model
    if json_model is not None:
        model_text += json_model
    stats["model"] = model_text
    stats["error_max"] = float(error_max)
    stats["error_min"] = float(error_min)
    file_log.close()
    return stats

if __name__ == '__main__':
    args = parse_arguments()
    print(torch.__version__)
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")
    else:
        device = torch.device('cpu')
        print("Device: " + str(device))

    bio_skin = bioskin.BioSkinInference(args.path_to_model + args.json_model, device=device, batch_size=args.batch_size)
    reconstruct_in_batches(device, args.path_to_model, args.json_model, bio_skin, args.input_folder, args.output_folder,
                           args.batch_size, args.export_spectrums)




