# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Differences in parameter maps')

    parser.add_argument('--input_folder',  nargs='?',
                        type=str, default='../input_images/',
                        help='Input Folder with diffuse albedos to reconstruct')

    args = parser.parse_args()
    return args


def run(args):
    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")


    filenamelist = os.listdir(args.input_folder)  # ["ronald_albedo_baked", "yaser_albedo_baked"]
    filenamelist_no_ext = filenamelist.copy()
    for i in range(0, len(filenamelist)):
        filenamelist_no_ext[i] = os.path.splitext(filenamelist[i])[0]

    filenamelist_no_ext.sort()
    first_component = filenamelist_no_ext[0]  # Get the first component
    filenamelist_no_ext = filenamelist_no_ext[1:] + [first_component]

    error_average = np.double(0.0)
    mse_average = np.double(0.0)
    error_max = 0.0
    error_min = 10000

    PATH_TO_EXR_FILE = os.path.join(args.input_folder, filenamelist_no_ext[2] + ".exr")
    ref_image = cv2.imread(PATH_TO_EXR_FILE, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    row, col, channel = ref_image.shape
    # reshape to nx3
    ref_image = ref_image[:, :, 0]
    ref_image_vec = ref_image.reshape((row * col))
    ref_image_vec = ref_image_vec[~np.isnan(ref_image_vec)]
    indexes = ref_image_vec > 0.001
    ref_image_vec = ref_image_vec[indexes]

    length = len(ref_image_vec)

    i = 0
    fig, ax = plt.subplots()

    for filename_no_ext in filenamelist_no_ext:
        PATH_TO_EXR_FILE = os.path.join(args.input_folder, filename_no_ext + ".exr")
        input_image = cv2.imread(PATH_TO_EXR_FILE, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        row, col, channel = input_image.shape
        # reshape to nx3
        input_image_vec = input_image[:, :, 0]
        input_image_vec = input_image_vec.reshape((row * col))
        input_image_vec = input_image_vec[~np.isnan(input_image_vec)]
        input_image_vec = input_image_vec[indexes]

        # compute errors and save results
        error_image = abs(input_image_vec - ref_image_vec)
        # diff_image = error_image.reshape((row, col, 3)).astype("float32")
        diff_image = error_image.astype("float32")
        diff_image = diff_image[~np.isnan(diff_image)]


        relative_error_image = error_image/ref_image_vec
        # rel_diff_image = relative_error_image.reshape((row, col, 3)).astype("float32")
        rel_diff_image = relative_error_image.astype("float32")
        rel_diff_image = rel_diff_image[~np.isnan(rel_diff_image)]

        current_error = np.double(torch.sum(torch.tensor(diff_image))) / np.double(length)
        abs_relative_diff = np.double(torch.sum(torch.tensor(rel_diff_image))) / np.double(length)
        # print("====== Abs. Rel. Error: " + filename_no_ext + " " + str(abs_relative_diff))

        print("====== ME: " + filename_no_ext + " " + str(current_error))
        current_mse = np.double(torch.sum(torch.tensor(diff_image*diff_image))) / np.double(length)
        # print("====== MSE: " + filename_no_ext + " " + str(current_mse))
        mse_average += current_mse
        error_average += current_error
        if math.isnan(error_average):
            print('WARNING: Nan')

        if current_error >= error_max:
            error_max = current_error
        if current_error < error_min:
            error_min = current_error

        # Calculate the histogram bins and frequencies
        hist, bins = np.histogram(input_image_vec, bins=10)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Define color temperatures in Kelvin
        color_temperatures = [2500, 4500, 6500, 8500, 10000]

        # Assign custom colors based on color temperatures
        colors = ['#FF6F40', '#FFB580', '#BEBEBE', '#8fd2ff', '#3FAFFF']

        # Plot the histogram as curves
        plt.plot(bin_centers, hist, '-o', color=colors[i], label=str(color_temperatures[i])+'K')

        i += 1

    property_name = 'EumelaninRatio'
    font_size = 16
    # Set custom size for axis numbers
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.legend(loc="upper right")
    plt.legend(fontsize=font_size)
    plt.xlabel('% concentration', fontsize=font_size)
    plt.title(property_name, fontsize=font_size)
    plt.savefig(args.input_folder + 'hists_' + property_name + '.jpeg', format='jpeg', dpi=500)
    plt.close()

    error_average /= float(len(filenamelist_no_ext))
    mse_average /= float(len(filenamelist_no_ext))

    print("====== AVERAGE STATS: MSE: " + str(mse_average) + ", Mean Error: " + str(error_average))


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
