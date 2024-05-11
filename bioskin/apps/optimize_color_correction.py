# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import cv2
import bioskin.bioskin as bioskin
import bioskin.utils.io as io
from bioskin.spectrum.color_spectrum import linear_to_sRGB


def parse_arguments():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optimizing a color correction matrix to minimize reconstruction error '
                                                 'of Skin property estimator assuming D65 uniform diffuse lighting')

    parser.add_argument('--path_to_model',  nargs='?',
                        # type=str, default='../models/spectral_exposure_aware/',
                        type=str, default='../pretrained_models/',
                        help='Folder with pretrained models')

    parser.add_argument('--json_model',  nargs='?',
                        type=str, default='BioSkinAO',
                        help='Json file with trained nn model description and params')

    parser.add_argument('--input_folder',  nargs='?',
                        type=str, default='../input/input_images_calibration/',
                        # type=str, default='../input/input_images_calibration_all/MUGSY/',
                        help='Input Folder with diffuse albedos to reconstruct and use in the loss')

    parser.add_argument('--output_folder',  nargs='?',
                        type=str, default='../results/calibration/')

    parser.add_argument('--batch_size',  nargs='?',
                        type=int, default=256,
                        help='Batch size to eval the network')

    parser.add_argument('--regularization',  nargs='?',
                        type=int, default=0.1,
                        help='Regularization weight')

    parser.add_argument('--resizing_factor',  nargs='?',
                        type=int, default=0.2,
                        help='Resizing factor of input calibration images')

    parser.add_argument('--exposure_correction',  nargs='?',
                        type=int, default=1.0,
                        help='Correcting exposure of calibration images')

    args = parser.parse_args()

    return args


class ColorCorrectionModel(nn.Module):
    def __init__(self, optimize_exposure=False, optimize_offset=False, optimize_gamma=False):
        super(ColorCorrectionModel, self).__init__()
        self.color_matrix = nn.Parameter(torch.eye(3, dtype=torch.float32, requires_grad=True))
        self.optimize_exposure = optimize_exposure
        self.optimize_offset = optimize_offset
        self.optimize_gamma = optimize_gamma
        self.exposure = nn.Parameter(torch.tensor(1.2, dtype=torch.float32, requires_grad=optimize_exposure))
        self.offset = nn.Parameter(torch.tensor(-0.004, dtype=torch.float32, requires_grad=optimize_offset))
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=optimize_gamma))

    def forward(self, x):
        color_matrix = self.color_matrix.to(x.device)
        exp = self.exposure.to(x.device)
        gamma = self.gamma.to(x.device)
        offset = self.offset.to(x.device)

        if self.optimize_exposure:
            x = x * exp
        if self.optimize_offset:
                x = x + offset
        if self.optimize_gamma:
            x = torch.pow(x, gamma)
        x = torch.matmul(x, color_matrix)
        return x

    def print_ccm(self):
        print("Color Correction Matrix: ")
        print(self.color_matrix)
        if self.optimize_exposure:
            print("Exposure: ")
            print(self.exposure)
        if self.optimize_offset:
            print("Offset: ")
            print(self.offset)
        if self.optimize_gamma:
            print("Gamma: ")
            print(self.gamma)

    def save_ccm(self, f):
        print("Color Correction Matrix: ", file=f)
        print(self.color_matrix, file=f)
        if self.optimize_exposure:
            print("Exposure: ", file=f)
            print(self.exposure, file=f)
        if self.optimize_offset:
            print("Offset: ", file=f)
            print(self.offset, file=f)
        if self.optimize_gamma:
            print("Gamma: ", file=f)
            print(self.gamma, file=f)


def load_calibration_images(input_folder, size_ratio=1.0, exposure=1.0, correct_gamma=True, flip=False):
    filename_list = os.listdir(input_folder)
    filename_list = [filename for filename in filename_list if "_mask" not in filename]
    filename_list = [filename for filename in filename_list if "." in filename]
    if '.DS_Store' in filename_list:  # This is for Mac OS users
        filename_list.remove('.DS_Store')

    input_images = []
    for filename in filename_list:
        path_to_file = os.path.join(input_folder, filename)
        image = io.load_image(path_to_file)
        image = image*exposure
        if correct_gamma:
            # gamma = 2.2
            # image = np.power(image, gamma)
            image = linear_to_sRGB(image)
        if flip:
            image = cv2.flip(image, 0)
        image = cv2.resize(image, (int(image.shape[1]*size_ratio), int(image.shape[0]*size_ratio)),
                           interpolation=cv2.INTER_CUBIC)
        input_images.append(image)

    return input_images


def concat_calibration_images(images, output_folder):
    if len(images) <= 1:
        combined_image = images[0]
        total_height = images[0].shape[0]
        total_width = images[0].shape[1]
    else:
        # Calculate the combined image dimensions
        total_height = max(image.shape[0] for image in images)
        total_width = sum(image.shape[1] for image in images)

        # Create a blank combined image with the determined dimensions
        combined_image = np.zeros((total_height, total_width, 3)).astype("float32")

        # Start position for the next image
        x_offset = 0

        # Copy each individual image into the combined image
        for image in images:
            height, width, _ = image.shape
            combined_image[:height, x_offset:x_offset + width, :] = image
            x_offset += width

        cv2.imwrite(output_folder + '0_input_non_corrected.exr', combined_image)
        # Display the combined image in a window
        display = False
        if display:
            cv2.imshow('Combined Image', combined_image*255.0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return combined_image, total_height, total_width


def colorfulness(image):
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    std_r = torch.std(r)
    std_g = torch.std(g)
    std_b = torch.std(b)
    return torch.sqrt((2 * std_r + 2 * std_g + std_b) / 3)


def gamut_volume(image):
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    max_r, min_r = torch.max(r), torch.min(r)
    max_g, min_g = torch.max(g), torch.min(g)
    max_b, min_b = torch.max(b), torch.min(b)
    return (max_r - min_r) * (max_g - min_g) * (max_b - min_b)


def rms_contrast(image):
    mean_intensity = torch.mean(image)
    return torch.sqrt(torch.mean((image - mean_intensity) ** 2))


def exposure(image):
    return torch.mean(image)


def forward_pass_and_compute_loss(input_image, ccm_model, device, bio_skin, k):
    # reshape to nx3
    input_reflectance = io.vectorize_image(input_image, device=device)
    input_reflectance_cc = ccm_model.forward(input_reflectance)

    skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error = bio_skin.reconstruct(input_reflectance_cc)

    total = 0.0
    regularization = 0.0

    regularization += abs(colorfulness(input_reflectance_cc) - colorfulness(ref_vis_rgb))
    total += 1.0
    regularization += abs(gamut_volume(input_reflectance_cc) - gamut_volume(ref_vis_rgb))
    total += 1.0
    regularization += abs(rms_contrast(input_reflectance_cc) - rms_contrast(ref_vis_rgb))
    total += 1.0
    regularization += abs(exposure(input_reflectance_cc) - exposure(ref_vis_rgb))
    total += 1.0

    blood_regularization = abs(torch.tensor(0.05)-torch.mean(skin_props[:, 1]))
    regularization += blood_regularization
    total += 1.0

    # TODO:
    # given enough skin types as anchoring points for the optimization, we can leverage such global statistics above,
    # but using the skin manifold statistics instead of the original reference image ones.

    regularization /= total
    loss = (1-k) * torch.mean(reconstruction_error) + k * regularization
    return loss, input_reflectance_cc, skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error


def color_correction(bio_skin, device, input_folder, output_folder, batch_size,
                     regularization, resizing_factor, exposure_correction):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    f = open(output_folder + 'ccm.txt', 'w')

    # init color correction matrix model
    ccm_model = ColorCorrectionModel()
    ccm_model = ccm_model.to(device)
    ccm_model.train()
    optimizer = optim.Adam(ccm_model.parameters(), lr=0.001)

    num_epochs = 1000

    input_images = load_calibration_images(input_folder,
                                           size_ratio=resizing_factor,
                                           exposure=exposure_correction,
                                           flip=False)
    calibration_image, total_height, total_width = concat_calibration_images(input_images, output_folder)

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        loss, input_reflectance_cc, skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error = \
            forward_pass_and_compute_loss(calibration_image, ccm_model, device, bio_skin, regularization)

        loss.backward()
        optimizer.step()
        print_step = 100
        if epoch % print_step == 0 or epoch == 0:
            print("====== MAE: " + str(loss.item()))

            if epoch % (2*print_step) == 0:
                io.save_tensor_to_image(output_folder + "/" + str(epoch), input_reflectance_cc, calibration_image.shape)
                bioskin.save_reconstruction(bio_skin.model_params, output_folder + "/", str(epoch), calibration_image,
                                    skin_props, ref_vis, ref_vis_rgb, ref_ir, ref_ir_avg, reconstruction_error,
                                    save_skin_props=False, save_spectrum=False)

            ccm_model.print_ccm()
            ccm_model.save_ccm(f)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}', f)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    f.close()


def apply_correction(input_folder, exposure_correction, device, ccm_matrix=None):
    if not ccm_matrix:
        ccm_matrix = torch.tensor([[0.7893, -0.0712, 0.0477],
                [-0.0084, 0.8850, 0.0289],
                [0.0054, -0.0918, 1.0041]], device=device)

    input_image = load_calibration_images(input_folder,
                                           size_ratio=1.0,
                                           exposure=exposure_correction,
                                           flip=False)[0]

    shape = input_image.shape
    input_image = io.vectorize_image(input_image, device=device)

    image_corrected = torch.matmul(input_image, ccm_matrix)

    io.save_tensor_to_image(args.output_folder + "/" + "corrected_test.jpeg", image_corrected, shape)


if __name__ == '__main__':
    args = parse_arguments()
    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")
    apply_correction(args.input_folder, args.exposure_correction, device)

