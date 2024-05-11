# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import cv2
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from bioskin.bioskin import BioSkinInference
from bioskin.bioskin import save_reconstruction
import bioskin.utils.io as io
import os
import torch.nn.functional as F


def parse_arguments():
    parser = argparse.ArgumentParser(description='Optimizing biophysical property maps')

    parser.add_argument('--json_model',  nargs='?',
                        type=str, default='../pretrained_models/BioSkinAO',
                        help='Json file with trained nn model description and params')

    parser.add_argument('--input_folder',  nargs='?',
                        type=str, default='../input_images/',
                        help='Input Folder with diffuse reflectance to infer its properties')

    parser.add_argument('--batch_size',  nargs='?',
                        type=int, default=256,
                        help='Batch size to eval the network')

    parser.add_argument('--regularization',  nargs='?',
                        type=int, default=0.5,
                        help='Regularization weight')

    parser.add_argument('--output_folder',  nargs='?',
                        type=str, default='../reconstruction_optimized/',
                        help='Output folder with estimated skin properties and reconstructed albedos')

    parser.add_argument('--max_width', nargs='?',
                        type=int, default=800,
                        help='Maximum image width. ')

    parser.add_argument('--num_epochs', nargs='?',
                        type=int, default=500,
                        help='Number of optimization epochs')

    parser.add_argument('--output_ratio', nargs='?',
                        type=int, default=20,
                        help='Number of epochs to save results')


    args = parser.parse_args()

    return args


class BioProps(nn.Module):
    def __init__(self, bio_skin_model, tensor_size, initial_values):
        super(BioProps, self).__init__()
        self.bio_skin_model = bio_skin_model
        self.num_properties = self.bio_skin_model.latent_size
        self.shape_reflectance = (tensor_size[0], tensor_size[1], 3)
        self.shape_props = (tensor_size[0], tensor_size[1], self.num_properties)

        initial_values = initial_values[:self.num_properties]

        # Create an initial tensor from the list of initial values
        initial_tensor = torch.tensor(initial_values, dtype=torch.float32)

        # Repeat the initial tensor to match the desired shape
        repeated_tensor = initial_tensor.repeat(tensor_size[0] * tensor_size[1], 1)

        self.skin_props = nn.Parameter(repeated_tensor, requires_grad=True)

    def initialize_skin_props(self, input_reflectance):
        skin_props = self.bio_skin_model.reflectance_to_skin_props(input_reflectance)
        self.skin_props.data.copy_(skin_props)

    def forward(self):
        self.ref_vis, self.ref_vis_rgb, self.ref_ir, self.ref_ir_avg = \
            self.bio_skin_model.skin_props_to_reflectance(self.skin_props)
        skin_property_maps = self.skin_props.reshape(self.shape_props)
        reflectance_map = self.ref_vis_rgb.reshape(self.shape_reflectance)

        return reflectance_map, skin_property_maps


def compute_gradients(map_slice):
    diff_y = map_slice[1:, :] - map_slice[:-1, :]
    diff_x = map_slice[:, 1:] - map_slice[:, :-1]
    min_dim_y = min(diff_y.shape[0], diff_x.shape[0])
    min_dim_x = min(diff_y.shape[1], diff_x.shape[1])
    diff_y = diff_y[:min_dim_y, :min_dim_x]
    diff_x = diff_x[:min_dim_y, :min_dim_x]
    grads = torch.pow(diff_y, 2) + torch.pow(diff_x, 2)
    return grads


def compute_loss(reflectance_input, reflectance_predicted, skin_properties, regularization_weight=0.1):
    loss = torch.mean(abs(reflectance_input - reflectance_predicted))
    # regularization term: smoothness of maps
    for i in range(0, 5):
        property_map = skin_properties[:, :, i].clone()
        map_grads = torch.sum(torch.abs(compute_gradients(property_map))) / property_map.shape[0]
        loss += regularization_weight * 0.1 * 0.2 * map_grads

    if skin_properties.shape[2] > 5:
        occlusion_index = skin_properties.shape[2]-1
        occlusion_map = skin_properties[:, :, occlusion_index].clone()
        occlusion_grads = torch.sum(torch.abs(compute_gradients(occlusion_map))) / occlusion_map.shape[0]
        loss += regularization_weight * 0.9 * occlusion_grads
    return loss


def optimize_bio_maps(bio_skin_model, device, input_folder, output_folder, regularization, num_epochs=100,
                      max_width=400, output_ratio=1):
    filename_list_no_ext, extensions = io.get_file_list(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for index, filename_no_ext in enumerate(filename_list_no_ext):
        print('-- Loading ' + filename_no_ext + ', ' + str(index) + ' of ' + str(len(filename_list_no_ext)))
        path_to_image = os.path.join(input_folder, filename_no_ext + extensions[index])
        input_image = io.load_image(path_to_image, max_width)
        input_reflectance_2D = torch.from_numpy(input_image.astype("float32")).to(device=device)
        input_reflectance_1D = io.vectorize_image(input_image, device=device)

        initial_values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        bio_props = BioProps(bio_skin_model, (input_image.shape[0], input_image.shape[1]), initial_values=initial_values)
        bio_props = bio_props.to(device)
        bio_props.train()
        optimizer = optim.Adam(bio_props.parameters(), lr=0.1)

        num_epochs = num_epochs
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            reflectance_map, skin_property_map = bio_props.forward()
            loss = compute_loss(input_reflectance_2D, reflectance_map, skin_property_map, regularization)
            reconstruction_error = abs(bio_props.ref_vis_rgb - input_reflectance_1D)
            loss.backward()
            optimizer.step()
            sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            sys.stdout.flush()

            if epoch % output_ratio == 0:
                save_reconstruction(bio_props.bio_skin_model.model_params,
                                output_folder + filename_no_ext + "_" + str(epoch), "", input_image,
                                bio_props.skin_props,  bio_props.ref_vis,  bio_props.ref_vis_rgb,  bio_props.ref_ir,
                                bio_props.ref_ir_avg, reconstruction_error, save_spectrum=False)

        save_reconstruction(bio_props.bio_skin_model.model_params,
                            output_folder + filename_no_ext + "_" + str(num_epochs), "", input_image,
                            bio_props.skin_props, bio_props.ref_vis, bio_props.ref_vis_rgb, bio_props.ref_ir,
                            bio_props.ref_ir_avg, reconstruction_error, save_spectrum=False)


if __name__ == '__main__':
    args = parse_arguments()

    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")

    bio_skin_model = BioSkinInference(args.json_model, device=device, batch_size=args.batch_size)
    optimize_bio_maps(bio_skin_model, device, args.input_folder, args.output_folder, args.regularization,
                      args.num_epochs, args.max_width,  args.output_ratio)
