import torch
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import argparse
from skimage import color as skimage_color
from sklearn.manifold import Isomap
import bioskin.spectrum.color_spectrum as cs
from bioskin.spectrum.color_spectrum import linear_to_sRGB
from bioskin.spectrum.color_spectrum import sRGB_to_linear
import numpy as np
from bioskin.parameters.parameter import SKIN_PROPS
import bioskin.dataset.dataset_filter as df
import bioskin.dataset.dataset_bin2pt as b2pt
import bioskin.bioskin as bioskin
import bioskin.lighting.light_transport as lt
import matplotlib.pyplot as plt
import bioskin.parameters.params_io as params_io
import bioskin.utils as utils


def parse_arguments():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Sampling Manifold of skin Tones')

    parser.add_argument('--json_model',  nargs='?',
                        type=str, default='../pretrained_models/BioSkinAO',
                        help='Json file with trained nn model description and params')

    parser.add_argument('--input_folder',  nargs='?',
                        type=str, default='../input_images/',
                        help='Input folder with diffuse albedos to be adapted/homogeneized')

    parser.add_argument('--output_folder',  nargs='?',
                        type=str, default='../results/',
                        help='Output folder')

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
    args = parser.parse_args()
    return args


def manifold_2D(skin_tones, output_folder, model_name, num_samples):
    rgb_skin_tones = skin_tones.cpu().detach().numpy()
    skin_tones_lab = skimage_color.rgb2lab(linear_to_sRGB(rgb_skin_tones))

    # isomap test
    num_neighbors = 10
    # distance_matrix = distance.cdist(skin_tones_lab, skin_tones_lab, 'euclidean')
    distance_matrix = torch.cdist(torch.from_numpy(skin_tones_lab), torch.from_numpy(skin_tones_lab), p=2)

    embedding = Isomap(n_neighbors=num_neighbors, n_components=2)
    lab_colors_2d = embedding.fit_transform(distance_matrix.cpu().detach().numpy())

    skin_tones_rgb = skimage_color.lab2rgb(skin_tones_lab)
    plt.scatter(lab_colors_2d[:, 0], lab_colors_2d[:, 1], c=skin_tones_rgb.tolist())
    plt.show()
    plt.savefig(output_folder + model_name + 'rgb_colors_2d_' + str(num_neighbors) + 'neighbors_' +
                str(num_samples) + 'samples_distance_matrix.png')


def closest_points(A, B):
    # Compute the pairwise distance between the points in A and B using the L2 norm
    distances = torch.cdist(A, B, p=2)
    # Find the indexes of the closest points in A to the points in B
    closest_indexes = torch.argmin(distances, dim=0)
    return closest_indexes


def manifold_sample(json_model, bio_skin, output_folder, sampling_scheme, ranges, num_samples, color_space, device,
                    plotting=True, export_animation=False):

    output_folder = output_folder + 'manifold_visualization/' + \
                    os.path.basename(os.path.normpath(json_model)) + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_name, _ = os.path.splitext(os.path.basename(json_model))


    # uniform random sampling
    skin_tones_spectrum, skin_tones, skin_tones_ir_spectrum, skin_tones_ir, skin_props = \
        bio_skin.sample_skintones_rgb(num_samples, mode=sampling_scheme, ranges=ranges)

    skin_tones = torch.clamp(skin_tones, min=0, max=1)
    skin_tones = skin_tones[:, [2, 1, 0]]

    # manifold_2D(skin_tones, output_folder, model_name, num_samples)

    print('Total skin prop. combinations: ' + str(skin_props.shape[0]))


    # directory = '../SkintoneRamps/'
    # samples = load_dots(directory, num_points=10)
    # samples /= 255.0
    #
    # # sample subset and filter of uniform random
    # skin_tones_filtered = df.filter_subset_samples(skin_tones.cpu().detach().numpy(), samples, threshold=0.1)
    skin_tones_filtered = skin_tones.cpu().detach().numpy()

    skin_tones_lab = skimage_color.rgb2lab(linear_to_sRGB(skin_tones_filtered))

    uniform_samples_lab, hull = df.sample_uniform_in_color_space(skin_tones_lab,
                                                                 n_vertex_samples=200,
                                                                 n_surface_samples=200,
                                                                 n_volume_samples=100)

    uniform_samples_lab = df.select_representative_points(uniform_samples_lab, 50)
    print('Total representative points: ' + str(uniform_samples_lab.shape[0]))
    uniform_samples_lab = df.sort_by_luminance(uniform_samples_lab)

    uniform_samples_rgb = sRGB_to_linear(skimage_color.lab2rgb(uniform_samples_lab))
    uniform_samples_rgb_filtered = uniform_samples_rgb

    # discard the ones not in the point cloud
    uniform_samples_rgb_filtered = df.filter_points(uniform_samples_rgb_filtered,
                                                    skin_tones.cpu().detach().numpy(),
                                                    threshold=0.01)
    print('Total points (points out of original cloud removed): ' + str(uniform_samples_rgb_filtered.shape[0]))

    indexes = closest_points(skin_tones, torch.from_numpy(uniform_samples_rgb_filtered).to(device))
    skin_tones_uniform = uniform_samples_rgb_filtered
    skin_props_uniform = skin_props[indexes]
    skin_tones_spectrum_uniform = skin_tones_spectrum[indexes]
    skin_tones_ir_spectrum_uniform = skin_tones_ir_spectrum[indexes]
    skin_tones_ir_uniform = skin_tones_ir[indexes]

    # filtered subset
    b2pt.save_tensor_to_text(torch.from_numpy(uniform_samples_rgb_filtered).to(device).type(torch.float32),
                             output_folder + model_name + "_samples_uniformRGB255_filtered_linear.txt", format='%.6f')

    uniform_samples_sRGB = linear_to_sRGB(uniform_samples_rgb_filtered)
    b2pt.save_tensor_to_text(torch.from_numpy(uniform_samples_sRGB).to(device).type(torch.float32),
                             output_folder + model_name + "_samples_uniformRGB255_filtered_nonlinear_sRGB.txt",
                             format='%.6f')

    # saving exrs
    # b2pt.save_tensor_to_exr(torch.from_numpy(uniform_samples_rgb_filtered).to(device).type(torch.float32),
    #                         output_folder + model_name + '_samples_uniformRGB_filtered_linear.exr',
    #                         bgr=True)

    b2pt.save_tensor_to_exr(skin_tones[indexes],
                            output_folder + model_name + '_samples_uniformRGB_filtered_linear_closest.exr',
                            bgr=True)

    # reflectances_visible, skin_tones_reconstructed, reflectances_ir, reflectances_ir_avg = \
    #      bio_skin.skin_props_to_reflectance(params_io.unwarp_parameter_maps(skin_props[indexes]))
    #
    # skin_tones_reconstructed = skin_tones_reconstructed[:, [2, 1, 0]]
    # b2pt.save_tensor_to_exr(skin_tones_reconstructed,
    #                         output_folder + model_name +
    #                         '_samples_uniformRGB_filtered_linear_closest_reconstructed.exr',  bgr=True)

    for ii in range(0, skin_props_uniform.shape[1]-1):
        b2pt.save_tensor_to_exr(skin_props_uniform[:, ii],
                                output_folder + model_name + '_samples_uniformRGB_filtered_linear_p' + str(ii) + '.exr',
                                bgr=True)

    # full manifold
    b2pt.save_tensor_to_exr(skin_tones, output_folder + model_name + 'skin_tones.exr', bgr=True)

    # visualize points
    if plotting:
        utils.plotting.plot_convex_hull(hull, output_folder + model_name + "sampled_uniformLAB_convex_hull.png")
        utils.plotting.plot_3d_points('reflectance',
                                      linear_to_sRGB(uniform_samples_rgb),
                                      linear_to_sRGB(uniform_samples_rgb),
                                      0,
                                      title='Sampled Data',
                                      path=output_folder,
                                      name=model_name + "_sampled_uniform_subset_RGB",
                                      space=color_space)

        utils.plotting.plot_3d_points('reflectance',
                                      linear_to_sRGB(uniform_samples_rgb_filtered),
                                      linear_to_sRGB(uniform_samples_rgb_filtered),
                                      0,
                                      title='Sampled Data',
                                      path=output_folder,
                                      name=model_name + "_sampled_uniform_subset_filtered_RGB",
                                      space=color_space)

        utils.plotting.plot_3d_points('reflectance',
                                      linear_to_sRGB(skin_tones),
                                      linear_to_sRGB(skin_tones),
                                      0,
                                      title='Sampled Data',
                                      path=output_folder,
                                      name=model_name + "_" + sampling_scheme + "_" + ranges + "_sampled_full_RGB",
                                      space=color_space)

        utils.plotting.plot_3d_points('reflectance',
                                      linear_to_sRGB(skin_tones_filtered),
                                      linear_to_sRGB(skin_tones_filtered),
                                      0,
                                      title='Sampled Data',
                                      path=output_folder,
                                      name=model_name + "_sampled_full_filtered_RGB",
                                      space=color_space)

        if export_animation:
            # export 3d animation of points
            utils.plotting.plot_3d_points_animation(linear_to_sRGB(skin_tones),
                                                    linear_to_sRGB(skin_tones),
                                                    0.005,
                                                    'o',
                                                    1.0,
                                                    ',',
                                                    output_folder,
                                                    '_full.gif')

    return skin_props_uniform, skin_tones_spectrum_uniform, skin_tones_uniform, skin_tones_ir_spectrum_uniform, \
           skin_tones_ir_uniform


def manifold_sample_Lab_slices(json_model, bio_skin, output_folder, sampling_scheme, ranges, num_samples, color_space,
                               device):
    output_folder = output_folder + 'manifold_visualization/' + \
                    os.path.basename(os.path.normpath(json_model)) + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_name, _ = os.path.splitext(os.path.basename(json_model))


    # uniform random sampling
    skin_tones_spectrum, skin_tones, skin_tones_ir_spectrum, skin_tones_ir, skin_props = \
        bio_skin.sample_skintones_rgb(num_samples, mode=sampling_scheme, ranges=ranges)

    skin_tones = torch.clamp(skin_tones, min=0, max=1)
    skin_tones = skin_tones[:, [2, 1, 0]]

    skin_tones = skin_tones.cpu().detach().numpy()
    skin_tones_lab = skimage_color.rgb2lab(linear_to_sRGB(skin_tones))

    sorted_point_cloud = skin_tones_lab[skin_tones_lab[:, 0].argsort()]

    min_l = np.min(sorted_point_cloud[:, 0])
    max_l = np.max(sorted_point_cloud[:, 0])

    n_slices = 20
    threshold = 0.1
    slice_thickness = (max_l - min_l) / n_slices
    slices = []
    for i in range(n_slices):
        # Compute the lower and upper bounds of the current slice
        lower = min_l + i * slice_thickness
        upper = min_l + (i + 1) * slice_thickness
        slice_mask = (sorted_point_cloud[:, 0] >= lower) & (sorted_point_cloud[:, 0] < upper)
        # center = 0.5 * (lower + upper)
        # slice_mask = (sorted_point_cloud[:, 0] <= center + threshold) & (sorted_point_cloud[:, 0] >= center - threshold)
        slice = sorted_point_cloud[slice_mask]

        # Add the slice to the list
        slices.append(slice)

    fig_size = (5, 5)
    for i, slice in enumerate(slices):

        slice_sRGB = skimage_color.lab2rgb(slice)

        # Create a figure for this slice
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.scatter(slice[:, 1], slice[:, 2], c=slice_sRGB)
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_xlim([0, 35])
        ax.set_ylim([0, 50])
        ax.set_title(f'Slice {i + 1}')
        plt.savefig(output_folder + model_name + '_slice' + str(i) + '.png')
        plt.close(fig)

        #
        # indices = np.argsort(hue_angle)
        # slice = slice[indices]

        # isomap test
        num_neighbors = 10
        # distance_matrix = distance.cdist(skin_tones_lab, skin_tones_lab, 'euclidean')
        slice_p = torch.from_numpy(slice)
        slice_p[:, 0] = torch.mean(slice_p[:, 0])
        distance_matrix = torch.cdist(torch.from_numpy(slice), torch.from_numpy(slice), p=2)

        fig = plt.figure(figsize=fig_size)

        # embedding = Isomap(n_neighbors=num_neighbors, n_components=2)
        # lab_colors_2d = embedding.fit_transform(distance_matrix.cpu().detach().numpy())
        # skin_tones_rgb = skimage_color.lab2rgb(slice)
        # plt.scatter(lab_colors_2d[:, 0], lab_colors_2d[:, 1], c=skin_tones_rgb.tolist())

        embedding = Isomap(n_neighbors=num_neighbors, n_components=1)
        lab_colors_1d = embedding.fit_transform(distance_matrix.cpu().detach().numpy())
        skin_tones_rgb = skimage_color.lab2rgb(slice)
        plt.scatter(lab_colors_1d, lab_colors_1d, c=skin_tones_rgb.tolist())

        plt.savefig(output_folder + model_name + 'rgb_colors_2d_' + str(num_neighbors) + 'neighbors_' + str(num_samples)
                    + 'samples_distance_matrix_slice' + str(i) + '.png')
        plt.close(fig)

        # plot hue angle
        fig, ax = plt.subplots()
        hue_angle = cs.hue_angle(slice, format="Lab")
        ax.scatter(hue_angle, slice[:, 0], c=slice_sRGB)
        ax.set_xlim([-20, 90])
        ax.set_ylim([0, 100])
        ax.set_xlabel('Hue Angle')
        ax.set_ylabel('Luminance')
        ax.set_title('Luminance vs Hue Angle')
        plt.savefig(output_folder + model_name + '_slice' + str(i) + '_hue_angle.png', dpi=300)
        plt.close(fig)

    skin_tones_sorted = skimage_color.lab2rgb(sorted_point_cloud)
    utils.plotting.plot_3d_points('reflectance',
                                  skin_tones_sorted,
                                  skin_tones_sorted,
                                  0,
                                  title='Sampled Data',
                                  path=output_folder,
                                  name=model_name + "_sorted_skintones_RGB",
                                  space=color_space)

    skin_tones_sorted = sRGB_to_linear(skin_tones_sorted)
    indexes = closest_points(torch.from_numpy(skin_tones).to(device), torch.from_numpy(skin_tones_sorted).to(device))
    skin_tones = torch.from_numpy(skin_tones_sorted).to(device)
    skin_props = skin_props[indexes]
    skin_tones_spectrum = skin_tones_spectrum[indexes]
    skin_tones_ir_spectrum = skin_tones_ir_spectrum[indexes]
    skin_tones_ir = skin_tones_ir[indexes]

    return skin_props, skin_tones_spectrum, skin_tones, skin_tones_ir_spectrum, skin_tones_ir


def skin_properties_sample(json_model, bio_skin, output_folder, ranges, num_samples, color_space,
                           export_animation=False):

    # combinatorials of evenly sampling properties
    skin_tones_spectrum, skin_tones, skin_tones_ir_spectrum, skin_tones_ir, skin_props = \
        bio_skin.sample_skintones_by_props(ranges=ranges)
    skin_tones = torch.clamp(skin_tones, min=0, max=1)
    skin_tones = skin_tones[:, [2, 1, 0]]

    print('Total skin prop. combinations: ' + str(skin_props.shape[0]))

    output_folder = output_folder + 'manifold_visualization/' + \
                    os.path.basename(os.path.normpath(json_model)) + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_name, _ = os.path.splitext(os.path.basename(json_model))

    # saving
    b2pt.save_tensor_to_exr(skin_tones, output_folder + model_name + 'skin_tones_combinatorial_rgb.exr', bgr=True)
    b2pt.save_tensor_to_text(skin_tones * 255, output_folder + model_name + "skin_tones_combinatorial_255.txt")
    utils.plotting.plot_3d_points('reflectance',
                                  linear_to_sRGB(skin_tones),
                                  linear_to_sRGB(skin_tones),
                                  0,
                                  title='Sampled Data',
                                  path=output_folder,
                                  name=model_name + "_sampled_combinatorial_subset_RGB",
                                  space=color_space)

    if export_animation:
        # export 3d animation of points
        utils.plotting.plot_3d_points_animation(linear_to_sRGB(skin_tones),
                                                linear_to_sRGB(skin_tones),
                                                1.0,
                                                'o',
                                                0.0,
                                                ',',
                                                output_folder,
                                                'sampled_skin_tones.gif')

    # visualize corresponding skin properties given sampled points
    for skin_prop_index in range(0, bio_skin.model_params.D_skin):
        utils.plotting.plot_3d_points('skin_props',
                                      linear_to_sRGB(skin_tones),
                                      skin_props,
                                      skin_prop_index,
                                      title='Skin Prop. ' + SKIN_PROPS[skin_prop_index],
                                      path=output_folder,
                                      name=model_name + "_" + str(skin_props.shape[0]),
                                      space=color_space)

    return skin_props, skin_tones_spectrum, skin_tones, skin_tones_ir_spectrum, skin_tones_ir


def add_specular_reflection_spectral(json_model, skin_tones_spectrum, skin_tones, skin_tones_spectrum_IR, skin_tones_IR,
                                     skin_props, output_folder, device, num_hemispheric_samples=90, R0_Schick=0.04):

    output_folder = output_folder + 'manifold_visualization/' + \
                    os.path.basename(os.path.normpath(json_model)) + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_name, _ = os.path.splitext(os.path.basename(json_model))

    spectrums_with_specular, skin_tones_with_specular = \
        lt.add_specular_reflectance(skin_tones_spectrum, R0_Schick, num_hemispheric_samples, device)

    print("Saving spectrums...")
    b2pt.save_tensor_to_text(skin_props, output_folder + 'spectral_skin_props.csv')

    utils.plotting.plot_spectrums(skin_tones_spectrum, linear_to_sRGB(skin_tones),
                                  output_folder + 'spectral_reflectance_diffuse')

    b2pt.save_tensor_to_text(skin_tones_spectrum, output_folder + 'spectral_reflectance_diffuse.csv')

    utils.plotting.plot_spectrums(spectrums_with_specular, linear_to_sRGB(skin_tones_with_specular),
                                  output_folder + 'spectral_reflectance_diffuse_and_specular', bgr=True)

    b2pt.save_tensor_to_text(spectrums_with_specular,
                             output_folder + 'spectral_reflectance_diffuse_and_specular.csv')

    utils.plotting.plot_spectrums_IR(skin_tones_spectrum, linear_to_sRGB(skin_tones),
                                  skin_tones_spectrum_IR, linear_to_sRGB(skin_tones_IR),
                                  output_folder + 'spectral_reflectance_diffuse_visible_and_IR')


if __name__ == '__main__':

    args = parse_arguments()

    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device) + "(" + str(torch.cuda.device_count()) + ")")

    bio_skin = bioskin.BioSkinInference(args.json_model, device=device)

    if args.sampling_mode == "UniformColor":
        skin_props, skin_tones_spectrum, skin_tones, skin_tones_spectrum_IR, skin_tones_IR = \
            manifold_sample(args.json_model, bio_skin, args.output_folder, args.sampling_scheme, args.ranges,
                               args.num_samples, args.color_space, device,  plotting=True, export_animation=False)
    elif args.sampling_mode == "SkinProps":
        skin_props, skin_tones_spectrum, skin_tones, skin_tones_spectrum_IR, skin_tones_IR = \
            skin_properties_sample(args.json_model, bio_skin, args.output_folder, args.ranges, args.num_samples,
                                      args.color_space, plotting=True, export_animation=False)
    elif args.sampling_mode == "LabSlices":
        skin_props, skin_tones_spectrum, skin_tones, skin_tones_spectrum_IR, skin_tones_IR = \
            manifold_sample_Lab_slices(args.json_model, bio_skin, args.output_folder, args.sampling_scheme,
                                       args.ranges, args.num_samples, args.color_space, device)
    else:
        print("Choose sampling mode: 'UniformColor|SkinProps'")
        exit(0)

    add_specular_reflection_spectral(args.json_model, skin_tones_spectrum, skin_tones, skin_tones_spectrum_IR,
                                     skin_tones_IR, skin_props, args.output_folder, device, num_hemispheric_samples=90,
                                     R0_Schick=0.04)

