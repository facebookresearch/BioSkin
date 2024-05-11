# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib
from sklearn.utils.fixes import parse_version
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import torch
import json
import argparse
import bioskin.dataset.dataset_bin2pt as b2pt


# `normed` is being deprecated in favor of `density` in histograms
if parse_version(matplotlib.__version__) >= parse_version('2.1'):
    density_param = {'density': True}
else:
    density_param = {'normed': True}


def parse_arguments():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Filtering the ranges')
    # modes
    parser.add_argument('--folder',  nargs='?',
                        type=str, default='../datasets/',
                        help='Dataset Folder')
    parser.add_argument('--dataset', nargs='?',
                        type=str, default='dataset_wide_thickness_wide_spectrum_380_1000nm_2nm_1M_photons',
                        help='Dataset Name')
    parser.add_argument('--sampling_type', nargs='?',
                        type=str, default='uniform',
                        help='Distribution of samples: uniform | normal')
    parser.add_argument('--sampling_p1', nargs='?',
                        type=float, default=0.15,
                        help='Distribution parameter 1')
    parser.add_argument('--sampling_p2', nargs='?',
                        type=float, default=1.85,
                        help='Distribution parameter 2')
    parser.add_argument('--num_samples', nargs='?',
                        type=int, default=14,
                        help='Number of light intensity samples')
    parser.add_argument('--size', nargs='?',
                        type=int, default=0,
                        help='Size of the augmented dataset in times of original size. If < 1 then it will depend '
                             'on the num_samples only')


    args = parser.parse_args()
    return args


def stringtify(args):

    pstring = '_' + str(args.sampling_type) + '_' + str(args.sampling_p1)\
              + '_' + str(args.sampling_p2) + '_samples_' + str(args.num_samples)
    return pstring


def process_arguments(args):

    # input
    args.params_train_name = args.folder + args.dataset + '/train/tensors/' + args.dataset + '_params_train.pt'
    args.params_test_name = args.folder + args.dataset + '/test/tensors/' + args.dataset + '_params_test.pt'
    args.spectrums_train_name = args.folder + args.dataset + '/train/tensors/' + args.dataset + '_spec_train.pt'
    args.spectrums_test_name = args.folder + args.dataset + '/test/tensors/' + args.dataset + '_spec_test.pt'
    args.rgb_train_name = args.folder + args.dataset + '/train/tensors/' + args.dataset + '_rgb_train.pt'
    args.rgb_test_name = args.folder + args.dataset + '/test/tensors/' + args.dataset + '_rgb_test.pt'

    #output
    params_string = stringtify(args)
    out_dataset_name = args.dataset + params_string
    path_train = args.folder + out_dataset_name + '/train/tensors/'
    path_test = args.folder + out_dataset_name + '/test/tensors/'
    if not os.path.exists(path_train):
        os.makedirs(path_train)
    if not os.path.exists(path_test):
        os.makedirs(path_test)

    args.train_name_out = path_train + out_dataset_name
    args.test_name_out = path_test + out_dataset_name

    dataset_specs_json = args.folder + args.dataset + '/' + args.dataset + '.json'
    print('Loading Dataset Specs from ' + str(dataset_specs_json))
    with open(dataset_specs_json) as f:
        dataset_specs = json.load(f)
        print(dataset_specs)
        with open(args.folder + out_dataset_name + '/' + out_dataset_name + '.json', "w") as f_out:
            json.dump(dataset_specs, f_out)
            f_out.close()
        f.close()

    return args


def augment_with_illuminant_intensity(out_size, num_samples, sampling_type, sp1, sp2, spectrum, rgb, parameters):
    num_spectrums = spectrum.size()[0]

    for i in range(0, num_samples):
        print('Sample ' + str(i) + '/' + str(num_samples))

        # shuffle tensors
        indices = torch.randperm(spectrum.shape[0])
        spectrum_shuffle = spectrum[indices]
        rgb_shuffle = rgb[indices]
        parameters_shuffle = parameters[indices]

        # build sampling vectors
        if sampling_type == 'uniform':
            exposures = torch.FloatTensor(num_spectrums, 1).uniform_(sp1, sp2)
        elif sampling_type == 'normal':
            exposures = torch.FloatTensor(num_spectrums, 1).normal_(sp1, sp2)
            exposures = torch.clamp(exposures, min=sp1, max=sp2)
        else:
            print("ERROR: unknown sampling type")
            return 0

        # scale RGB tensor by the sampled exposures
        rgb_shuffle = torch.clone(rgb_shuffle)
        rgb_shuffle_scaled = rgb_shuffle * exposures

        # attach exposures as an extra column in parameters_and_exposures
        parameters_and_i_shuffle = torch.cat((parameters_shuffle, exposures), 1)

        # append to the augmented tensor
        if i == 0:
            spectrum_a = torch.clone(spectrum_shuffle)
            rgb_a = torch.clone(rgb_shuffle)
            rgb_a_scaled = torch.clone(rgb_shuffle_scaled)
            parameters_a = torch.clone(parameters_shuffle)
            parameters_and_i_a = torch.clone(parameters_and_i_shuffle)
            exposures_a = torch.clone(exposures)
        else:
            spectrum_a = torch.cat((spectrum_a, spectrum_shuffle), 0)
            rgb_a = torch.cat((rgb_a, rgb_shuffle), 0)
            rgb_a_scaled = torch.cat((rgb_a_scaled, rgb_shuffle_scaled), 0)
            parameters_a = torch.cat((parameters_a, parameters_shuffle), 0)
            parameters_and_i_a = torch.cat((parameters_and_i_a, parameters_and_i_shuffle), 0)
            exposures_a = torch.cat((exposures_a, exposures), 0)

    # out_size
    if out_size > 0:
        subset_size = num_spectrums*out_size
        subset_indices = torch.randperm(spectrum_a.shape[0])[:subset_size]
        spectrum_a = spectrum_a[subset_indices]
        rgb_a = rgb_a[subset_indices]
        rgb_a_scaled = rgb_a_scaled[subset_indices]
        parameters_a = parameters_a[subset_indices]
        parameters_and_i_a = parameters_and_i_a[subset_indices]
        exposures_a = exposures_a[subset_indices]

    return spectrum_a, rgb_a, parameters_a, rgb_a_scaled, exposures_a, parameters_and_i_a


def enhance_and_export(size, num_samples, sampling_type, sp1, sp2, spectrum, rgb, params, name_out, suffix, save_to_txt=False):

    spec_a, rgb_a, params_a, rgb_a_scaled, exposures, params_and_i_a = \
        augment_with_illuminant_intensity(size, num_samples, sampling_type, sp1, sp2, spectrum, rgb, params)

    print('Saving agumented torch tensors in '+ name_out + '_spec_' + suffix)
    torch.save(spec_a, name_out + '_spec_' + suffix + '.pt')
    torch.save(params_a, name_out + '_params_' + suffix + '.pt')
    torch.save(params_and_i_a, name_out + '_params_and_intensity_' + suffix + '.pt')
    torch.save(rgb_a, name_out + '_rgb_org_' + suffix + '.pt')
    torch.save(rgb_a_scaled, name_out + '_rgb_' + suffix + '.pt')
    torch.save(exposures, name_out + '_exp_' + suffix + '.pt')

    # save to exrs
    b2pt.save_tensor_to_exr(rgb_a, name_out + '_rgb_org_' + suffix + '.exr', bgr=True)
    b2pt.save_tensor_to_exr(rgb_a_scaled, name_out + '_rgb_' + suffix + '.exr', bgr=True)
    b2pt.save_tensor_to_exr(exposures, name_out + '_exp_' + suffix + '.exr', bgr=True)

    if save_to_txt:
        b2pt.save_tensor_to_text(spec_a, name_out + '_spec_' + suffix + '.txt')
        b2pt.save_tensor_to_text(spec_a, name_out + '_spec_' +  suffix + '.csv')
        b2pt.save_tensor_to_text(spec_a, name_out + '_spec_' + suffix + '.txt')
        b2pt.save_tensor_to_text(params_a, name_out + '_params_' + suffix + '.csv')
        b2pt.save_tensor_to_text(rgb_a, name_out + '_rgb_org_' + suffix + '.txt')
        b2pt.save_tensor_to_text(rgb_a, name_out + '_rgb_org_' + suffix + '.csv')
    return 0


def run(args):
    args = process_arguments(args)

    spectrums_train = torch.load(args.spectrums_train_name)
    rgb_train = torch.load(args.rgb_train_name)
    parameters_train = torch.load(args.params_train_name)

    spectrums_test = torch.load(args.spectrums_test_name)
    rgb_test = torch.load(args.rgb_test_name)
    parameters_test = torch.load(args.params_test_name)

    # train is 480k qMC random, test is 120k random uniform
    enhance_and_export(args.size, args.num_samples, args.sampling_type, args.sampling_p1, args.sampling_p2,
                        spectrums_train, rgb_train, parameters_train, args.train_name_out, 'train')
    enhance_and_export(args.size, int((120/420)*args.num_samples), args.sampling_type, args.sampling_p1, args.sampling_p2,
                       spectrums_test, rgb_test, parameters_test, args.test_name_out, 'test')


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
    print("Finished Augmenting Dataset!")