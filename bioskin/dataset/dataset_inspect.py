# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import matplotlib
import matplotlib.pylab as plt
from sklearn.utils.fixes import parse_version
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import torch
import sys
import argparse


if parse_version(matplotlib.__version__) >= parse_version('2.1'):
    density_param = {'density': True}
else:
    density_param = {'normed': True}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Smoothing Spectral Reflectance Curves using KDE')
    # modes
    parser.add_argument('--folder',  nargs='?',
                        type=str, default='datasets/dataset_1000nm_2nm_1M_photons/',
                        help='Dataset Folder')
    parser.add_argument('--dataset', nargs='?',
                        type=str, default='dataset_wide_thickness_1000nm_2nm_10M_photons',
                        help='Dataset Name')

    args = parser.parse_args()
    return args


def plot_spectrum_tensor(spectrum_tensor):
    size = spectrum_tensor.size()
    num_spectrum = size[0]
    size_spectrum = size[1]
    for i in range(0, num_spectrum):
        if i % 100 == 0:
            progress = 100.0 * i / num_spectrum
            sys.stdout.write('...' + str("%.2f" % progress) + '%\r')
            x = np.linspace(0, 1, size_spectrum)
            plt.plot(x, spectrum_tensor[i, :])
            plt.ylim(0, 0.8)
            plt.show()


if __name__ == '__main__':

    args = parse_arguments()

    train_filename = args.folder + 'train/' + args.dataset + '_spec_train'
    test_filename = args.folder + 'test/' + args.dataset + '_spec_test'

    spectrums_train = torch.load(train_filename + '.pt')
    spectrums_test = torch.load(test_filename + '.pt')

    spectrums_train_smooth = plot_spectrum_tensor(spectrums_train)
    spectrums_test_smooth = plot_spectrum_tensor(spectrums_test)

    print("Finished Smoothing Spectral Tensors!")