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
from scipy.signal import savgol_filter


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


def smooth_spectrum_tensor(spectrum_tensor, plot=False):
    size = spectrum_tensor.size()
    num_spectrum = size[0]
    size_spectrum = size[1]
    for i in range(0, num_spectrum):
        if i % 100 == 0:
            progress = 100.0 * i / num_spectrum
            sys.stdout.write('Smoothing...' + str("%.2f" % progress) + '%\r')
        x = np.linspace(0, 1, size_spectrum)
        y = spectrum_tensor[i, :]
        y = savgol_filter(y, 15, 3)  # window size 15, polynomial order 3
        y = savgol_filter(y, 15, 3)
        spectrum_tensor[i] = torch.Tensor(y)

        if plot:
            plt.plot(x, spectrum_tensor[i, :])
            plt.plot(x, y, color='red')
            plt.ylim(0, 0.8)
            plt.show()
            print('checking')
    return spectrum_tensor


if __name__ == '__main__':

    args = parse_arguments()

    spectrums_train = torch.load(args.folder + 'train/' + args.dataset + '_spec_train.pt')
    spectrums_test = torch.load(args.folder + 'test/' + args.dataset + '_spec_test.pt')

    spectrums_train_smooth = smooth_spectrum_tensor(spectrums_train, plot=False)
    spectrums_test_smooth = smooth_spectrum_tensor(spectrums_test)

    torch.save(spectrums_train_smooth, args.folder + 'train/' + args.dataset + '_smooth_spec_train.pt')
    torch.save(spectrums_test_smooth, args.folder + 'test/' + args.dataset + '_smooth_spec_test.pt')

    print("Finished Smoothing Spectral Tensors!")