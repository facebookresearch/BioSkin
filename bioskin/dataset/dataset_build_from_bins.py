# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import bioskin.dataset.dataset_bin2pt as bpt
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Build Tensors from Binaries computed by Biophysical Skin Model')
    # modes
    parser.add_argument('--foldername',  nargs='?',
                        type=str, default='datasets/dataset_1000nm_2nm_1M_photons/',
                        help='Dataset Folder')
    parser.add_argument('--outfilename', nargs='?',
                        type=str, default='dataset_1000nm_2nm_1M_photons',
                        help='Out Tensors Name')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()
    # ## concat when distributed in independent tensors (random)
    # # bpt.bin_load_concat(foldername, outfilename, train_test_ratio)
    bpt.bin_load_add(args.foldername + 'train/', args.outfilename, 1.0, "train")
    bpt.bin_load_add(args.foldername + 'test/', args.outfilename, 1.0, "test")







