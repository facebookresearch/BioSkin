# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import io
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import argparse
import bioskin.dataset.dataset_bin2pt as b2pt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Dump Tensors to csv')

    parser.add_argument('--foldername',  nargs='?',
                        type=str, default='datasets/dataset_1000nm_2nm_1M_photons/',
                        help='Dataset Folder')
    parser.add_argument('--outfilename', nargs='?',
                        type=str, default='dataset_1000nm_2nm_1M_photons',
                        help='Out Tensors Name')
    parser.add_argument('--export_csv', nargs='?',
                        type=bool, default=True,
                        help='Export csv')
    parser.add_argument('--export_params', nargs='?',
                        type=bool, default=True,
                        help='Export parameters')
    parser.add_argument('--export_exr', nargs='?',
                        type=bool, default=True,
                        help='Export exr')
    parser.add_argument('--bgr', nargs='?',
                        type=bool, default=True,
                        help='Swap BGR-->RGB')
    parser.add_argument('--spectral', nargs='?',
                        type=bool, default=True,
                        help='Spectral or rgb')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()

    out = io.open(args.foldername + args.outfilename + ".csv", "w")
    outimage = args.foldername + args.outfilename + ".exr"

    mode = '_rgb'
    if args.spectral:
        mode = '_spec'
    albedo_train = torch.load(args.foldername + 'train/' + args.outfilename + mode + '_train.pt')
    albedo_test = torch.load(args.foldername + 'test/' + args.outfilename + mode + '_test.pt')

    b2pt.save_tensor_to_text(albedo_train, args.foldername + args.outfilename + mode + '_train.pt')
    b2pt.save_tensor_to_exr(albedo_train)

    b2pt.save_tensor_to_text(albedo_train)
    b2pt.save_tensor_to_exr(albedo_train)






