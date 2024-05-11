# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from bioskin.utils.io import create_gif_from_images


def parse_arguments():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optimizing biophysical property maps')

    parser.add_argument('--folder_path',  nargs='?',
                        type=str, default='../reconstruction_optimized_raw/',
                        help='Folder with frames')

    parser.add_argument('--gif_name',  nargs='?',
                        type=str, default='Reconstruction.gif',
                        help='GIF name result')

    parser.add_argument('--image_extension',  nargs='?',
                        type=str, default='_vis.jpeg',
                        help='Extension of images to compose the GIF')

    parser.add_argument('--duration', nargs='?',
                        type=float, default=0.5,
                        help='Duration of each frame')

    parser.add_argument('--max_frames', nargs='?',
                        type=int, default=50,
                        help='Number of frames')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    create_gif_from_images(args.folder_path, args.gif_name, args.image_extension,
                           duration=args.duration, max_frames=args.max_frames)
