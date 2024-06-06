# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import cv2
import torch
import numpy as np
import imageio
import re
import bioskin.spectrum.color_spectrum as color


def get_file_list(input_folder):
    valid_extensions = ['.exr', '.jpeg', '.jpg', '.png']
    filename_list = os.listdir(input_folder)
    # remove masks from the list (strings containing "_mask")
    filename_list = [filename for filename in filename_list if "_mask" not in filename]
    filename_list = [filename for filename in filename_list if "_spec" not in filename]

    filename_list_no_ext = []
    extensions = []
    # get names and extensions
    for i in range(0, len(filename_list)):
        extension = os.path.splitext(filename_list[i])[1]
        if extension in valid_extensions:
            filename_list_no_ext.append(os.path.splitext(filename_list[i])[0])
            extensions.append(extension)

    if '.DS_Store' in filename_list_no_ext:  # This is for Mac OS users
        index_to_delete = filename_list_no_ext.index('.DS_Store')
        if index_to_delete != -1:
            del extensions[index_to_delete]
        filename_list_no_ext.remove('.DS_Store')
    return filename_list_no_ext, extensions


def vectorize_image(input_image, device, monochrome=False):
    row, col, channel = input_image.shape
    # reshape to nx3
    input_image_vec = input_image.reshape((row * col, 3))
    if monochrome == 1:
        input_image_vec = input_image_vec[:, 0]
    input_image_vec = np.clip(input_image_vec, a_min=0.0, a_max=1.0)  # clamping to 0 1
    tensor = torch.from_numpy(input_image_vec.astype("float32"))
    tensor = tensor.to(device=device)
    return tensor


def replace_nan_with_valid_neighbor(arr):
    valid_neighbors = arr[~np.isnan(arr)]
    if valid_neighbors.size > 0:
        return valid_neighbors[0]
    else:
        return np.nan


def detect_and_fix_nans(image, path_to_image=""):
    nan_indices = np.argwhere(np.isnan(image))
    for index in nan_indices:
        print("Warning: fixing NaNs found in loaded image " + path_to_image)
        i, j, k = index
        iminus = max(0, i - 1)
        iplus = min(image.shape[0] - 1, i + 1)
        jminus = max(0, j - 1)
        jplus = min(image.shape[1] - 1, j + 1)
        neighbors = [(iminus, j), (iplus, j), (i, jminus), (i, jplus)]
        for neighbor_i, neighbor_j in neighbors:
            if not np.isnan(image[neighbor_i, neighbor_j, :].any()):
                image[i, j, :] = image[neighbor_i, neighbor_j, :]
                break
    return image


def load_image(path_to_image, max_width=0, verbose=True):
    try:
        image = cv2.imread(path_to_image, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if image is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"File {path_to_image} not found.")
        return None
    extension = os.path.splitext(path_to_image)[1]

    if verbose:
        print(path_to_image)
        print("Image size: ", image.shape)

    # If png, apply gamma and normalize
    if extension == ".png" or extension == ".jpg" or extension == ".jpeg":
        image = image.astype(np.float32) / 255.0
        image = color.sRGB_to_linear(image)
        image = np.clip(image, 0, 1)

    image = detect_and_fix_nans(image, path_to_image)

    if image.shape[1] > max_width != 0:
        ratio = max_width / image.shape[1]
        new_width = max_width
        new_height = int(image.shape[0] * ratio)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        if verbose:
            print("Downsampling image " + path_to_image + " to " + str(max_width))
    return image


def save_image(image, path_to_save, max_width=0, verbose=True):
    try:
        if image is None:
            raise ValueError("Image is None.")
    except ValueError as e:
        print(e)
        return None
    extension = os.path.splitext(path_to_save)[1]
    # If png, apply gamma and normalize
    if extension == ".png" or extension == ".jpg" or extension == ".jpeg":
        image = np.clip(image, 0, 1)
        image = color.linear_to_sRGB(image)
        image = (image * 255.0).astype(np.uint8)
    if image.shape[1] > max_width != 0:
        ratio = max_width / image.shape[1]
        new_width = max_width
        new_height = int(image.shape[0] * ratio)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        if verbose:
            print("Downsampling image to " + str(max_width))
    # Save the image
    cv2.imwrite(path_to_save, image)
    if verbose:
        print(f"Image saved at {path_to_save}")


def vector1D_to_image(vector, row, col, channels=3):
    return vector.reshape((row, col, channels)).astype("float32")


def save_jpeg(path, image, linear_input=True):
    if linear_input:
        image = color.linear_to_sRGB(image)
    cv2.imwrite(path + '.jpeg', image * 255)


def tensor_to_image(t, shape, channels=3):
    t_image = vector1D_to_image(t.cpu().detach().numpy(), shape[0], shape[1], channels=channels)
    return t_image


def cpu_tensor_to_image(t, shape, channels=3):
    t_image = vector1D_to_image(t.numpy(), shape[0], shape[1], channels=channels)
    return t_image


def save_tensor_to_image(path, t, shape, channels=3, cpu=False):
    if cpu:
        t_image = cpu_tensor_to_image(t, shape, channels)
    else:
        t_image = tensor_to_image(t, shape, channels)
    cv2.imwrite(path + ".exr", t_image)
    save_jpeg(path, t_image)


def extract_number(filename):
    return int(re.search(r'(\d+)', filename).group(1))


def create_gif_from_images(folder_path, gif_name, image_extension="jpg", duration=0.1, max_frames=None):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(image_extension)]

    # Sort files based on the numerical value embedded in the filename
    sorted_files = sorted(image_files, key=extract_number)

    # Limit the number of frames if max_frames is specified
    if max_frames is not None:
        sorted_files = sorted_files[:max_frames]

    images = [imageio.imread(os.path.join(folder_path, f)) for f in sorted_files]

    # Create GIF with controlled speed (frame duration)
    imageio.mimsave(os.path.join(folder_path, gif_name), images, duration=duration, loop=0)



