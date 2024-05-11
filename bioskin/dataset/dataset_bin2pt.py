# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import struct
import functools
import numpy as np
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import cv2


def bin2pt_test(filename):
    f = open(filename, "rb")

    ntuple = struct.unpack('f', f.read(4))
    n = int(functools.reduce(lambda sub, ele: sub * 10 + ele, ntuple))
    mtuple = struct.unpack('f', f.read(4))
    m = int(functools.reduce(lambda sub, ele: sub * 10 + ele, mtuple))

    values_test = struct.unpack('f' * n * m, f.read(4 * n * m))
    values_test = torch.tensor(values_test, dtype=torch.float32)
    values_test = torch.reshape(values_test, [n, m])
    torch.save(values_test, filename + '.test.pt')

    print('\nbin' + filename + 'converted to pt!\n')


def bin2pt(filename, train_test_ratio):
    f = open(filename, "rb")

    ntuple = struct.unpack('f', f.read(4))
    n = int(functools.reduce(lambda sub, ele: sub * 10 + ele, ntuple))
    mtuple = struct.unpack('f', f.read(4))
    m = int(functools.reduce(lambda sub, ele: sub * 10 + ele, mtuple))

    n_train = int(train_test_ratio * n)
    values_train = struct.unpack('f' * n_train * m, f.read(4 * n_train * m))
    tensor_train = torch.tensor(values_train, dtype=torch.float32)
    tensor_train = torch.reshape(tensor_train, [n_train, m])
    torch.save(tensor_train, filename + '.train.pt')

    n_test = int((1 - train_test_ratio) * n)
    values_test = struct.unpack('f' * n_test * m, f.read(4 * n_test * m))
    values_test = torch.tensor(values_test, dtype=torch.float32)
    values_test = torch.reshape(values_test, [n_test, m])
    torch.save(values_test, filename + '.test.pt')

    print('\nbin' + filename + 'converted to pt!\n')


def swap_to_bgr(tuple_rgb, nelements):
    vector = np.array(tuple_rgb)
    count = 1
    for i in range(0, nelements):
        if count == 3:
            count = 0
            aux = vector[i]
            vector[i] = vector[i - 2]
            vector[i - 2] = aux
        count = count+1
    bgr_tuple = tuple(vector)
    return bgr_tuple


def load_bin(filename, swap_color):
    f = open(filename, "rb")

    ntuple = struct.unpack('f', f.read(4))
    n = int(functools.reduce(lambda sub, ele: sub * 10 + ele, ntuple))
    mtuple = struct.unpack('f', f.read(4))
    m = int(functools.reduce(lambda sub, ele: sub * 10 + ele, mtuple))

    values = struct.unpack('f' * n * m, f.read(4 * n * m))
    if swap_color:
        values = swap_to_bgr(values, n*m)
    tensor = torch.tensor(values, dtype=torch.float32)
    tensor = torch.reshape(tensor, [n, m])

    f.close
    return tensor


def load_txt(filename):
    f = open(filename, "r")
    n = int(float(f.readline()))
    m = int(float(f.readline())) - 1
    values = np.zeros((n*m))
    print(values.shape)
    index = 0
    while (True):
        # read next line
        line = f.readline()
        # if line is empty, you are done with all lines in the file
        if not line:
            break
        # you can access the line
        numbers = line.split(',')
        del numbers[-1]
        for number in numbers:
            values[index] = float(number)
            index += 1

    f.close
    tensor = torch.tensor(values, dtype=torch.float32)
    tensor = torch.reshape(tensor, [n, m])
    return tensor


def load_header(filename):
    f = open(filename, "rb")

    n_tuple = struct.unpack('f', f.read(4))
    n = int(functools.reduce(lambda sub, ele: sub * 10 + ele, n_tuple))

    nrgb_tuple = struct.unpack('f', f.read(4))
    nrgb = int(functools.reduce(lambda sub, ele: sub * 10 + ele, nrgb_tuple))

    nspectral_tuple = struct.unpack('f', f.read(4))
    nspectral = int(functools.reduce(lambda sub, ele: sub * 10 + ele, nspectral_tuple))

    nparams_tuple = struct.unpack('f', f.read(4))
    nparams = int(functools.reduce(lambda sub, ele: sub * 10 + ele, nparams_tuple))

    return n, nrgb, nspectral, nparams


def save_tensor_to_text(tensor, filename, format=None):
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().detach().numpy()
    if format:
        np.savetxt(filename, tensor, fmt=format, delimiter=",")
    else:
        np.savetxt(filename, tensor, delimiter=",")


def save_tensor_to_exr(tensor, filename, bgr=True):
    size = int(np.ceil(np.sqrt(tensor.shape[0])))
    img = np.zeros([size, size, 3], dtype=np.float32)
    if len(tensor.shape) == 2:
        channels = int(tensor.shape[1])
    elif len(tensor.shape) == 1:
        channels = 1

    for i in range(0, size):
        for j in range(0, size):
            ii = (i * size + j)
            if ii < tensor.shape[0]:
                if channels == 3:
                    r = float(tensor[ii][0].item())
                    g = float(tensor[ii][1].item())
                    b = float(tensor[ii][2].item())
                    if bgr:
                        img[i][j][2] = r
                        img[i][j][1] = g
                        img[i][j][0] = b
                    else:
                        img[i][j][0] = r
                        img[i][j][1] = g
                        img[i][j][2] = b
                elif channels == 2:
                    img[i][j][0] = float(tensor[ii][0].item())
                    img[i][j][1] = float(tensor[ii][1].item())
                    img[i][j][2] = 0.0
                elif channels == 1:
                    img[i][j][0] = float(tensor[ii].item())
                    img[i][j][1] = float(tensor[ii].item())
                    img[i][j][2] = float(tensor[ii].item())

    coloring = img.astype("float32")
    cv2.imwrite(filename, coloring)


def save_tensor_to_jpeg(tensor, filename, bgr=True, apply_gamma=True):
    size = int(np.ceil(np.sqrt(tensor.shape[0])))
    img = np.zeros([size, size, 3], dtype=np.float32)
    channels = int(tensor.shape[1])

    for i in range(0, size):
        for j in range(0, size):
            ii = (i * size + j)
            if ii < tensor.shape[0]:
                if channels == 3:
                    r = float(tensor[ii][0].item())
                    g = float(tensor[ii][1].item())
                    b = float(tensor[ii][2].item())
                    if bgr:
                        img[i][j][2] = r
                        img[i][j][1] = g
                        img[i][j][0] = b
                    else:
                        img[i][j][0] = r
                        img[i][j][1] = g
                        img[i][j][2] = b
                elif channels == 2:
                    img[i][j][0] = float(tensor[ii][0].item())
                    img[i][j][1] = float(tensor[ii][1].item())
                    img[i][j][2] = 0.0
                elif channels == 1:
                    img[i][j][0] = float(tensor[ii].item())
                    img[i][j][1] = float(tensor[ii].item())
                    img[i][j][2] = float(tensor[ii].item())
    if apply_gamma:
        img = img ** (1/2.2)
    img = img * 255
    cv2.imwrite(filename, img)


def save_tensors(filename, tensor, train_test_ratio, type, n):
    n_train = int(train_test_ratio * n)
    n_test = n - n_train

    if (train_test_ratio == 1.0):
        torch.save(tensor, filename + '_' + type + '.pt')
        save_tensor_to_text(tensor, filename + '_' + type + '.txt')

    else:
        tensor_train, tensor_test = torch.split(tensor, (n_train, n_test))
        torch.save(tensor_train, filename + '_train.pt')
        torch.save(tensor_test, filename + '_test.pt')
        save_tensor_to_text(tensor_train, filename + '_train.csv')
        save_tensor_to_text(tensor_test, filename + '_test.csv')


def bin_load_concat(foldername, outfilename, train_test_ratio):
    file_header = ""
    for root, dirs, files in os.walk(foldername):
        for filename in files:
            if "HEADER" in filename:
                file_header = filename
    if file_header != "":
        n, nrgb, nspectral, nparams = load_header(foldername + file_header)
    else:
        n = 120000
        nrgb = 3
        nspectral = 41
        nparams = 5
    nbatches = 0
    for root, dirs, files in os.walk(foldername):
        for filename in files:
            if "RGB.bin" in filename:
                nbatches += 1


    tensor_rgb = torch.empty(0, nrgb)
    tensor_spectral = torch.empty(0, nspectral)
    tensor_parameters = torch.empty(0, nparams)

    for root, dirs, files in os.walk(foldername):
        for filename in files:
            if "RGB.bin" in filename:
                tensor = load_bin(foldername + filename, True)
                tensor_rgb = torch.cat((tensor_rgb, tensor), 0)
            elif "spectral.bin" in filename:
                tensor = load_bin(foldername + filename, False)
                tensor_spectral = torch.cat((tensor_spectral, tensor), 0)
            elif "parameters.bin" in filename:
                tensor = load_bin(foldername + filename, False)
                tensor_parameters = torch.cat((tensor_parameters, tensor), 0)

    save_tensors(foldername + outfilename + "_spec", tensor_spectral, train_test_ratio,  type, n)
    save_tensors(foldername + outfilename + "_params", tensor_parameters, train_test_ratio,  type, n)
    save_tensors(foldername + outfilename + "_rgb", tensor_rgb, train_test_ratio, type, n)
    save_tensor_to_exr(tensor_rgb, foldername + outfilename + type + "_rgb.exr", True)

    return tensor_rgb, tensor_spectral, tensor_parameters


def bin_load_add(foldername, outfilename, train_test_ratio, type):
    print("Loading tensors...\n")
    file_header = ""
    for root, dirs, files in os.walk(foldername):
        for filename in files:
            if "HEADER.bin" in filename:
                file_header = filename
    if file_header != "":
        n, nrgb, nspectral, nparams = load_header(foldername + file_header)
    else:
        n = 120000
        nrgb = 3
        nspectral = 41
        nparams = 5
        print('WARNING: header not found, using default sizes')

    nbatches = 0
    for root, dirs, files in os.walk(foldername):
        for filename in files:
            if "RGB.bin" in filename:
                nbatches += 1

    tensor_rgb = torch.empty(n, nrgb)
    tensor_spectral = torch.empty(n, nspectral-1)
    tensor_parameters = torch.empty(n, nparams)

    for root, dirs, files in os.walk(foldername):
        for filename in files:
            if "RGB.bin" in filename:
                tensor = load_bin(foldername + filename, True)
                tensor_rgb = torch.add(tensor_rgb, tensor)
            # elif "spectral.bin" in filename:
            #     tensor = load_bin(foldername + filename, False)
            #     tensor_spectral = torch.add(tensor_spectral, tensor)
            elif "spectral.txt" in filename:
                tensor = load_txt(foldername + filename)
                tensor_spectral = torch.add(tensor_spectral, tensor)
            elif "parameters.bin" in filename:
                tensor = load_bin(foldername + filename, False)
                tensor_parameters = torch.add(tensor_parameters, tensor)

    print("Saving tensors...\n")
    save_tensors(foldername + outfilename + "_spec", tensor_spectral, train_test_ratio, type, n)
    save_tensors(foldername + outfilename + "_params", tensor_parameters, train_test_ratio, type, n)
    save_tensors(foldername + outfilename + "_rgb", tensor_rgb, train_test_ratio, type, n)
    save_tensor_to_exr(tensor_rgb, foldername + outfilename + type + "_rgb.exr", True)

    # return tensor_rgb, tensor_spectral, tensor_parameters
    return tensor_spectral
