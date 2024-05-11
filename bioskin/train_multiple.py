# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import train
from argparse import Namespace


if __name__ == '__main__':

    datasets = ['dataset_wide_thickness_wide_spectrum_380_1000nm_2nm_1M_photons',
                'dataset_wide_thickness_wide_spectrum_380_1000nm_2nm_1M_photons_m[0.001-0.430]_b[0.001-0.200]_t[0.001-0.035]_r[0.700-0.780]_o[0.900-1.000]']
    loss_modes = ['visible_range', 'full_spectrum']
    losses_color = ['sam', 'l1']
    losses_color_full_cycle = ['l1', 'sam']
    batch_sizes = [256, 512, 1024, 16384, 8192]
    learning_rates = [0.0001]
    H_decs = [256, 310, 400, 512, 620, 1024, 2048]
    #losses weight
    w_p =       [1,     1,     1,      1,      1,      1,      1,		1]
    w_a =       [0,     2,     2,      1.5,    1,      0,      0,		1]
    w_full =    [1,     1.5,   2,      2,      2,      2,    1.5,		1]

    w_p =       [1,     1,     1,      1,      1,      1,       1,      1,		1,      1]
    w_a =       [0,     0,     0,      1,      1,      1,       1.5,    1.5,    2,      2]
    w_full =    [1,     1.5,   2,      1,      1.5,    2,       1.5,	2,      1.5,    2]


    for dataset_filename in datasets:
        for loss_mode in loss_modes:
            for loss_color in losses_color:
                for loss_color_full_cycle in losses_color_full_cycle:
                    for batch_size in batch_sizes:
                        for learning_rate in learning_rates:
                            for H_dec in H_decs:
                                for i in range(0, len(w_p)):
                                    ii = len(w_p) - i - 1
                                    print("weights:[" + str(w_p[ii]) + ", " + str(w_a[ii]) + ", " + str(w_full[ii]) + "]\n")
                                    args_rec = {'dataset_filename': dataset_filename,
                                                'loss_mode': loss_mode,
                                                'loss_color': loss_color,
                                                'loss_color_full_cycle': loss_color_full_cycle,
                                                'learning_rate': learning_rate,
                                                'batch_size': batch_size,
                                                'H_dec': H_dec,
                                                'color_mode': 'spectral',
                                                'param_mode': 'linear',
                                                'loss_params': 'MSE',
                                                'n_epochs': max(50, int(H_dec/10)),
                                                'writing_rate': 1,
                                                'D_rgb': 3,
                                                'H_enc': 70,
                                                'w_p': w_p[ii],
                                                'w_a': w_a[ii],
                                                'w_full': w_full[ii],
                                                'dataset_folder': '../datasets/',
                                                'output_model_folder': '../models/',
                                                'output_model_name': '',
                                                'learning_rate_update': 10
                                                }
                                    args_rec = Namespace(**args_rec)
                                    print(args_rec)
                                    train.run(args_rec)