# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import train_exposure_aware
from argparse import Namespace
import dataset.dataset_augment_illuminants as dai

if __name__ == '__main__':

    sampling_p1 = [0.01, 0.05, 0.1]
    sampling_p2 = [1.0, 1.2, 1.5]
    num_samples = [14, 20, 30]
    datasets = []
    for num_sp in num_samples:
        for sp1 in sampling_p1:
            for sp2 in sampling_p2:
                args_dataset_augment = {'sampling_p1': sp1,
                                        'sampling_p2': sp2,
                                        'num_samples': num_sp,
                                        'size': 0,
                                        'folder': '../datasets/',
                                        'dataset': 'dataset_wide_thickness_wide_spectrum_380_1000nm_2nm_1M_photons',
                                        'sampling_type': 'uniform'
                                        }

                args_dataset_augment = Namespace(**args_dataset_augment)
                print(args_dataset_augment)
                datasets.append('dataset_wide_thickness_wide_spectrum_380_1000nm_2nm_1M_photons_uniform_'
                                + str(sp1) + '_'+str(sp2) + '_' + 'samples_' + str(num_sp))
                dai.run(args_dataset_augment)



    # train
    # datasets = ['dataset_wide_thickness_wide_spectrum_380_1000nm_2nm_1M_photons']
    loss_modes = ['visible_range']
    losses_color = ['sam', 'l1']
    losses_color_full_cycle = ['l1']
    batch_sizes = [256, 512, 1024, 16384, 8192]
    learning_rates = [0.0001]
    H_decs = [310, 512, 620]
    #losses weight
    w_p = [1.0, 2.0]
    w_a = [1.0]
    w_e = [0.0, 0.5, 1, 1.5, 2.0]
    w_full = [1]



    for dataset_filename in datasets:
        for loss_mode in loss_modes:
            for loss_color in losses_color:
                for loss_color_full_cycle in losses_color_full_cycle:
                    for batch_size in batch_sizes:
                        for learning_rate in learning_rates:
                            for H_dec in H_decs:
                                for i in range(0, len(w_p)):
                                    for j in range(0, len(w_e)):
                                        print("weights:[" + str(w_p[i]) + ", " + str(1.0) + ", " + str(w_e[j]) + ", " + str(1.0) + "]\n")
                                        args_rec = {'dataset_filename': dataset_filename,
                                                    'loss_mode': loss_mode,
                                                    'loss_color': loss_color,
                                                    'loss_color_full_cycle': loss_color_full_cycle,
                                                    'learning_rate': learning_rate,
                                                    'batch_size': batch_size,
                                                    'H_dec': H_dec,
                                                    'network_structure': 2,
                                                    'color_mode': 'spectral',
                                                    'param_mode': 'linear',
                                                    'loss_params': 'MSE',
                                                    'n_epochs': 25, #max(50, int(H_dec/10)),
                                                   'learning_rate_update': 5,
                                                    'writing_rate': 1,
                                                    'D_rgb': 3,
                                                    'H_enc': 70,
                                                    'w_p': w_p[i],
                                                    'w_a': 1.0,
                                                    'w_e': w_e[j],
                                                    'w_full': 1.0,
                                                    'dataset_folder': '../datasets/',
                                                    'output_model_folder': '../models/',
                                                    'output_model_name': ''
                                                    }
                                    args_rec = Namespace(**args_rec)
                                    print(args_rec)
                                    train_exposure_aware.run(args_rec)
