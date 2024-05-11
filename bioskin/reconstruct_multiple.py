# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import argparse
from argparse import Namespace
import reconstruct
import pandas as pd
import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description='Reconstruct skin props using learned map Reflectance->SkinProps->Reflectance')

    parser.add_argument('--input_folder',  nargs='?',
                        type=str, default='input_images/',
                        help='Input Folder with diffuse albedos to reconstruct')

    parser.add_argument('--models_folder',  nargs='?',
                        type=str, default='pretrained_models/',
                        help='Folder with trained models')

    parser.add_argument('--reconstruction_folder',  nargs='?',
                        type=str, default='reconstruction/',
                        help='Output folder with skin properties and reconstructed albedos')

    parser.add_argument('--batch_size',  nargs='?',
                        type=int, default=32768,  # 1048576
                        help='Batch size to eval the network')

    args = parser.parse_args()

    if not os.path.exists(args.reconstruction_folder):
        os.makedirs(args.reconstruction_folder)

    return args


def export_to_csv(filename, dict):
    data = pd.DataFrame.from_dict(dict)
    data.to_csv(filename)
    print(data)


def export_to_json(filename, dict):
    with open(filename, 'w') as f:
        json.dump(dict, f)
        f.close()


def save_stats(args, global_stats, x):

    datetime_prefix = str(x.year) + '_' + str(x.month) + '_' + str(x.day) + '_' + str(x.hour) + \
                      '_' + str(x.minute) + "_" + str(x.microsecond) + "__"

    errors_avg = sorted(global_stats.items(), key=lambda x: x[1]['error_average'])
    errors_max = sorted(global_stats.items(), key=lambda x: x[1]['error_max'])

    export_to_json(args.reconstruction_folder + '/' + datetime_prefix + 'stats_avg_errors.json', errors_avg)
    export_to_json(args.reconstruction_folder + '/' + datetime_prefix + 'stats_max_errors.json', errors_max)

    export_to_csv(args.reconstruction_folder + '/' + datetime_prefix + 'stats_avg_errors.csv', errors_avg)
    export_to_csv(args.reconstruction_folder + '/' + datetime_prefix + 'stats_max_errors.csv', errors_max)

    return 0


if __name__ == '__main__':

    date_time = datetime.datetime.now()

    args = parse_arguments()

    global_stats = {}
    file_log = open(args.reconstruction_folder + "/LOG.txt", "w")
    file_log.write("Models not loaded\n")

    models_list = os.listdir(args.models_folder)
    models_list_no_ext = []
    print('--> Testing models:' + '\n')
    for i in range(0, len(models_list)):
        if os.path.splitext(models_list[i])[1] == '.json':
            models_list_no_ext.append(os.path.splitext(models_list[i])[0])

    print(models_list_no_ext)

    i = 0
    for model_name in models_list_no_ext:
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Testing model ' + model_name)
        if not os.path.exists(args.reconstruction_folder + model_name):
            args_rec = {'path_to_model': args.models_folder, 'json_model': model_name,
                        'input_folder': args.input_folder, 'output_folder': args.reconstruction_folder,
                        'prefix': '', 'batch_size': args.batch_size}
            args_rec = Namespace(**args_rec)
            stats = reconstruct.reconstruct_image_batch(args_rec)

            if stats:
                global_stats[i] = stats.copy()
                save_stats(args, global_stats, date_time)
                i += 1
            else:
                file_log.write(model_name + "\n")
        else:
            print('---------------->> Skipping already tested model ' + model_name)

    save_stats(args, global_stats, date_time)
    if file_log:
        file_log.close()




