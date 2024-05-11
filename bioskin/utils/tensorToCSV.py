# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import pandas as pd
import argparse

def export(x, name):
    x_np = x.numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv(name + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export .pt tensor to csv')
    parser.add_argument('--dataset', nargs='?',
                        type=str,
                        help='Dataset path')
    args = parser.parse_args()
    x = torch.load(args.dataset)
    export(x, args.dataset)
