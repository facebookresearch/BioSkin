# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = 'true'
import torch
import numpy as np
import bioskin.editing.operators as op


SKIN_PROPS = ["Melanin", "Haemoglobin", "Thickness", "Blood Oxygenation", "Melanin Type Ratio", "Occlusion"]

# Plausible range, not including rushes, spots, etc.
REGULAR_SKIN_MIN_VALUES = [0.005, 0.001, 0.005, 0.6, 0.72, 0.001]
REGULAR_SKIN_MAX_VALUES = [0.40, 0.0225, 0.025, 0.8, 0.76, 1.0]

# Plausible range
SKIN_MIN_VALUES = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
SKIN_MAX_VALUES = [0.45, 0.25, 0.035, 1.0, 1.0, 1.0]

# Full range
MIN_PROP_VALUES = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
MAX_PROP_VALUES = [1.0, 1.0, 0.035, 1.0, 1.0, 1.0]


class Parameter:
    def __init__(self, name, image):
        self.name = name
        self.map = image.copy()
        self.map_edited = image.copy()
        self.map_edited_last = image.copy()  # to concatenate operators
        self.previous_operator = None
        self.editing_stack = []

    def reset(self):
        self.map_edited = self.map.copy()
        self.map_edited_last = self.map.copy()
        self.previous_operator = None
        self.editing_stack = []

    def update(self, image):
        self.map_edited = image.copy()
        self.map_edited_last = image.copy()  # to concatenate operators
        self.previous_operator = None
        self.editing_stack = []

    def mean(self):
        if isinstance(self.map_edited, np.ndarray):
            tensor = torch.from_numpy(self.map_edited)
        else:
            tensor = self.map_edited.clone().detach()
        return torch.mean(tensor[:, :, 0])

    def mean_masked(self):
        if isinstance(self.map_edited, np.ndarray):
            img = torch.from_numpy(self.map_edited)
        else:
            img = self.map_edited.clone().detach()
        img = img[:, :, 0]
        mask = (img < 0.95) & (img > 0.001)
        return torch.mean(img[mask])

    def min(self):
        if isinstance(self.map_edited, np.ndarray):
            tensor = torch.from_numpy(self.map_edited)
        else:
            tensor = self.map_edited.clone().detach()
        return torch.min(tensor[:, :, 0])

    def max(self):
        if isinstance(self.map_edited, np.ndarray):
            tensor = torch.from_numpy(self.map_edited)
        else:
            tensor = self.map_edited.clone().detach()
        return torch.max(tensor[:, :, 0])

    def mode(self):
        if isinstance(self.map_edited, np.ndarray):
            tensor = torch.from_numpy(self.map_edited)
        else:
            tensor = self.map_edited.clone().detach()
        return op.compute_mode(tensor[:, :, 0])

    def mode_masked(self):
        if isinstance(self.map_edited, np.ndarray):
            img = torch.from_numpy(self.map_edited)
        else:
            img = self.map_edited.clone().detach()
        img = img[:, :, 0]
        mask = (img != 1.0) & (img != 0.0)
        return op.compute_mode(img[mask])

    def push_operation(self, operator, value):
        self.editing_stack.append((operator, value))

    def pop_operation(self):
        return self.editing_stack.pop()

    def apply_operation(self, operator, value):
        if operator == self.previous_operator:
            self.pop_operation()  # new operation overwrites old one if same operator
        else:
            self.map_edited = self.map_edited_last
        edited_map_tensor = op.apply_operator(torch.tensor(self.map_edited[:, :, 0]), operator, value)
        edited_map_3c = torch.zeros(edited_map_tensor.shape[0], edited_map_tensor.shape[1], 3)
        edited_map_3c[:, :, 0] = edited_map_tensor.clone()
        edited_map_3c[:, :, 1] = edited_map_tensor.clone()
        edited_map_3c[:, :, 2] = edited_map_tensor.clone()
        self.map_edited_last = edited_map_3c.numpy()

        self.push_operation(operator, value)
        self.previous_operator = operator

    def apply_operations_stack(self):
        editing_stack = self.editing_stack.copy()
        while editing_stack:
            operator, value = editing_stack.pop(0)
            self.apply_operation(operator, value)

    def apply_operations_stack_external_map(self, map_in):
        editing_stack = self.editing_stack.copy()
        if isinstance(map_in, torch.Tensor):
            map = map_in.clone()
        elif isinstance(map_in, np.ndarray):
            map = map_in.copy()

        self.reset()
        while editing_stack:
            operator, value = editing_stack.pop(0)
            self.apply_operation(operator, value)
            map = op.apply_operator(map, operator, value, self)
        return map


