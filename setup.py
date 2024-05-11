# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import setuptools

setuptools.setup(
    name="bioskin", 
    version="1.0",
    author="Carlos Aliaga",
    url="https://github.com/facebookresearch/BioSkin",
    description="This package contains the implementation of a neural model that estimates the biophysical skin properties from a single RGB diffuse reflectance with baked occlusion, allowing deocclusion, and spectral upsampling in both the visible and near infrared (NIR) spectra (380-1000 nm). It also contains several practical applications, as well as a gui to perform interactive edits directly over the estimated skin properties", 
    packages=['bioskin','bioskin.apps','bioskin.dataset','bioskin.editing','bioskin.lighting','bioskin.loss','bioskin.parameters','bioskin.spectrum','bioskin.utils'],
    license="MIT",
    python_requires=">=3.8"
)