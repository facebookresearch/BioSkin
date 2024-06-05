# BioSkin

<img src=figures/teaser2.jpg>

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

This is the implementation of "**A Hyperspectral Space of Skin Tones for Inverse Rendering of Biophysical Skin Properties**". 
[Carlos Aliaga](https://scholar.google.com/citations?user=k68oeiEAAAAJ&hl=en&oi=ao), 
[Mandy Xia](https://mandyxmq.github.io/), [Hao Xie](https://www.linkedin.com/in/hao-xie/), 
[Adrian Jarabo](https://scholar.google.com/citations?hl=en&user=pXKBhbkAAAAJ), 
[Gustav Braun](https://www.linkedin.com/in/gusbraun/), 
[Christophe Hery](https://www.linkedin.com/in/christophehery/). 
Computer Graphics Forum (EGSR 2023) [[Paper]](https://diglib.eg.org/handle/10.1111/cgf14887).
Further references: 
[[ArXiv 2022 Paper]](https://research.facebook.com/publications/estimation-of-spectral-biophysical-skin-properties-from-captured-rgb-albedo/) 
[[SIG'22 Course]](https://blog.selfshadow.com/publications/s2022-spectral-course/).

### Citation
```
@article {10.1111:cgf.14887,
journal = {Computer Graphics Forum},
title = {{A Hyperspectral Space of Skin Tones for Inverse Rendering of Biophysical Skin Properties}},
author = {Aliaga, Carlos and Xia, Mengqi and Xie, Hao and Jarabo, Adrian and Braun, Gustav and Hery, Christophe},
year = {2023},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.14887}
}
```

## Installation

### Download
`git clone git@github.com:facebookresearch/BioSkin.git`

### Requirements

`python -m venv ./venv` \
`.\venv\Scripts\Activate.ps1` \
`python -m pip install --upgrade pip` \
`python -m pip install -r requirements.txt`

### Packaging

Simply run `python -m pip install .` and the package should be available by `import bioskin`.

## Running
To have an overview of the different use cases and practical applications, check _examples.py_.
Run selecting the application with --mode. Example:
```
python examples.py --mode GUI --input_folder test_images/ --output_folder results/ --path_to_model pretrained_models/ 
--json_model BioSkinAO --max_width 800
```

* _GUI_: Interactive estimation and editing tool. See next section.
* _Reconstruction_: Estimates skin properties from diffuse albedo textures.
* _Sampler_:  Skin tones sampler.
* _Augment_:  Augment dataset with skin properties.
* _OptimizeCCM_:  Optimize color correction matrix given an image or a set of images (with consistent illumination).
* _Homogenize_:  Match skin properties and reflectance to a target (all textures to the last one loaded).
* _OptimizeProps_:  Optimize bio maps using the decoder only (skin properties --> reflectance spectra).

For training new models, RGB or spectral of different wavelength ranges, different losses, etc. please check 
_bioskin/train.py_ or _bioskin/train_exposure_aware.py_.

### GUI/Demo
* The GUI displays the first character from the input folder. Switch using '_Character_' menu.
* The main window can be interacted with like general maps (zooming, moving, etc).
* Clicking on the component images (and small reflectance) on the right switches the main view to that specific texture 
for more details.
* Use '_View_' menu to switch from visible range reflectance, to near infrared reflectance, original and edited (key `e` 
toggles the original and the edited versions). 
* Edits are stacked one after another. Once you are done, press '_Push Edits and Update Reflectance_' to see the effect 
on the albedo. Also use:
  * '_Reset Reflectance_' button to recover the original skin reflectance and all unedited property maps.
  * '_Reset Property_' button to undo and recover the original skin property map.
  * '_Load from File_' button to perform edits outside the program. Edit any map (the .jpeg) in the 
_output_folder_ and press the '_Load from File_' button.

### License
BioSkin is MIT licensed, as found in the LICENSE file.

<img src=figures/gui.jpg width=1000>




