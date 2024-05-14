# MOoSE: Multi-Orientation Sharing Experts for Open-set Scene Text Recognition

This repository is the author's implementation of the paper MOoSE: Multi-Orientation Sharing Experts for Open-set Scene Text Recognition.
![ostr101](https://github.com/lancercat/Moose/assets/59994105/4d549996-6a33-4fc2-a481-c813eb083061)

E.g.,
![problem1](https://github.com/lancercat/Moose/assets/59994105/e96210ab-0e96-4540-b7b7-a6ef976171c6)


The repo implements a Multi-Orientation Sharing Experts framework that allows you to handle seen and unseen scene text written in various orientations.
![framework](https://github.com/lancercat/Moose/assets/59994105/5aad128f-741e-4210-86a1-29575f80a8b0)


## Getting Started
Implementation has been tested on a Lenovo P360 Ultra with an RTX4060Ti GPU connected through a Thunderbolt dock (TH3P4G2).

Software: Manjaro Linux, Python 3.11

### Environment
Some dependencies will need to be compiled, so you need to follow the guide below.

https://github.com/lancercat/make_envNG



### Dataset
the training samples include English and Chinese word crops taken from ART
[28], RCTW [29], CTW [30], LSVT [31], and the Latin and Chinese sections of
the MLT [32] dataset. Note that the samples that include characters other
than the 3755 Tier1 Chinese characters, 26 English letters, and the 10 digits
(3791 classes in total) are excluded from the training set to avoid label
leaking. The testing set includes the Japanese subsets of the MLT dataset.

You can download them from the following links (you need to download both datasets)

- https://www.kaggle.com/datasets/object300/mose-extra/

- https://www.kaggle.com/vsdf2898kaggle/osocrtraining

### Models
All models are released to the following repo

- All Models(45G): https://www.kaggle.com/datasets/object300/moose-models-release

## Logs
Our training logs are released with a visualizer.

- Logs: https://www.kaggle.com/datasets/object300/log-moose-release

- Log Visualizer: https://github.com/lancercat/minijinja

Note the log visualizer includes ssh related behavior,

as it is a part of a home-grown server management library,

used for autonomous monitoring, issuing commands to, and collecting results from a server fleet.

## Results & Manual
The Results and a detailed manual is included in the manul.pdf file.
![moostr-gzsl-remix](https://github.com/lancercat/Moose/assets/59994105/7208ba0a-a86c-4f46-a7e1-9c29f42877fe)

(The filename is an intended pun, manuals, aka Pallas cats, are the oldest cats)


## Code
The methods are configured and can be launched from the neko_2023_NGNW/project_moose/[methodname] directory.


For training, launch neko_2023_NGNW/project_moose/[methodname]/train_osr_hv_cat_wandb.py

For testing, launch  neko_2023_NGNW/project_moose/[methodname]/eval_osr_hv.py

For more details, please consult the manual.

### Naming
This section lists the correspondences between names in the paper and codenames in the repo

- Horizontal
- - 1 horizontal
- - hori_sharebbn_s_05_ld_long
- Horizontal-MoE
- - 2 horizontal
- - hori_sharebbn_62_ld_long
- Single-Horizontal
- - 1 horizontal 1 vertical
- - sharebbn_05_ld_long
- Rotated
- - 2 horizontal, but rotates vertical samples and pipe them to horizontal experts
- - sharebbn_62RS_ld_long

- Share All
- - 2 horizontal 1 vertival
- - shareall_62_ld_long

- Share None
- - 2 horizontal 1 vertival
- - sharenone_62_ld_long
- MOoSE
- - 2 horizontal 1 vertival
- - sharebbn_62_ld_long
- MOoSE-XL
- - 2 horizontal 1 vertival
- - yukon_sharebbn_62_ld_long_XL

## Contact
- Chang Liu lasercat@gmx.us, chang.liu@ltu.se
- Simon Corbillé simon.corbille@associated.ltu.se
- Elisa H. Barney Smith elisa.barney@ltu.se
