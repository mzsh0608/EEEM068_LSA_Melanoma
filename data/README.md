# Data Directory

This directory stores local dataset metadata and generated split files.

The raw SIIM-ISIC image dataset is not stored in the Git repository.

Expected local structure:

data/
├── train.csv
├── train_folds.csv
└── train_images/
    └── *.jpg

Raw images are excluded through `.gitignore`.