# Iris recognition using CNNs for feature extraction


## Requirements

Install Anaconda environment from the `environment.yml` file.

## Files

`train.py` - model training for closed set recognition (check argparser for arguments)  
`eval_closed_set.py` - closed set evaluation script  
`eval_open_set.py` - open set evaluation script  
`cv_open_set.py` - open set cross validation script  

For training and evaluation you need preprocessed images. Scripts can be found in `/scripts` directory.
Pretrained models can be downloaded [here](https://drive.google.com/file/d/1pf-X13Dpsj-Q8p5Xzl0-uCoB1IeFmYD7/view?usp=sharing) and preprocessed images for open and closed set recognition [here](https://drive.google.com/file/d/172GZgrAzNrA146BjFpF1vXtAH-zWymlA/view?usp=sharing).
Models should be unzipped into `/models` directory and image folders to `/data` directory. Both are in .gitignore due to filesize issues.