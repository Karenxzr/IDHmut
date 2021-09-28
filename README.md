# IDHmut
Example workflow for predict IDH mutation status from histopathological whole slide images (WSI) using deep learning.
<p align="left">
  <img src="/Figures/Picture1.png" width="600" title="workflow">
</p>

## 1. Tiling Slides
The first step is to tile the slides into patches and store them into specified folder. A csv file with two columns ('SVS_Path','PatientID') is required to store svs file paths and IDs. See csv folder for example file for --df_path argument.

User has to assign two arguments: --df_path and --target_path

### Basic command:

`python3 Tiling.py --df_path '/path/to/csv/tilingslides.csv' --target_path '/root/folder/path/for/tiles' `

Some optional arguments:

'--workers',type=int,default=8

'--tilesize',type=int,default=256

'--stride',type=int,default=256 (when set to a number smaller than tilesize, tiles will have overlaps)

'--tissuepct_value',type=int,default=0.7 (for quality control)

'--magnification',type=str,default='multi' (by default 2.5x, 5x, 10x, 20x will all be tiled, otherwise assign one target magnification). Note that for training, tiles from a specific magnification will be used. It is recommended to examine the validation performance using different magnifications to choose the optimal one. 

### Example: (will tile 2.5x patches with tissue percentage over 50%)

`python3 Tiling.py --df_path '/path/to/csv/file.csv' --target_path '/root/folder/path/for/tiles'  --tissuepct_value 0.5 --magnification '2.5x'`


## 2. Model Training

After tiling the slides, we can start training models. 

Prepare a csv file for --df_path is required. Example can be found in csv folder with name of `training.csv`. Basically, three columns are required: 1. label column, use `--y_col` to assign label name; 2. 'Path': paths for the tiles generated from each slide in the previous step; 3. 'Train_Test': containing values of 'Train'/'Validation'/'Test'. Train and Validation are required.

Some important arguments:

'--result_dir', type=str': folder path for saving the model

'--df_path': path to the training meta data

'--gpu': at least two gpus are required, default is '0,1,2,3'

'--patch_n': patch number to sample for each slide during each iteration 

'--balance': weights for balancing loss function for each class; default is 0.5. If weighting loss function for class imbalancement, --balance_training_off is suggested

'--balance_training': used to balance class imbalancement during training. Will automatically sample the same number of samples for each class. Suggest leaving --balance as default

'--CNN' choose from resnet and densenet

'--y_col' label name in the csv file. Values must be numeric: 0 or 1

'--freeze_batchnorm' suggest to set this for more stable results

'--pooling': select from attention, max and mean for aggregating embedding layers of tiles from one slide

'--A' if set pooling as attention, set a number to A for number of nodes in attention network. Default is 16.

### Example Codes:

`python3 Train.py --result_dir ‘/path/for/model’ --df_path ‘/path/to/training.csv’ --workers 16 --CNN densenet --no_age --patch_n 200 --spatial_sample_off --n_epoch 100 --lr 0.00001 --optimizer Adam --use_scheduler --balance 0.5 --balance_training --freeze_batchnorm --pooling mean --notes model0`

## 3. Model Evaluation
`python3 Evaluation.py`


# Instructions for using GPU and computational clusters
## 1. Create a conda environemnt:
`conda env create N` where N is the name of your environment

## 2. Activate the enviroment:
`conda activate N` where N is the name of your conda environment

## 2. Check the cuda version compatible with the driver
nvidia-smi

## 3. Install the correct cuda and pytorch:
`conda install pytorch torchvision cudatoolkit=11.0 -c pytorch`

## 4. Download and install cuDNN by following these [instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

## 5. Untar the downloaded cuDNN folder then move its contents to the conda library:
`cp cuda/include/cudnn*.h /path/to/your/environment/include`
`cp cuda/lib64/libcudnn* /path/to/your/environment/lib`

## 6. Example submission script for training using 2 GPUs on SLURM:

```
#! /bin/bash -l
#SBATCH --partition=panda-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --job-name=training
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

source ~/.bashrc

conda activate /path/to/your/environment

python3 code/Train.py --result_dir 'data/model/' --df_path 'MetaData_training.csv' --workers 15 --CNN densenet --no_age --y_col 'Gleason_HighLow' --patch_n 200 --spatial_sample_off --n_epoch 100 --lr 0.00001 --optimizer Adam --use_scheduler --balance 0.5 --balance_training --freeze_batchnorm --pooling mean --notes model0 --gpu 2
```
