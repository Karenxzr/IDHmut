# IDHmut

## 1. Tiling Slides
The first step is to tile slides into patches and store patches into specified folder. A csv file with two columns ('SVS_Path','PatientID') is required to store svs file paths and IDs.

User has to assign two arguments: --df_path and --target_path

Example command:

`python3 Tiling.py --df_path '/path/to/csv/file.csv' --target_path '/root/folder/path/for/tiles' `

Some optional arguments:

'--workers',type=int,default=8

'--tilesize',type=int,default=256

'--stride',type=int,default=256 (when set to a number smaller than tilesize, tiles will have overlaps)

'--tissuepct_value',type=int,default=0.7 (for quality control)

'--magnification',type=str,default=None (when leave it as default 2.5x, 5x,10x,20x will all be tiled, otherwise assign one target magnification)

Example command: (will tile 2.5x patches with tissue percentage over 50%)

`python3 Tiling.py --df_path '/path/to/csv/file.csv' --target_path '/root/folder/path/for/tiles'  --tissuepct_value 0.5 --magnification '2.5x'`


## 2. Training Model

`python3 Train.py --result_dir ‘/path/for/model’ --df_path ‘/path/to/metadata.csv’ --workers 16 --CNN densenet --no_age --patch_n 200 --spatial_sample_off --n_epoch 100 --lr 0.00001 --optimizer Adam --use_scheduler --balance 0.5 --balance_training --freeze_batchnorm --pooling mean --notes model0`

## 3. Evaluate Model
`python3 Evaluation.py`
