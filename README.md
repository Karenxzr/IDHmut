# IDHmut

## Tiling Slides
The first step is to tiling slides into patches and store patches into specified folder. A csv file with two columns ('SVS_Path','PatientID') is required to store svs file paths and IDs.


`python3 Tiling.py --df_path '/path/to/csv/file.csv' --target_path '/root/folder/path/for/tiles' `


#optional
parser.add_argument('--workers',type=int,default=8)
parser.add_argument('--tilesize',type=int,default=256)
parser.add_argument('--stride',type=int,default=256)
parser.add_argument('--tissuepct_value',type=int,default=0.7)
parser.add_argument('--magnification',type=str,default=None)
