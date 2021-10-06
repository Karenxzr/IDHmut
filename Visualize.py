from Model import visualization as v
import argparse
import os
import numpy as np
from Preprocess import filter


parser = argparse.ArgumentParser(description='Generate Prediction Map')
parser.add_argument('--magnification', type=str, default='2.5x')
parser.add_argument('--svs_path', type=str)
parser.add_argument('--model_folder_path', type=str)
parser.add_argument('--by', type=str,default='acc')
parser.add_argument('--step_pct', type=float, default=1.0)
parser.add_argument('--target_path',type=str)
parser.add_argument('--type',type=str,default='prediction')#set to filter to generate probobility map for training acceptance
parser.add_argument('--file_name',type=str,default='Prediction_Map.npy')
parser.add_argument('--down_size',type=int,default=1)



def main():
    global args
    args = parser.parse_args()
    target_path = os.path.join(args.target_path, args.file_name)
    print('target_path is: '+str(target_path))
    if args.type=='prediction':
        map_array = v.patch_prediction_pixel(args.svs_path, args.model_folder_path, args.by, args.magnification, args.step_pct, args.down_size)
        np.save(target_path,map_array)
    elif args.type=='filter':
        map_array = v.patch_filter_pixel(args.svs_path, args.model_folder_path, args.by, args.magnification, args.step_pct, args.down_size)
        np.save(target_path,map_array)


if __name__ == '__main__':
    main()
