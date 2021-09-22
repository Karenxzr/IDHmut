from Preprocess import Tilingsvs
import argparse
import pandas as pd
import os
import re
from multiprocessing import Pool
from Preprocess import manage


parser = argparse.ArgumentParser(description = 'Tiling SVS Files into Patches')
#mandatory
parser.add_argument('--df_path', type=str)#data frame with two columns at least:'SVS_Path','PatientID'
parser.add_argument('--target_path', type=str)
#optional
parser.add_argument('--workers',type=int,default=8)
parser.add_argument('--tilesize',type=int,default=256)
parser.add_argument('--stride',type=int,default=256)
parser.add_argument('--tissuepct_value',type=int,default=0.7)
parser.add_argument('--magnification',action="store")
_StoreAction(option_strings=['-m', '--magnification'], dest='magnification', nargs='*', const=None, default=None, type=None, choices=None, metavar=None)

def tiling(svspath,patientid,targetpath,tilesize=256,stride=256,tissuepct_value=0.7,magnification=None):
    """
    :param magnification:single magnification from ['2.5x','5x','10x','20x']
    """
    
    if magnification is not None:
        assert (
          magnification.lower() in ['2.5x','5x','10x','20x']
        ),f"invalid magnification, choose from ['2.5x','5x','10x','20x']"
        mag =  [magnification.lower()]
    else:
        mag =  ['2.5x','5x','10x','20x']
    
    print('tiling on magnifications of'+str(mag))
    
    if '2.5x' in mag:
        patch_path2_5x = os.path.join(targetpath, '2_5x', patientid)
        if not os.path.isdir(patch_path2_5x):
            os.makedirs(patch_path2_5x)
        Tilingsvs.tiling_qualified_separate_2_5x(svspath=svspath,targetpath=patch_path2_5x,tilesize=tilesize,stride=stride, tissuepct_value=tissuepct_value)
    
    if '5x' in mag:
        patch_path5x = os.path.join(targetpath, '5x', patientid)
        if not os.path.isdir(patch_path5x):
            os.makedirs(patch_path5x)
        Tilingsvs.tiling_qualified_separate_5x(svspath=svspath,targetpath=patch_path5x,tilesize=tilesize,stride=stride, tissuepct_value=tissuepct_value)
            
    if '10x' in mag:
        patch_path10x = os.path.join(targetpath, '10x', patientid)
        if not os.path.isdir(patch_path10x):
            os.makedirs(patch_path10x)
        Tilingsvs.tiling_qualified_separate_10x(svspath=svspath,targetpath=patch_path10x,tilesize=tilesize,stride=stride, tissuepct_value=tissuepct_value)
            
    if '20x' in mag:
        patch_path20x = os.path.join(targetpath, '20x', patientid)
        if not os.path.isdir(patch_path20x):
            os.makedirs(patch_path20x)
        Tilingsvs.tiling_qualified_separate_20x(svspath=svspath,targetpath=patch_path20x,tilesize=tilesize,stride=stride, tissuepct_value=tissuepct_value)
            


def main():
    # set environment
    global args
    args = parser.parse_args()
    p = Pool(args.workers)

    #read dataframe
    df = pd.read_csv(args.df_path)
    svs_paths = list(df['SVS_Path'])
    patientid = list(df['PatientID'])
    
    iter_object=[]
    for ind in range(len(svs_paths)):
        svs = svs_paths[ind]
        patient_id = patientid[ind]
        # create iter object
        iter_object.append((svs, patient_id ,args.target_path))

    #start task
    p.starmap(tiling, iter(iter_object))
    #p.close()
    #p.join()


if __name__ == '__main__':
    main()

