import os
import re
import numpy as np
import filter

def list_svs(rootfolder,wholepath= True):
    '''
    used to list all svs file paths under the root folder
    :param rootfolder: root folder containing svs files
    :return: list of svs paths
    '''
    path = []
    for root, directories, files in os.walk(rootfolder, topdown=False):
        for file in files:
            svs_path=os.path.join(root, file)
            if '.svs' in svs_path:
                path.append(svs_path)
    if wholepath:
        path = [os.path.join(rootfolder, i) for i in path]
    return path

def list_png(rootfolder,wholepath= True):
    '''
    used to list all npy file paths under the root folder
    :param rootfolder: root folder containing npy files
    :return: list of npy paths
    '''
    path = []
    for root, directories, files in os.walk(rootfolder, topdown=False):
        for file in files:
            png_path=os.path.join(root, file)
            if '.png' in png_path:
                path.append(png_path)
    if wholepath:
        path = [os.path.join(rootfolder,i) for i in path]
    return path

def list_npy(rootfolder,wholepath= True):
    '''
    used to list all npy file paths under the root folder
    :param rootfolder: root folder containing npy files
    :return: list of npy paths
    '''
    path = []
    for root, directories, files in os.walk(rootfolder, topdown=False):
        for file in files:
            npy_path=os.path.join(root, file)
            if '.npy' in npy_path:
                path.append(npy_path)
    if wholepath:
        path = [os.path.join(rootfolder,i) for i in path]
    return path

def Barcode(string):
    pattern = re.compile(r'(TCGA-.{2}-.{4})')
    barcode = re.search(pattern, string).group(1)
    return barcode

def DX(string):
    pattern = re.compile(r'-DX(\d)\.')
    dx = re.search(pattern, string).group(1)
    return dx

def DX_(string):
    pattern = re.compile(r'_DX(\d).')
    dx = re.search(pattern, string).group(1)
    return dx


def Coordinates(string):
    '''
    :param string: input in format of XXX_x_y
    :return: (x, y)
    '''
    pattern = re.compile(r'.+_(\d+)_(\d+)')
    x = re.search(pattern, string).group(1)
    y = re.search(pattern, string).group(2)
    return int(x), int(y)

def Fold_Image(image,num_w,num_h,tissue_pct=0):
    '''
    :input image numpy
    :param fold image into 4d array. when tissue_pct, no filter will be used
    :return: 4d folded array
    '''
    image_w=image.shape[1]
    image_h=image.shape[0]
    tile_w = int(image_w/num_w)
    tile_h = int(image_h/num_h)
    out = []
    for w in range(num_w):
        for h in range(num_h):
            sub_img = image[h*tile_h:(h+1)*tile_h,w*tile_w:(w+1)*tile_w,:]
            if tissue_pct==0:
                out.append(sub_img)
            else:
                if filter.tissue_pct>=tissue_pct:
                    out.append(sub_img)
                else:
                    pass
    out = np.stack(out)
    return out
