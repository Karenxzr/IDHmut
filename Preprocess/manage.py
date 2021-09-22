import os
import re
import numpy as np
from Preprocess import filter

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
    '''
    used to extract Barcode for TCGA data
    '''
    pattern = re.compile(r'(TCGA-.{2}-.{4})')
    barcode = re.search(pattern, string).group(1)
    return barcode

def DX(string):
    '''
    used to extract slide number for TCGA data
    '''
    pattern = re.compile(r'-DX(\w|\d+)-')
    dx = re.search(pattern, string).group(1)
    return dx

def DX_(string):
    pattern = re.compile(r'_DX(\d|\w)_')
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

