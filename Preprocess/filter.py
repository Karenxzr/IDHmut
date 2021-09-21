import numpy as np
from skimage.color import  rgb2hed

##core function: mask/ tissue_pct
##generate background and penmarks binary mask (0: black, tissue . 1: white, background and penmarks)



def tissue_pct(tile, maskpen=True):
    #arg: excludepen: if true, also mask penmarks
    if maskpen:
        pct = 1 - np.mean(mask(tile))
    else:
        pct = 1 - np.mean(filter_background(tile))
    return pct

def h_score(tile, h_threshold=0.4):
    tile_h = rgb2hed(tile)[...,0]
    tile_h = np.interp(tile_h, (tile_h.min(), tile_h.max()), (0, 1))
    tile_h_binary = tile_h>h_threshold
    tile_h_binary = tile_h_binary.astype(np.uint8)
    return tile_h_binary.mean()


def filter_background(rgb):
    '''
    background retrun true
    '''
    rgb =  np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    white_background = rgb > 215
    black_background = rgb < 40
    background = np.logical_or(white_background,black_background)
    return background

def filter_nored(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh):
  r = rgb[:, :, 0] > red_lower_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] < blue_upper_thresh
  result = ~(r & g & b)
  return result

def filter_nored_pen(rgb):
  result = filter_nored(rgb, 150, 80, 90) & \
           filter_nored(rgb, 110, 20, 30) & \
           filter_nored(rgb, 185, 65, 105) & \
           filter_nored(rgb, 195, 85, 125) & \
           filter_nored(rgb, 220, 115, 145) & \
           filter_nored(rgb, 125, 40, 70) & \
           filter_nored(rgb, 200, 120, 150) & \
           filter_nored(rgb, 100, 50, 65) & \
           filter_nored(rgb, 85, 25, 45)
  return result

def filter_nogreen(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh):
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] > green_lower_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  return result

def filter_nogreen_pen(rgb):
  result = filter_nogreen(rgb, 150, 160, 140) & \
           filter_nogreen(rgb, 70, 110, 110) & \
           filter_nogreen(rgb, 45, 115, 100) & \
           filter_nogreen(rgb, 30, 75, 60) & \
           filter_nogreen(rgb, 195, 220, 210) & \
           filter_nogreen(rgb, 225, 230, 225) & \
           filter_nogreen(rgb, 170, 210, 200) & \
           filter_nogreen(rgb, 20, 30, 20) & \
           filter_nogreen(rgb, 50, 60, 40) & \
           filter_nogreen(rgb, 30, 50, 35) & \
           filter_nogreen(rgb, 65, 70, 60) & \
           filter_nogreen(rgb, 100, 110, 105) & \
           filter_nogreen(rgb, 165, 180, 180) & \
           filter_nogreen(rgb, 140, 140, 150) & \
           filter_nogreen(rgb, 185, 195, 195)
  return result


def filter_noblue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh):
  r = rgb[:, :, 0] < red_upper_thresh
  g = rgb[:, :, 1] < green_upper_thresh
  b = rgb[:, :, 2] > blue_lower_thresh
  result = ~(r & g & b)
  return result


def filter_noblue_pen(rgb):
  result = filter_noblue(rgb, 60, 120, 190) & \
           filter_noblue(rgb, 120, 170, 200) & \
           filter_noblue(rgb, 175, 210, 230) & \
           filter_noblue(rgb, 145, 180, 210) & \
           filter_noblue(rgb, 37, 95, 160) & \
           filter_noblue(rgb, 30, 65, 130) & \
           filter_noblue(rgb, 130, 155, 180) & \
           filter_noblue(rgb, 40, 35, 85) & \
           filter_noblue(rgb, 30, 20, 65) & \
           filter_noblue(rgb, 90, 90, 140) & \
           filter_noblue(rgb, 60, 60, 120) & \
           filter_noblue(rgb, 110, 110, 175)
  return result


def mask(rgb):
    img_nopen=filter_nogreen_pen(rgb) & filter_nored_pen(rgb) & filter_noblue_pen(rgb)
    img_pen = (1-img_nopen).astype(np.bool) #pen mask
    background=filter_background(rgb) #background mask
    mask= img_pen | background #general mask
    return(mask)
