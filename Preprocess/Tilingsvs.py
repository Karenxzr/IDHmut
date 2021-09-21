import numpy as np
import openslide
import math
import filter
from skimage.transform import resize,rescale
import os


def tiling_qualified_separate_2_5x(svspath,targetpath,tilesize,stride, tissuepct_value=0.8):
    """
    tiling WSI based on fixed threshold, tiling 2.5x specifically

    svspath: svs path of input image
    tilesize: tilesize at level 2
    stride: at level 0, set to tilesize if no overlapping
    targetpath: location to save patches from each slide. ex. 'patches/TCGA-XX-XXXX/'
    return: save each patch seperately
    """
    # use thumbnail image at level 3, downsizing factor: 64
    slide = openslide.open_slide(svspath)
    m=round(float(slide.properties['aperio.MPP']), 2)//0.25
    level0=int(40/m)

    bestlevel=0
    for i in range(slide.level_count):
        if round(level0/ slide.level_downsamples[i], 1) // 2.5 > 0:
            bestlevel = i
    rescale_factor=round(level0 / slide.level_downsamples[bestlevel],1)/2.5

    input_img = np.array(slide.read_region((0, 0), bestlevel, slide.level_dimensions[bestlevel]))[:, :, :3]
    if rescale_factor>1:
        input_img = np.round(rescale(input_img, 1/rescale_factor, multichannel=True, preserve_range=True, anti_aliasing=True)).astype(np.uint8)
    img_h, img_w = input_img.shape[:2]

    # calculate tile numbers (step_num_h*step_num_w)
    img_h_adj = img_h - tilesize
    img_w_adj = img_w - tilesize
    step_num_h = math.floor(img_h_adj / stride) + 1
    step_num_w = math.floor(img_w_adj / stride) + 1
    print('tile number: ' + str(step_num_h * step_num_w))

    tile_coord = []  # record coordinates at level 0 scale
    for w in range(step_num_w):
        for h in range(step_num_h):
            x = int(w * stride)
            y = int(h * stride)
            sub_img = input_img[y:y+tilesize,x:x+tilesize,:]
            # QC
            if filter.tissue_pct(sub_img) > tissuepct_value:
                tile_coord.append((x, y))
                tile = sub_img

                filename = os.path.join(targetpath,str(int(x*16)) + '_' + str(int(y*16)) + '.npy')
                np.save(filename, tile)
    slide.close()
    
def tiling_qualified_separate_5x(svspath,targetpath,tilesize,stride, tissuepct_value=0.8):
    """
    tiling WSI based on fixed threshold, tiling 5x specifically

    svspath: svs path of input image
    tilesize: tilesize at level 2
    stride: at level 0, set to tilesize if no overlapping
    targetpath: location to save patches from each slide. ex. 'patches/TCGA-XX-XXXX/'
    return: save each patch seperately
    """
    
    slide = openslide.open_slide(svspath)
    m=round(float(slide.properties['aperio.MPP']), 2)//0.25
    level0=int(40/m)

    bestlevel=0
    for i in range(slide.level_count):
        if round(level0/ slide.level_downsamples[i], 1) // 5 > 0:
            bestlevel = i
    rescale_factor=round(level0 / slide.level_downsamples[bestlevel],1)/5
    
    input_img = np.array(slide.read_region((0, 0), bestlevel, slide.level_dimensions[bestlevel]))[:, :, :3]
    if rescale_factor>1:
        input_img = np.round(rescale(input_img, 1/rescale_factor, multichannel=True, preserve_range=True, anti_aliasing=True)).astype(np.uint8)
    img_h, img_w = input_img.shape[:2]

    # calculate tile numbers (step_num_h*step_num_w)
    img_h_adj = img_h - tilesize
    img_w_adj = img_w - tilesize
    step_num_h = math.floor(img_h_adj / stride) + 1
    step_num_w = math.floor(img_w_adj / stride) + 1
    print('tile number: ' + str(step_num_h * step_num_w))

    tile_coord = []  # record coordinates at level 0 scale
    for w in range(step_num_w):
        for h in range(step_num_h):
            x = int(w * stride)
            y = int(h * stride)
            sub_img = input_img[y:y+tilesize,x:x+tilesize,:]
            # QC
            if filter.tissue_pct(sub_img) > tissuepct_value:
                tile_coord.append((x, y))
                tile = sub_img

                filename = os.path.join(targetpath,str(int(x*8)) + '_' + str(int(y*8)) + '.npy')
                np.save(filename, tile)
    slide.close()
    
def tiling_qualified_separate_10x(svspath,targetpath,tilesize,stride, tissuepct_value=0.8):
    """
    tiling WSI based on fixed threshold, tiling 10x specifically

    svspath: svs path of input image
    tilesize: tilesize at level 2
    stride: at level 0, set to tilesize if no overlapping
    targetpath: location to save patches from each slide. ex. 'patches/TCGA-XX-XXXX/'
    return: save each patch seperately
    """
    # use thumbnail image at level 3, downsizing factor: 64
    slide = openslide.open_slide(svspath)
    m=round(float(slide.properties['aperio.MPP']), 2)//0.25
    level0=int(40/m)

    bestlevel=0
    for i in range(slide.level_count):
        if round(level0/ slide.level_downsamples[i], 1) // 10 > 0:
            bestlevel = i
    rescale_factor=round(level0 / slide.level_downsamples[bestlevel],1)/10

    input_img = np.array(slide.read_region((0, 0), bestlevel, slide.level_dimensions[bestlevel]))[:, :, :3]
    if rescale_factor>1:
        input_img = np.round(rescale(input_img, 1/rescale_factor, multichannel=True, preserve_range=True, anti_aliasing=True)).astype(np.uint8)
    img_h, img_w = input_img.shape[:2]

        # calculate tile numbers (step_num_h*step_num_w)
    img_h_adj = img_h - tilesize
    img_w_adj = img_w - tilesize
    step_num_h = math.floor(img_h_adj / stride) + 1
    step_num_w = math.floor(img_w_adj / stride) + 1
    print('tile number: ' + str(step_num_h * step_num_w))

    tile_coord = []  # record coordinates at level 0 scale
    for w in range(step_num_w):
        for h in range(step_num_h):
            x = int(w * stride)
            y = int(h * stride)
            sub_img = input_img[y:y+tilesize,x:x+tilesize,:]
            # QC
            if filter.tissue_pct(sub_img) > tissuepct_value:
                tile_coord.append((x, y))
                tile = sub_img
                #np.save(targetpath +'DX_'+manage.DX(svspath)+'_'+str(int(x*16)) + '_' + str(int(y*16)) + '.npy', tile)
                filename = os.path.join(targetpath, str(int(x * 4)) + '_' + str(int(y * 4)) + '.npy')
                np.save(filename, tile)
    slide.close()

def tiling_qualified_separate_20x(svspath,targetpath,tilesize,stride, tissuepct_value=0.8):
    """
    tiling WSI based on fixed threshold, tiling 20x specifically
    """
    np.seterr(all='ignore')
    slide = openslide.open_slide(svspath)
    m = round(float(slide.properties['aperio.MPP']), 2) // 0.25
    level0 = int(40 / m)
    level_n = level0 / int(slide.level_downsamples[-1])

    upscale_factor = int(20 / level_n)  # calculate relative tile size in thumbnail
    downscale_factor = int(level0 / 20)  # resize from level0 to 20x

    thumbnail = np.array(slide.read_region((0, 0), len(slide.level_dimensions) - 1, slide.level_dimensions[-1]))[:, :,:3]
    
    # transfer size to thumbnail sizes
    tilesize_small = tilesize / upscale_factor
    stride_small = stride / upscale_factor
    img_h = int(thumbnail.shape[0]-offset/upscale_factor)
    img_w = int(thumbnail.shape[1]-offset/upscale_factor)

    # calculate tile numbers (step_num_h*step_num_w)
    img_h_adj = img_h - tilesize_small
    img_w_adj = img_w - tilesize_small
    step_num_h = math.floor(img_h_adj / stride_small) + 1
    step_num_w = math.floor(img_w_adj / stride_small) + 1
    print('tile number: ' + str(step_num_h * step_num_w))

    if offset>0:
        thumbnail = thumbnail[int(offset/upscale_factor):-int(offset/upscale_factor),int(offset/upscale_factor):-int(offset/upscale_factor),:]
    #estimate h score
    h_value = min(round(filter.h_score(thumbnail)*h_factor,2),0.5)

    tile_coord = []  # record coordinates at level 0 scale
    for w in range(step_num_w):
        for h in range(step_num_h):
            sub_img = thumbnail[int(h * stride_small):int(h * stride_small + tilesize_small),
                      int(w * stride_small):int(w * stride_small + tilesize_small), :]
            # QC
            if filter.tissue_pct(sub_img) > tissuepct_value and filter.h_score(sub_img)>h_value:
                x = int((w * stride + offset)* downscale_factor)
                y = int((h * stride + offset)* downscale_factor)
                tile_coord.append((x, y))
                tile = np.array(slide.read_region((x, y), 0,
                                                  (int(tilesize*downscale_factor), int(tilesize*downscale_factor))))[:, :, :3]
                if downscale_factor>1:
                    tile = resize(tile, (tilesize, tilesize, 3),anti_aliasing=True, preserve_range=True)
                    tile = tile.astype(np.uint8)

                filename = os.path.join(targetpath, str(x) + '_' + str(y) + '.npy')
                np.save(os.path.join(filename, tile)
   
    slide.close()
    
def tiling_qualified_separate_20x(svspath,targetpath,tilesize,stride,tissuepct_value=0.8,h_factor=2,offset=0):
    """
    tiling WSI based on fixed threshold, tiling 20x specifically
    """
    np.seterr(all='ignore')
    slide = openslide.open_slide(svspath)
    m = round(float(slide.properties['aperio.MPP']), 2) // 0.25
    level0 = int(40 / m)
    level_n = level0 / int(slide.level_downsamples[-1])

    upscale_factor = int(20 / level_n)  # calculate relative tile size in thumbnail
    downscale_factor = int(level0 / 20)  # resize from level0 to 20x

    thumbnail = np.array(slide.read_region((0, 0), len(slide.level_dimensions) - 1, slide.level_dimensions[-1]))[:, :,
                :3]


    # transfer size to thumbnail sizes
    tilesize_small = tilesize / upscale_factor
    stride_small = stride / upscale_factor
    img_h = int(thumbnail.shape[0]-offset/upscale_factor)
    img_w = int(thumbnail.shape[1]-offset/upscale_factor)

    # calculate tile numbers (step_num_h*step_num_w)
    img_h_adj = img_h - tilesize_small
    img_w_adj = img_w - tilesize_small
    step_num_h = math.floor(img_h_adj / stride_small) + 1
    step_num_w = math.floor(img_w_adj / stride_small) + 1
    print('tile number: ' + str(step_num_h * step_num_w))

    if offset>0:
        thumbnail = thumbnail[int(offset/upscale_factor):-int(offset/upscale_factor),int(offset/upscale_factor):-int(offset/upscale_factor),:]
    #estimate h score
    h_value = min(round(filter.h_score(thumbnail)*h_factor,2),0.5)

    tile_coord = []  # record coordinates at level 0 scale
    for w in range(step_num_w):
        for h in range(step_num_h):
            sub_img = thumbnail[int(h * stride_small):int(h * stride_small + tilesize_small),
                      int(w * stride_small):int(w * stride_small + tilesize_small), :]
            # QC
            if filter.tissue_pct(sub_img) > tissuepct_value and filter.h_score(sub_img)>h_value:
                x = int((w * stride + offset)* downscale_factor)
                y = int((h * stride + offset)* downscale_factor)
                tile_coord.append((x, y))
                tile = np.array(slide.read_region((x, y), 0,
                                                  (int(tilesize*downscale_factor), int(tilesize*downscale_factor))))[:, :, :3]
                if downscale_factor>1:
                    tile = resize(tile, (tilesize, tilesize, 3),anti_aliasing=True, preserve_range=True)
                    tile = tile.astype(np.uint8)

                filename = os.path.join(targetpath, str(x) + '_' + str(y) + '.npy')
                np.save(os.path.join(filename, tile)

    slide.close()


