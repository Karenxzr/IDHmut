import numpy as np
import sys
import os
from skimage.transform import resize
import openslide
import torch
import math
import torchvision.transforms as transforms
from Preprocess import filter


def patch_filter_pixel(svs_path,model_folder_path,by,magnification, step_pct,down_size=1):
    #magnification: '2.5x','5x','10x','20x'
    #step_pct: step length/patch size

    #---read slide
    slide=openslide.open_slide(svs_path)
    print('slide loaded')
    # ---calculate magnification at level0 and rescale factor
    m = int(round(float(slide.properties['aperio.MPP']) / 0.25, 0))
    level0 = int(40 / m)
    print('level0 magnification is'+str(level0))
    mag_dict = dict({'2.5x': 16, '5x': 8, '10x': 4, '20x': 2})
    scale_factor = mag_dict[magnification]//m

    #---prepare canvas
    (w0, h0) = slide.level_dimensions[0]
    print('slide dimension on level0 is'+str(w0)+'_'+str(h0))
    (w, h) = (w0//scale_factor, h0//scale_factor)
    print('slide dimension on target magnification is' + str(w) + '_' + str(h))
    step = int(step_pct * 256)
    print('step size is'+str(step))
    step_num_w = math.floor((w-256)/step)
    step_num_h = math.floor((h - 256) / step)
    tile_number = (step_num_w+1)*(step_num_h+1)
    print('tile number to evaluate is: '+str(tile_number))

    count_map = np.zeros((h//down_size, w//down_size))+sys.float_info.epsilon
    pred_map = np.zeros((h//down_size, w//down_size))
    single_patch_count = np.ones((256//down_size, 256//down_size))

    #---read patch and predict
    patch_size = int(256 * scale_factor)
    for w_ in range(step_num_w + 1):
        for h_ in range(step_num_h + 1):
            (i, j) = (int(w_*step*scale_factor),int(h_*step*scale_factor))
            patch_level0 = np.array(slide.read_region((i, j), 0, (patch_size, patch_size)))[:, :, :3]
            patch = np.array(resize(patch_level0, (256, 256, 3),preserve_range=True))
            
            #tissue pct for training acceptance 
            pct = filter.tissue_pct(patch)
            print('tissue percentage is:'+str(pct))
            if magnification=='2.5x':
                if pct>=0.8:
                    patch_pred = 1
                else:
                    patch_pred = 0
            else: 
                if pct>=0.5:
                    patch_pred = 1
                else:
                    patch_pred = 0
           

            w_start = int(w_*step)//down_size
            w_end = int(w_start+256//down_size)
            h_start = int(h_ * step)//down_size
            h_end = int(h_start+256//down_size)

            count_map[h_start:h_end,w_start:w_end] += single_patch_count
            pred_map[h_start:h_end,w_start:w_end] += patch_pred*single_patch_count

    prediction_map = pred_map/count_map
    return prediction_map

def patch_prediction_pixel(svs_path,model_folder_path,by,magnification, step_pct,down_size=1):
    #magnification: '2.5x','5x','10x','20x'
    #step_pct: step length/patch size

    #---read slide
    slide=openslide.open_slide(svs_path)
    print('slide loaded')
    # ---calculate magnification at level0 and rescale factor
    m = int(round(float(slide.properties['aperio.MPP']) / 0.25, 0))
    level0 = int(40 / m)
    print('level0 magnification is'+str(level0))
    mag_dict = dict({'2.5x': 16, '5x': 8, '10x': 4, '20x': 2})
    scale_factor = mag_dict[magnification]//m

    #---prepare model
    file_list = os.listdir(model_folder_path)
    model0_name = ''
    model1_name = ''
    if by == 'loss':
        model0_name = [file for file in file_list if 'vlossCNN' in file][0]
        model1_name = [file for file in file_list if 'vlossAT' in file][0]
    elif by == 'acc':
        model0_name = [file for file in file_list if 'vaccCNN' in file][0]
        model1_name = [file for file in file_list if 'vaccAT' in file][0]
    model0_path = os.path.join(model_folder_path, model0_name)
    model1_path = os.path.join(model_folder_path, model1_name)

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    model0 = torch.load(model0_path, map_location=device0)
    model1 = torch.load(model1_path, map_location=device1)
    print('model loaded')
    model0.eval()
    model1.eval()

    transform_ = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    #---prepare canvas
    (w0, h0) = slide.level_dimensions[0]
    print('slide dimension on level0 is'+str(w0)+'_'+str(h0))
    (w, h) = (w0//scale_factor, h0//scale_factor)
    print('slide dimension on target magnification is' + str(w) + '_' + str(h))
    step = int(step_pct * 256)
    print('step size is'+str(step))
    step_num_w = math.floor((w-256)/step)
    step_num_h = math.floor((h - 256) / step)
    tile_number = (step_num_w+1)*(step_num_h+1)
    print('tile number to evaluate is: '+str(tile_number))

    count_map = np.zeros((h//down_size, w//down_size))+sys.float_info.epsilon
    pred_map = np.zeros((h//down_size, w//down_size))
    single_patch_count = np.ones((256//down_size, 256//down_size))

    #---read patch and predict
    patch_size = int(256 * scale_factor)
    for w_ in range(step_num_w + 1):
        for h_ in range(step_num_h + 1):
            (i, j) = (int(w_*step*scale_factor),int(h_*step*scale_factor))
            patch_level0 = np.array(slide.read_region((i, j), 0, (patch_size, patch_size)))[:, :, :3]
            patch = np.array(resize(patch_level0, (256, 256, 3)))
            patch = transform_(patch)
            patch = torch.unsqueeze(patch,0).float()
            embed = model0(patch).to(device1)
            patch_pred, _, _ = model1(embed)
            patch_pred = patch_pred.to('cpu').detach().numpy()[0][0]
            
            print('patch_pred is '+str(patch_pred))

            w_start = int(w_*step)//down_size
            w_end = int(w_start+256//down_size)
            h_start = int(h_ * step)//down_size
            h_end = int(h_start+256//down_size)

            count_map[h_start:h_end,w_start:w_end] += single_patch_count
            pred_map[h_start:h_end,w_start:w_end] += patch_pred*single_patch_count

    prediction_map = pred_map/count_map
    return prediction_map
