from Model import visualization as v
import argparse
import os
import numpy as np
from Preprocess import filter
import matplotlib.pyplot as plt
from skimage.transform import resize
import openslide


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
#to save prediction heatmap
parser.add_argument('--heatmaps',action='store_true')
parser.add_argument('--heatmaps_off',dest = 'heatmaps',action='store_false')
parser.add_argument('--cmap',type=str,default='jet')
parser.add_argument('--heatmap_name',type=str,default='Heatmap.png')



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
        
    if args.heatmaps:
        prediction_map = v.patch_prediction_pixel(args.svs_path, args.model_folder_path, args.by, args.magnification, args.step_pct, args.down_size)
        filter_map = v.patch_filter_pixel(args.svs_path, args.model_folder_path, args.by, args.magnification, args.step_pct, args.down_size)
        target_path = os.path.join(args.target_path, 'Prediction_Map.npy')
        np.save(target_path,prediction_map)
        target_path = os.path.join(args.target_path, 'Filter_Map.npy')
        np.save(target_path,filter_map)
        #read slides
        slide = openslide.open_slide(args.svs_path)
        slide = np.array(slide.read_region((0,0), 2, slide.level_dimensions[2]))[:, :, :3]
        target_shape = (slide.shape[0],slide.shape[1])
        #same size
        prediction_map = resize(prediction_map,target_shape,anti_aliasing=True)
        filter_map = resize(filter_map,target_shape,anti_aliasing=True)
        #plot
        fig = plt.figure(figsize=(20,20))
        fig.add_subplot(111)
        plt.axis('off')
        plt.imshow(slide,cmap='gray')
        plt.imshow(prediction_map,cmap=args.cmap,alpha=filter_map*0.3)
        fig.tight_layout()
        heatmap_path = os.path.join(args.target_path, args.heatmap_name)
        plt.savefig(heatmap_path,dpi=600)
    else:
        if args.type=='prediction':
            map_array = v.patch_prediction_pixel(args.svs_path, args.model_folder_path, args.by, args.magnification, args.step_pct, args.down_size)
            np.save(target_path,map_array)
        elif args.type=='filter':
            map_array = v.patch_filter_pixel(args.svs_path, args.model_folder_path, args.by, args.magnification, args.step_pct, args.down_size)
            np.save(target_path,map_array)
        


if __name__ == '__main__':
    main()
