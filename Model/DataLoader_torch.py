import torch
import torch.utils.data as data_utils
import random
import torchvision.transforms as transforms
import numpy as np
from Preprocess import Morph_Augmentor


##for bag classification
class Classification_Generator(data_utils.Dataset):
    def __init__(self,dataframe,patch_n=200,sigma = 0.05,
                 ColorAugmentation=True,spatial_sample=False,target_w=256, target_h=256,
                 p=0.5,y_col='IDH',KeepPath=False,loadage=False):
        self.y_col=y_col #column name for label
        self.patch_n = patch_n #number of patches generated for each iteration
        self.color=ColorAugmentation
        self.spatial=spatial_sample
        self.target_w=target_w
        self.target_h=target_h
        self.sigma = sigma #control intesity of color augmentation
        self.p=p #probability for rotation etc
        self.KeepPath=KeepPath #whether load path along with image
        self.loadage=loadage #whether load age along with image
        self.df=dataframe #dataframe with meta data, each row is one slide
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
        if self.KeepPath:
            path, input_array = Morph_Augmentor.Augmentation_from_Folder(self.df.loc[index,'Path'],target_w=self.target_w, target_h=self.target_h, 
                                                               p_flip=self.p,p_rotate=self.p, samples=self.patch_n,
                                                                   sigma=self.sigma,ColorAugmentation=self.color,
                                                                   spatial_sample=self.spatial,KeepPath = True)
        else:
            input_array = Morph_Augmentor.Augmentation_from_Folder(self.df.loc[index,'Path'],target_w=self.target_w, target_h=self.target_h, 
                                                               p_flip=self.p,p_rotate=self.p, samples=self.patch_n,
                                                                   sigma=self.sigma,ColorAugmentation=self.color,
                                                                   spatial_sample=self.spatial,KeepPath = False)
            age = self.df.loc[index,'Age']
        input_array /= 255
        label = np.array(self.df.loc[index, self.y_col])
        input=[]
        for i in range(len(input_array)):
            input.append(self.transform(input_array[i,...]))
        input = torch.stack(input,dim=0)
        if self.KeepPath:
            return input, label, path
        elif self.loadage:
            return input, label, age
        else:
            return input, label


