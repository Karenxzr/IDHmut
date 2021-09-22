import torch
import torch.utils.data as data_utils
import Morph_Augmentor
import random
import torchvision.transforms as transforms
import numpy as np


##for contrastive learning pretraining--bag pretrain
class Generator(data_utils.Dataset):
    def __init__(self,path_list, patch_n=20,sigma = 0.1,spatial_sample=False):
        self.path_list=path_list
        self.patch_n = patch_n
        self.spatial=spatial_sample
        self.sigma = sigma
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
        p1_0 = random.uniform(0,1)
        p2_0 = random.uniform(0,1)
        p3_0 = random.uniform(0,1)
        p1_1 = random.uniform(0,1)
        p2_1 = random.uniform(0,1)
        p3_1 = random.uniform(0,1)
        input_array0 = Morph_Augmentor.augmentation_from_folder_heavy(self.path_list[index],target_w=256, target_h=256, p_flip=p1_0,
                                                                   p_rotate=p2_0, samples=self.patch_n,sigma=self.sigma,p_blur=p3_0,
                                                               ColorAugmentation=True,spatial_sample=self.spatial)
        input_array1 = Morph_Augmentor.augmentation_from_folder_heavy(self.path_list[index],target_w=256, target_h=256, p_flip=p1_1,
                                                                   p_rotate=p2_1, samples=self.patch_n,sigma=self.sigma,p_blur=p3_1,
                                                               ColorAugmentation=True,spatial_sample=self.spatial)
  
        input_array0=input_array0/255
        input_array1=input_array1/255
        #shape: patch_n, 3, 256, 256, tensor
        return trans(input_array0), trans(input_array1)

##for contrastive learning pretraining---patch pretrain    
class Generator_patch(data_utils.Dataset):
    def __init__(self,path_list,sigma = 0.1,p_blur=0.3,target_w=192,target_h =192 ,spatial_sample=True):
        self.path_list=path_list
        self.spatial=spatial_sample
        self.sigma = sigma
        self.p_blur = p_blur
        self.target_w=target_w
        self.target_h=target_h
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    
    def __len__(self):
        return len(self.path_list)
   
    
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
        p1_0 = random.uniform(0,1)
        p2_0 = random.uniform(0,1)
        
        p1_1 = random.uniform(0,1)
        p2_1 = random.uniform(0,1)
        
        input_array0 = Morph_Augmentor.patch_augmentation(self.path_list[index],target_w=self.target_w, target_h=self.target_h, p_flip=p1_0,
                                                                   p_rotate=p2_0, sigma=self.sigma,p_blur=self.p_blur,
                                                               ColorAugmentation=True,spatial_sample=self.spatial)
        input_array1 = Morph_Augmentor.patch_augmentation(self.path_list[index],target_w=self.target_w, target_h=self.target_h, p_flip=p1_1,
                                                                   p_rotate=p2_1, sigma=self.sigma,p_blur=self.p_blur,
                                                               ColorAugmentation=True,spatial_sample=self.spatial)
  
        input_array0=input_array0/255
        input_array1=input_array1/255
        input_array0= self.transform(input_array0)
        input_array1=self.transform(input_array1)
        #shape: patch_n, 3, 256, 256, tensor
        return input_array0, input_array1    
    
    
def trans(before):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    after=[]
    for i in range(len(before)):
        after.append(transform(before[i,...]))
    return torch.stack(after,0)



##for bag classification--single resolution
class Classification_Generator(data_utils.Dataset):
    def __init__(self,dataframe,patch_n=200,positive_n=1,negative_n=1,sigma = 0.05,
                 ColorAugmentation=True,spatial_sample=False,p=0.5,y_col='IDH',KeepPath=False,loadage=False):
        self.y_col=y_col
        self.positive_n = positive_n
        self.negative_n = negative_n
        self.patch_n = patch_n
        self.color=ColorAugmentation
        self.spatial=spatial_sample
        self.sigma = sigma
        self.p=p
        self.KeepPath=KeepPath
        self.loadage=loadage
        dataframe['times'] = np.where(dataframe[self.y_col] == 1, self.positive_n, self.negative_n)
        df=dataframe.loc[dataframe.index.repeat(dataframe.times)].reset_index(drop=True)
        self.df=df
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
            path, input_array = Morph_Augmentor.Augmentation_from_Folder(self.df.loc[index,'Path'],target_w=256, target_h=256, 
                                                               p_flip=self.p,p_rotate=self.p, samples=self.patch_n,
                                                                   sigma=self.sigma,ColorAugmentation=self.color,
                                                                   spatial_sample=self.spatial,KeepPath = True)
        else:
            input_array = Morph_Augmentor.Augmentation_from_Folder(self.df.loc[index,'Path'],target_w=256, target_h=256, 
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

##for patch pyramid
##TODO: 
class pyramid_generator(data_utils.Dataset):
    def __init__(self,dataframe,attention = None, patch_n=200,mode = 'low',top_n = 10,sigma = 0.05,
                 ColorAugmentation=True,spatial_sample=False,p=0.5,y_col='IDH',double_output=False,gpu_num=4):
        #put attention weights into dataframe 
        #attention is a numpy list that has the same order with the dataframe if not none
        #double_output set to true if random sample 'patch_n' patches, output low and high reso patches
        self.y_col=y_col
        self.df=dataframe
        self.n = self.df.shape[0]
        self.gpu_num=gpu_num
        if attention is None:
            self.attention = list(np.repeat(None,self.n))
        else:
            self.attention = attention
            
        self.patch_n = patch_n
        self.mode = mode
        self.double_output = double_output
        self.top_n = top_n
        self.color=ColorAugmentation
        self.spatial=spatial_sample
        self.sigma = sigma
        self.p=p
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return self.n
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
        input_array = Morph_Augmentor.pyramid_augmentation(FolderPath = self.df.loc[index,'Path'],target_w=256, target_h=256, 
                                                               attention = self.attention[index], mode = self.mode, top_n = self.top_n,
                                                               p_flip=self.p,p_rotate=self.p, samples=self.patch_n,gpu_num = self.gpu_num,
                                                                   sigma=self.sigma,ColorAugmentation=self.color,spatial_sample=self.spatial)
        label = np.array(self.df.loc[index, self.y_col])
        if self.attention[0] is None and self.mode =='low':
            input_low, input_high, high_index = input_array
            input_low/=255
            input_high/=255
            input_low = trans(input_low)
            input_high = trans(input_high)
            high_index = np.array(high_index)
            if self.double_output:
                return input_low,input_high, high_index, label
            else:
                return input_high, label
        else:
            input_array = input_array.astype(float)
            input_array /= 255
            input_array = trans(input_array)
            return input_array, label

# For sample by attention weights
class weighted_sample_generator(data_utils.Dataset):
    def __init__(self,dataframe,dataframe_weights,subfolderdict=None,
                 patch_n=200,sigma = 0.05,gamma=1,
                 ColorAugmentation=True,spatial_sample=False,
                 key = 'slide_identifier',
                 y_col='IDH'):
        self.y_col=y_col
        self.key_name=key
        self.sigma = sigma
        self.gamma = gamma
        self.df = dataframe
        self.df_weight = dataframe_weights
        self.subfolderdict=subfolderdict
        self.patch_n = patch_n
        self.color=ColorAugmentation
        self.spatial=spatial_sample
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()

        
        if self.patch_n==0:
            input_array = Morph_Augmentor.Augmentation_from_Folder(self.df.loc[index,'Path'],target_w=256, target_h=256, 
                                                               p_flip=0,p_rotate=0, samples=0,
                                                                   sigma=self.sigma,ColorAugmentation=self.color,
                                                                   spatial_sample=self.spatial,KeepPath = False)
        else:
            df_weight_single = self.df_weight[self.df_weight[self.key_name]==self.df.loc[index,self.key_name]].reset_index(drop=True)
            input_array = Morph_Augmentor.augmentation_weighted_sampling(dataframe=df_weight_single,subfolderdict=self.subfolderdict,
                                                                     sample_size=self.patch_n,
                                                                     gamma=self.gamma, sigma=self.sigma,
                                                                     coloraugmentation=self.color)

        input_array = input_array/255
        label = np.array(self.df.loc[index, self.y_col])
        input=[]
        for i in range(len(input_array)):
            input.append(self.transform(input_array[i,...]))
        input = torch.stack(input,dim=0)
        return input, label
    
    
####TODO:    
#load 4d numpy
class np_4d_generator(data_utils.Dataset):
    def __init__(self,dataframe,ColorAugmentation=True,spatial_sample=False,
                 y_col='IDH'):
        self.y_col=y_col
        self.df = dataframe
        self.color=ColorAugmentation
        self.spatial=spatial_sample
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()

        
        input_array = Morph_Augmentor.Augmentation_from_Folder(self.df.loc[index,'Path'],target_w=256, target_h=256, 
                                                               p_flip=0,p_rotate=0, samples=0,
                                                                   sigma=self.sigma,ColorAugmentation=self.color,
                                                                   spatial_sample=self.spatial,KeepPath = False)

        input_array = input_array/255
        label = np.array(self.df.loc[index, self.y_col])
        input=[]
        for i in range(len(input_array)):
            input.append(self.transform(input_array[i,...]))
        input = torch.stack(input,dim=0)
        return input, label
