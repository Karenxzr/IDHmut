import torch
from Model import DataLoader_torch
import pandas as pd
import argparse
import re
import numpy as np
import os
from Model import Model_Evaluation as ME
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Evaluate Model')
#required to set
parser.add_argument('--gpu', type=str, default='0,1,2,3')
parser.add_argument('--df_path', type=str)
parser.add_argument('--Model_Folder', type=str)
parser.add_argument('--action',type=str,default='summary')#choose from summary: save model performance as csv/ patch: save patch prediction as csv
parser.add_argument('--by', type=str, default='loss')
parser.add_argument('--notes',type=str,default='')

#set if action is summary
#don't forget to set key_word
parser.add_argument('--repeat',type=int,default=1)#repeat how many times of evaluation
parser.add_argument('--patch_n',type=int,default=0)#how many patches each evaluation, 0: all patches

#set if action is patch
parser.add_argument('--row_slice',type=int,default=-1)#set to -1 for all rows otherwise will evaluate specific row
parser.add_argument('--key_word',type=str,default='Test')#set to 'All' if evaluating the whole dataframe
parser.add_argument('--light_mode', action="store_true")#set light mode when oom to make patch only predictions
parser.add_argument('--light_mode_off',dest='light_mode',action='store_false')

#age
parser.add_argument('--no_age',  action='store_true')
parser.add_argument('--add_age', dest='no_age', action='store_false')#concatenate age on embedding

# set true if for two forward if oom
parser.add_argument('--two_forward', action="store_true")
parser.add_argument('--two_forward_off',dest='two_forward',action='store_false')

def tensorlist2array(input):
    output = [i.item() for i in input]
    return np.array(output)

def Model_Evaluation_Age(model0_path, model1_path, dataframe, keyword = 'Test',patch_n=0, two_forward=True):
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    
    model0 = torch.load(model0_path, map_location=device0)
    model1 = torch.load(model1_path, map_location=device1)
    model0.eval()
    model1.eval()
    
    if keyword == 'All': 
        df_test = dataframe
    else:
        df_test = dataframe[dataframe['Train_Test'] == keyword].reset_index(drop=True)
        
    test_dset = DataLoader_torch.Classification_Generator(df_test, patch_n=patch_n, p=0,
                                               ColorAugmentation=False,loadage=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    
    y_true = []
    y_pred = []
    if not two_forward:
        with torch.no_grad():
            for batch_idx, (data, label,age) in enumerate(test_loader):
                label = label[0].to(device0).float()
                data = data.squeeze(0).float()
                age=torch.from_numpy(np.array([[age]])).to(device1)

                embed = []
                for minibatch_ind in range(0, len(data), 30):
                    data0 = data[minibatch_ind:min(len(data), minibatch_ind + 30), ...]
                    data0 = data0.to(device0).float()
                    embed0 = model0(data0)
                    embed.append(embed0)
                embed = torch.cat(embed, dim=0).to(device1)
                age = torch.tile(age, (embed.size()[0], 1)).float()
                embed = torch.cat((embed,age),dim=1)

                pred, yhat, _ = model1(embed)
                y_true.append(label)
                y_pred.append(pred)
                
    else:
        with torch.no_grad():
            for batch_idx, (data, label, age) in enumerate(test_loader):
                bag_label = label[0].to(device1).float()
                data = data.squeeze(0).float()
                age=torch.from_numpy(np.array([[age]])).to(device1)
                # first forward pass, get normalized attention weight. 
                weight = None
                for minibatch_ind in range(0,len(data),30):
                    data0 = data[minibatch_ind:min(len(data),minibatch_ind+30),...]
                    data0 = data0.to(device0)
                    embed0 = model0(data0) # extract patch level feature
                    age = torch.tile(age, (embed0.size()[0], 1)).float()
                    embed0 = torch.cat((embed0,age),dim=1)
                    # get attention
                    if weight is None:
                        weight = model1(embed0, inf_mode='weight')
                    else:
                        temp_weight = model1(embed0, inf_mode='weight')
                        weight = torch.cat((weight, temp_weight), 1)
                # normalize weights
                weight = F.softmax(weight, dim=1)
                
                # get weighted sum
                embed = None
                for minibatch_ind in range(0,len(data),30):
                    data0 = data[minibatch_ind:min(len(data),minibatch_ind+30),...]
                    weight0 = weight[:, minibatch_ind:min(len(data),minibatch_ind+30)]
                    data0 = data0.to(device0)
                    embed0 = model0(data0) # extract patch level feature
                    age = torch.tile(age, (embed0.size()[0], 1)).float()
                    embed0 = torch.cat((embed0,age),dim=1)
                    # get attention
                    if embed is None:
                        embed = torch.mm(weight0, embed0)
                    else:
                        embed += torch.mm(weight0, embed0)
                # get final prediction
                ypred, yhat = model1(embed, inf_mode='classify')        
                y_true.append(label)
                y_pred.append(ypred)
                
    return tensorlist2array(y_true), tensorlist2array(y_pred)

def Model_Evaluation_(model0_path, model1_path, dataframe, keyword = 'Test',patch_n=0, two_forward=True):
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    
    model0 = torch.load(model0_path, map_location=device0)
    model1 = torch.load(model1_path, map_location=device1)
    model0.eval()
    model1.eval()
    
    if keyword == 'All': 
        df_test = dataframe
    else:
        df_test = dataframe[dataframe['Train_Test'] == keyword].reset_index(drop=True)
        
    test_dset = DataLoader_torch.Classification_Generator(df_test, patch_n=patch_n, p=0,
                                               ColorAugmentation=False)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)
    
    y_true = []
    y_pred = []
    if not two_forward:
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                label = label[0].to(device0).float()
                data = data.squeeze(0).float()

                embed = []
                for minibatch_ind in range(0, len(data), 30):
                    data0 = data[minibatch_ind:min(len(data), minibatch_ind + 30), ...]
                    data0 = data0.to(device0).float()
                    embed0 = model0(data0)
                    embed.append(embed0)
                embed = torch.cat(embed, dim=0).to(device1)

                pred, yhat, _ = model1(embed)
                y_true.append(label)
                y_pred.append(pred)
                
    else:
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                bag_label = label[0].to(device1).float()
                data = data.squeeze(0).float()
                # first forward pass, get normalized attention weight. 
                weight = None
                for minibatch_ind in range(0,len(data),30):
                    data0 = data[minibatch_ind:min(len(data),minibatch_ind+30),...]
                    data0 = data0.to(device0)
                    embed0 = model0(data0) # extract patch level feature
                    # get attention
                    if weight is None:
                        weight = model1(embed0, inf_mode='weight')
                    else:
                        temp_weight = model1(embed0, inf_mode='weight')
                        weight = torch.cat((weight, temp_weight), 1)
                # normalize weights
                weight = F.softmax(weight, dim=1)
                
                # get weighted sum
                embed = None
                for minibatch_ind in range(0,len(data),30):
                    data0 = data[minibatch_ind:min(len(data),minibatch_ind+30),...]
                    weight0 = weight[:, minibatch_ind:min(len(data),minibatch_ind+30)]
                    data0 = data0.to(device0)
                    embed0 = model0(data0) # extract patch level feature
                    # get attention
                    if embed is None:
                        embed = torch.mm(weight0, embed0)
                    else:
                        embed += torch.mm(weight0, embed0)
                # get final prediction
                ypred, yhat = model1(embed, inf_mode='classify')        
                y_true.append(label)
                y_pred.append(ypred)
                
    return tensorlist2array(y_true), tensorlist2array(y_pred)


def Model_Evaluation(Model_Folder, df_path, by='loss',keyword = 'Test',patch_n=0, two_forward=True,repeat=1,withage=False):
    dataframe = pd.read_csv(df_path)
    file_list = os.listdir(Model_Folder)
    if by == 'loss':
        model0_name = [file for file in file_list if 'vlossCNN' in file][0]
        model1_name = [file for file in file_list if 'vlossAT' in file][0]
    elif by == 'acc':
        model0_name = [file for file in file_list if 'vaccCNN' in file][0]
        model1_name = [file for file in file_list if 'vaccAT' in file][0]
    model0_path = os.path.join(Model_Folder, model0_name)
    model1_path = os.path.join(Model_Folder, model1_name)

    
    fconv = open(os.path.join(Model_Folder, 'modelperformanceby' + str(args.by)+str(args.notes)+'.csv'), 'w')
    fconv.write('n,metric,value\n')
    fconv.close()
    
    pred_df=[]
    for i in range(repeat):
        if withage:
            y_true, y_pred = Model_Evaluation_Age(model0_path, model1_path, dataframe,keyword=keyword,patch_n=patch_n, two_forward=two_forward)
        else:
            y_true, y_pred = Model_Evaluation_(model0_path, model1_path, dataframe,keyword=keyword,patch_n=patch_n, two_forward=two_forward)
            
        Prediction_df = pd.DataFrame({'y_true':y_true,'y_pred':y_pred})

        acc = accuracy_score(y_true, np.round(y_pred))
        auc = roc_auc_score(y_true, y_pred)
        print('acc is'+str(acc))
        print('auc is'+str(auc))
        fconv = open(os.path.join(Model_Folder, 'modelperformanceby' + str(args.by)+str(args.notes)+'.csv'), 'a')
        fconv.write('{},acc,{}\n'.format(i, acc))
        fconv.write('{},auc,{}\n'.format(i, auc))
        fconv.close()
        
        pred_df.append(Prediction_df)
    
    pred_df=pd.concat(pred_df,axis=0)
    return pred_df


def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    if args.no_age:
        withage=False
    else:
        withage=True
    if args.action=='summary':
        df = Model_Evaluation(args.Model_Folder, args.df_path, by=args.by, keyword = args.key_word, patch_n=args.patch_n, 
                              two_forward=args.two_forward,repeat=args.repeat,withage=withage)
        df.to_csv(os.path.join(args.Model_Folder, 'slideprediction' + str(args.by)+str(args.notes)+'.csv'))
    elif args.action=='patch':
        df = ME.save_patch_prediction_to_dataframe(args.Model_Folder, args.df_path,by=args.by,row_slice = args.row_slice,key_word=args.key_word,y_col='IDH',light_mode=args.light_mode)
        df.to_csv(os.path.join(args.Model_Folder, 'patchprediction' + str(args.by)+str(args.notes)+'.csv'))
        df.to_pickle(os.path.join(args.Model_Folder, 'patchprediction' + str(args.by)+str(args.notes)+'.pkl'))


if __name__ == "__main__":
    main()
