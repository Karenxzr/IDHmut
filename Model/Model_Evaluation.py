import torch
import DataLoader_torch
import pandas as pd
import numpy as np
import os



def tensorlist2array(input):
    output = [i.item() for i in input]
    return np.array(output)

def get_slide_prediction(model0_path, model1_path, dataframe, key_word='Test',y_col='IDH'):
    device = torch.device("cuda:0")
    model0 = torch.load(model0_path, map_location=device)
    model1 = torch.load(model1_path, map_location=device)
    model0.eval()
    model1.eval()

    if key_word=='All':
        df_test = dataframe
    else:
        df_test = dataframe[dataframe['Train_Test'] == key_word].reset_index(drop=True)

    path = list(df_test['Path'])

    test_dset = DataLoader_torch.Classification_Generator(df_test, patch_n=0, p=0,y_col=y_col,
                                               ColorAugmentation=False,spatial_sample=False,
                                                          KeepPath = False)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False,
                                              num_workers=4, pin_memory=True)

    with torch.no_grad():
        y_true = []
        y_pred = []

        for batch_idx, (data, label) in enumerate(test_loader):
            label = label[0].to(device).float()
            data = data.squeeze(0).float()

            embed = []
            for minibatch_ind in range(0, len(data), 20):
                data0 = data[minibatch_ind:min(len(data), minibatch_ind + 20), ...]
                data0 = data0.to(device).float()
                embed0 = model0(data0)
                embed.append(embed0)
            embed = torch.cat(embed, dim=0)

            pred, yhat, _ = model1(embed)

            y_true.append(label)
            y_pred.append(pred)
    return tensorlist2array(y_true), tensorlist2array(y_pred), list(path)


def get_patch_only_prediction(model0_path, model1_path, dataframe, row_slice=-1, key_word='Train',y_col='IDH'):
    
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    model0 = torch.load(model0_path, map_location=device0)
    model1 = torch.load(model1_path, map_location=device1)
    model0.eval()
    model1.eval()

    if row_slice>=0:
        df_test=dataframe.loc[[row_slice]]
    else:
        if key_word=="All":
            df_test = dataframe
        else:
            df_test = dataframe[dataframe['Train_Test'] == key_word].reset_index(drop=True)

    slide_path = list(df_test['Path'])
    test_dset = DataLoader_torch.Classification_Generator(df_test, patch_n=0, p=0, y_col=y_col,
                                                          ColorAugmentation=False, spatial_sample=False,
                                                          KeepPath=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)

    with torch.no_grad():
        patch_list = []
        patch_pred = []
        patch_attention = []
        
    #loop through slides
    for batch_idx, (data, label, path) in enumerate(test_loader):
        patch_pred_=[]
        patch_att_=[]
        #path is a list cooresponding to 'batch' each dimension of data
        label = label[0].to(device0).float()
        data = data.squeeze(0).float()
    
        #loop through patches
        for i in range(len(data)):
            data0=data[i,...]
            data0 = data0.to(device0).float()
            patch_x = model0(data0).to(device1)
            patch_y, _, patch_att = model1(patch_x)
            
            patch_y=patch_y.item()
            patch_att=patch_att.item()
            patch_pred_.append(patch_y)
            patch_att_.append(patch_att)

        #back to slide
        patch_list.append(path)
        patch_pred.append(patch_pred_)
        patch_attention.append(patch_att_)
        
    return patch_attention, patch_list, patch_pred, slide_path
        
        
def get_patch_prediction(model0_path, model1_path, dataframe, row_slice=-1, key_word='Train',y_col='IDH'):
    '''
    :param model0_path: CNN model
    :param model1_path: Attention Model
    :param dataframe: dataframe that would like to make predictions
    :param row_slice: put -1 if all slides would like to make patch prediction, put row number to assign specific slide
    :param key_word: subset of prediction, only useful when row_slice is negative
    :param y_col: label column
    :return:
    '''
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    model0 = torch.load(model0_path, map_location=device0)
    model1 = torch.load(model1_path, map_location=device1)
    model0.eval()
    model1.eval()

    if row_slice>=0:
        df_test=dataframe.loc[[row_slice]]
    else:
        if key_word=="All":
            df_test = dataframe
        else:
            df_test = dataframe[dataframe['Train_Test'] == key_word].reset_index(drop=True)

    slide_path = list(df_test['Path'])

    test_dset = DataLoader_torch.Classification_Generator(df_test, patch_n=0, p=0, y_col=y_col,
                                                          ColorAugmentation=False, spatial_sample=False,
                                                          KeepPath=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True)

    with torch.no_grad():
        y_true = []
        y_pred = []
        y_attention = []
        patch_list = []
        patch_pred = []

        for batch_idx, (data, label, path) in enumerate(test_loader):
            patch_pred_=[]
            #path is a list cooresponding to 'batch' each dimension of data
            label = label[0].to(device0).float()
            data = data.squeeze(0).float()

            embed = []
            for minibatch_ind in range(0, len(data), 20):
                data0 = data[minibatch_ind:min(len(data), minibatch_ind + 20), ...]
                data0 = data0.to(device0).float()
                embed0 = model0(data0)
                embed.append(embed0)
            embed = torch.cat(embed, dim=0).to(device1)
            
            for em_i in range(embed.shape[0]):
                patch_x = embed[em_i,...].unsqueeze(dim=0)
                patch_y,_,_ = model1(patch_x)
                patch_y=patch_y.item()
                patch_pred_.append(patch_y)

            pred, yhat, attention = model1(embed)
            
            attention = list(tensorlist2array(attention[0]))
            pred = pred.item()
            yhat = yhat.item()
            label = label.item()

            y_true.append(label)
            y_pred.append(pred)
            y_attention.append(attention)
            patch_list.append(path)
            patch_pred.append(patch_pred_)
    return y_true, y_pred, y_attention, patch_list, patch_pred, slide_path


def save_model_performance_matrix(Model_Folder, df_path, by='acc',key_word='Test',y_col='IDH'):
    #set key_word to None when evaluate the whole dataframe
    dataframe = pd.read_csv(df_path)
    file_list = os.listdir(Model_Folder)
    if by == 'acc':
        model0_name = [file for file in file_list if 'vlossCNN' in file][0]
        model1_name = [file for file in file_list if 'vlossAT' in file][0]
    elif by == 'loss':
        model0_name = [file for file in file_list if 'vaccCNN' in file][0]
        model1_name = [file for file in file_list if 'vaccAT' in file][0]
    model0_path = os.path.join(Model_Folder, model0_name)
    model1_path = os.path.join(Model_Folder, model1_name)

    y_true, y_pred, _ = get_slide_prediction(model0_path, model1_path, dataframe,key_word=key_word,y_col=y_col)

    acc = accuracy_score(y_true, np.round(y_pred))
    auc = roc_auc_score(y_true, y_pred)

    fconv = open(os.path.join(Model_Folder, 'modelperformanceby' + str(args.by)+'.csv'), 'w')
    fconv.write('vali_metric,metric,value\n')
    fconv.write('{},acc,{}\n'.format(by, acc))
    fconv.write('{},auc,{}\n'.format(by, auc))
    fconv.close()


def save_patch_prediction_to_dataframe(Model_Folder, df_path, by='loss',row_slice=-1,key_word='Test',y_col='IDH',light_mode=False):
    #set row_slice to -1 and key_word to None to evaluate all dataset or key_word to 
    #set row_slice to specific row number to evaluate subset of dataset regardless of key_word
    dataframe = pd.read_csv(df_path)
    file_list = os.listdir(Model_Folder)
    if by == 'acc':
        model0_name = [file for file in file_list if 'vlossCNN' in file][0]
        model1_name = [file for file in file_list if 'vlossAT' in file][0]
    elif by == 'loss':
        model0_name = [file for file in file_list if 'vaccCNN' in file][0]
        model1_name = [file for file in file_list if 'vaccAT' in file][0]
    model0_path = os.path.join(Model_Folder, model0_name)
    model1_path = os.path.join(Model_Folder, model1_name)
    
    if light_mode:
        patch_attention, patch_list, patch_pred, slide_path = get_patch_only_prediction(model0_path, model1_path, dataframe,row_slice=row_slice, key_word=key_word,y_col=y_col)
        df = pd.DataFrame({'attention_weights': patch_attention, 'patch_name': patch_list, 'patch_pred': patch_pred,'slide_path': slide_path})
    else:
        y_true, y_pred, y_attention, patch_list, patch_pred, slide_path = get_patch_prediction(model0_path, model1_path, dataframe,row_slice=row_slice, key_word=key_word,y_col=y_col)
        df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred, 'attention_weights': y_attention, 'patch_name': patch_list, 'patch_pred': patch_pred,'slide_path': slide_path})
    return df
