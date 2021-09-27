from Model import DataLoader_torch
from Model import Models 
from collections import OrderedDict
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.optim as optim
import datetime
import sklearn
import json
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='ATTENTION DENSENET MODEL')
parser.add_argument('--result_dir', type=str, default='/nfs03/data/TCGA_Brain/Results/TCGA2.5Classifier/')
parser.add_argument('--df_path', type=str, default='/home/gid-xuz/csv/Image_IDH_TCGA_collapse.csv')
parser.add_argument('--gpu', type=str, default='0,1,2,3')
parser.add_argument('--color',action='store_true')
parser.add_argument('--color_off',dest='color',action='store_false')
parser.add_argument('--A', type=int, default=16,help='node number for attention')
parser.add_argument('--balance', default=0.5, type=float,
                    help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--patch_n', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--y_col',type=str,default='IDH')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--spatial_sample',  action='store_true')
parser.add_argument('--spatial_sample_off', dest='spatial_sample', action='store_false')
parser.add_argument('--no_age',  action='store_true')
parser.add_argument('--add_age', dest='no_age', action='store_false')#concatenate age on embedding
parser.add_argument('--notes',type=str,default='')
parser.add_argument('--balance_training',action='store_true')
parser.add_argument('--balance_training_off',action='store_false',dest='balance_training')
parser.add_argument('--freeze',type=int,default=0,help='layer number to freeze')
parser.add_argument('--CNN',type=str,default='resnet', help='choose from resnet/densenet')
parser.add_argument('--pretrain',type=str, default='imagenet',help = 'put pretrained model path here')
parser.add_argument('--use_scheduler',action='store_true')
parser.add_argument('--use_scheduler_off',dest='use_scheduler',action='store_false')
parser.add_argument('--freeze_batchnorm',action='store_true')
parser.add_argument('--freeze_batchnorm_off',action='store_false',dest='freeze_batchnorm')
parser.add_argument('--freeze_CNN_model',action='store_true')
parser.add_argument('--freeze_CNN_model_off',dest = 'freeze_CNN_model',action='store_false')
parser.add_argument('--pooling', type=str, default='attention', help='aggregation method of model, choose from mean, attention, max')
parser.add_argument('--loader_mode',type=str,default='classification', help='set to pyramid for zoom in, set to weights '
                                                                            'for weigted sampling')

def main():
    #-------Environment
    global args
    args = parser.parse_args()
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    result_path = os.path.join(args.result_dir, str(datetime.datetime.now())[0:19])
    print('the result dir is: ', result_path)
    os.makedirs(result_path)
    if args.pretrain=='imagenet':
        model_name = str(args.CNN) + 'Freeze'+str(args.freeze) + 'Emb' + str(args.pooling) + 'lr' + str(args.lr)[2:] + 'epoch'+str(args.n_epoch)+ \
                  'Opt'+str(args.optimizer)+'patch' + str(args.patch_n) + 'Balweight'+str(args.balance) + 'Balsample'+str(args.balance_training)+\
                'imagenet_'+str(args.notes)
    else:
        model_name = 'Pretrained_' + str(args.CNN) + 'Freeze'+str(args.freeze) + 'Emb' + str(args.pooling) + 'lr' + str(args.lr)[2:] + 'epoch'+str(args.n_epoch)+ \
                  'Opt'+str(args.optimizer)+'patch' + str(args.patch_n) + 'Balweight'+str(args.balance) + 'Balsample'+str(args.balance_training)+\
                '_'+str(args.notes)
      
    fconv = open(os.path.join(result_path, model_name)+'.csv', 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()

    #--------CNN
    print('Building Model and Optimizer')
    if args.CNN=='resnet':
        CNN_model = Models.ResNet18(Freeze_Num=args.freeze)
        if args.no_age:
            attention_model = Models.attention(D=args.A,L=1000)
        else:
            attention_model = Models.attention(D=args.A,L=1001)
    elif args.CNN=='densenet':
        CNN_model = Models.DenseNet(Freeze_Num=args.freeze)
        if args.no_age:
            attention_model = Models.attention(D=args.A,L=1024)
        else:
            attention_model = Models.attention(D=args.A,L=1025)
    else:
        raise Exception('choose from resnet and densenet')
    
    if args.pretrain!='imagenet':
        print('Using custom pretrained weights')
        file_list = os.listdir(args.pretrain)
        model0_name = [file for file in file_list if 'vlossCNN' in file][0]
        model1_name = [file for file in file_list if 'vlossAT' in file][0]
        model0_path = os.path.join(args.pretrain, model0_name)
        model1_path = os.path.join(args.pretrain, model1_name)
        
        state_dict = torch.load(model0_path).state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v   
        CNN_model.load_state_dict(new_state_dict)
        if args.no_age:
            attention_model.load_state_dict(torch.load(model1_path).state_dict())
    
    CNN_model = torch.nn.DataParallel(CNN_model,output_device=1)
    CNN_model.to(device0)
    attention_model.to(device1)
    
    # freeze CNN model if needed 
    if args.freeze_CNN_model: 
        CNN_model.eval()
        for param in CNN_model.parameters():
            param.requires_grad = False
    
    #optimizer
    if args.freeze_CNN_model:
        param_list = [{'params': attention_model.parameters()}]
    else: 
        param_list = [{'params': CNN_model.parameters()},
                      {'params': attention_model.parameters()}]
    optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    if args.optimizer=='Adam':
        optimizer = optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.n_epoch/3, args.n_epoch/3*2], gamma=0.1)
    
    #--------INPUT
    df=pd.read_csv(args.df_path)
    train_index = df[df['Train_Test']=='Train'].index
    val_index = df[df['Train_Test']=='Validation'].index
    df_train=df[df['Train_Test']=='Train'].reset_index(drop=True)
    df_vali=df[df['Train_Test']=='Validation'].reset_index(drop=True)
    
    if args.no_age:
        loadage=False
    else:
        loadage=True
        
    train_dset = DataLoader_torch.Classification_Generator(df_train, y_col=args.y_col,patch_n=args.patch_n,p=0.5,
                                            ColorAugmentation=args.color,spatial_sample=False,loadage=loadage)
    vali_dset = DataLoader_torch.Classification_Generator(df_vali, patch_n=args.patch_n,y_col=args.y_col,p=0,
                                            ColorAugmentation=False,spatial_sample=False,loadage=loadage)
        
    if args.loader_mode == 'weights':
        df_weight = pd.read_csv(args.dataframe_weight_path)
        if args.subfolderdict is not None:
            subfolderdict_path=args.subfolderdict
            with open(subfolderdict_path) as f:
                d = json.load(f)
        else:
            d = None
            
        train_dset = DataLoader_torch.weighted_sample_generator(dataframe=df_train, dataframe_weights=df_weight,subfolderdict=d,
                                                                patch_n=args.patch_n,gamma=args.gamma,
                                                                ColorAugmentation=True,
                                                                spatial_sample=args.spatial_sample)
        vali_dset = DataLoader_torch.weighted_sample_generator(dataframe=df_vali, dataframe_weights=df_weight,subfolderdict=d,
                                                                patch_n=200,gamma=args.gamma,
                                                                ColorAugmentation=False,
                                                                spatial_sample=args.spatial_sample)
   

    train_loader = torch.utils.data.DataLoader(
        train_dset,batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        vali_dset,batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    ###sampler
    if args.balance_training:
        class_sample_count = np.array([len(np.where(df_train[args.y_col] == t)[0]) for t in np.unique(df_train[args.y_col])])
        weight = 1. / class_sample_count
        weight = pd.DataFrame(weight.flatten(), index=list(np.unique(df_train[args.y_col])))
        sample_weights=np.array([weight.loc[t] for t in df_train[args.y_col]])
        sample_weights = torch.from_numpy(sample_weights)
        sample_weights = sample_weights.double()
        sampler = torch.utils.data.WeightedRandomSampler(weights = sample_weights,  num_samples = int(class_sample_count.min()*2),replacement=False)
    
        train_loader = torch.utils.data.DataLoader(
        train_dset,batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,sampler=sampler)
        print('Balance Training Weighted Sampler Used')

    #--------Loop Epoch
    print('Start Training:')
    lowest_val_loss = 100.
    highest_val_acc = 0.
    
    for epoch in range(1,args.n_epoch+1):
            
        loss,acc= train(epoch,model=[CNN_model,attention_model],
                        train_loader=train_loader,optimizer=optimizer,
                        device0=device0,device1=device1,
                        freeze_bn=args.freeze_batchnorm,freeze_CNN_model=args.freeze_CNN_model,
                       withage=loadage)
        vloss, vacc, vauc = validation(epoch,model=[CNN_model,attention_model],
                                       val_loader=val_loader,  
                                   device0=device0,device1=device1,
                                      withage=loadage)
        print('finished one validation')
        #save log
        fconv = open(os.path.join(result_path, model_name)+'.csv', 'a')
        fconv.write('{},loss,{}\n'.format(epoch, loss))
        fconv.write('{},acc,{}\n'.format(epoch, acc))
        fconv.write('{},val_loss,{}\n'.format(epoch, vloss))
        fconv.write('{},val_acc,{}\n'.format(epoch, vacc))
        fconv.write('{},val_auc,{}\n'.format(epoch, vauc))
        fconv.close()
        
        if args.use_scheduler:
            scheduler.step()
        
        #save best model
        if vloss < lowest_val_loss:
            lowest_val_loss=vloss
            torch.save(CNN_model, os.path.join(result_path, model_name) + '_vlossCNN.pt')
            torch.save(attention_model, os.path.join(result_path, model_name) + '_vlossAT.pt')
        if vacc > highest_val_acc:
            highest_val_acc=vacc
            torch.save(CNN_model, os.path.join(result_path, model_name) + '_vaccCNN.pt')
            torch.save(attention_model, os.path.join(result_path, model_name) + '_vaccAT.pt')

def train(epoch, model, train_loader, optimizer, device0, device1, freeze_bn, freeze_CNN_model, withage=False):
    
    print('Epoch' + str(epoch) + 'starts:')
    model0, model1 = model
    if not freeze_CNN_model:
        model0.train()
    model1.train()
    
    if freeze_bn:
        freeze_batchnorm(model0)
                
        
    train_loss = 0.
    train_acc = 0.
    if withage:
        for batch_idx, (data, label, age) in enumerate(train_loader):
            age = torch.from_numpy(np.array([[age]])).to(device1)
        
        # reset gradients
            optimizer.zero_grad()
        #prepare data
            bag_label = label[0].to(device1).float()
            data = data.squeeze(0).float()
        #accumulate embedding
            embed=[]
            for minibatch_ind in range(0,len(data),20):
                data0 = data[minibatch_ind:min(len(data),minibatch_ind+20),...]
                data0 = data0.to(device0)
                embed0 = model0(data0)
                embed.append(embed0)
            embed = torch.cat(embed,dim=0)
            age = torch.tile(age, (embed.size()[0], 1)).float()
            embed = torch.cat((embed,age),dim=1)
        
        # calculate loss and metrics
            ypred, yhat, _ = model1(embed,  pooling=args.pooling)
            loss = Models.Loss(y_pred=ypred,y_true=bag_label,balance=args.balance).to(device1)
            train_loss += loss.item()
            acc= Models.ACC(yhat,bag_label)
            train_acc += acc
        # backward pass
            loss.backward()
        # step 
            optimizer.step()
            if batch_idx % 20 == 0:
                print('Train Epoch: {}/{}'.format(batch_idx, len(train_loader.dataset)))
      
      
      
    else:
        for batch_idx, (data, label) in enumerate(train_loader):
    
        # reset gradients
            optimizer.zero_grad()
        #prepare data
            bag_label = label[0].to(device1).float()
            data = data.squeeze(0).float()
        #accumulate embedding
            embed=[]
            for minibatch_ind in range(0,len(data),20):
                data0 = data[minibatch_ind:min(len(data),minibatch_ind+20),...]
                data0 = data0.to(device0)
                embed0 = model0(data0)
                embed.append(embed0)
            embed = torch.cat(embed,dim=0)
        # calculate loss and metrics
            ypred, yhat, _ = model1(embed, temp=temp, pooling=args.pooling)
            loss = Models.Loss(y_pred=ypred,y_true=bag_label,balance=args.balance).to(device1)
            train_loss += loss.item()
            acc= Models.ACC(yhat,bag_label)
            train_acc += acc
        # backward pass
            loss.backward()
        # step 
            optimizer.step()
            if batch_idx % 20 == 0:
                print('Train Epoch: {}/{}'.format(batch_idx, len(train_loader.dataset)))
    
    
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    print('Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}'.format(epoch, train_loss, train_acc))
    return train_loss, train_acc

def validation(epoch, model, val_loader, device0, device1,withage=False):
    # switch model to evaluation mode
    model0, model1 = model
    model0.eval()
    model1.eval()
    vali_loss = 0.
    vali_acc = 0.
    
    auc_ytrue=[]
    auc_ypred=[]
    with torch.no_grad():
        if withage:
            for batch_idx, (data, label, age) in enumerate(val_loader):
                age = torch.from_numpy(np.array([[age]])).to(device1)
                bag_label = label[0].to(device1).float()
                auc_ytrue.append(label[0].float())
            
                data = data.squeeze(0).float()
                #accumulate embedding
                embed = []
                for minibatch_ind in range(0,len(data),30):
                    data0 = data[minibatch_ind:min(len(data),minibatch_ind+30),...]
                    data0 = data0.to(device0)
                    embed0 = model0(data0) # extract patch level feature
                    embed.append(embed0)
                embed = torch.cat(embed,dim=0)
                age = torch.tile(age, (embed.size()[0], 1)).float()
                embed = torch.cat((embed,age),dim=1)
            # calculate loss and metrics
                ypred, yhat, _ = model1(embed, temp=temp, pooling=args.pooling)
                loss = Models.Loss(y_pred=ypred,y_true=bag_label,balance=args.balance).to(device1)
                vali_loss += loss.item()
                acc= Models.ACC(y_pred=yhat, y_true=bag_label)
                vali_acc += acc
                auc_ypred.append(ypred)
        else:
            for batch_idx, (data, label) in enumerate(val_loader):
                bag_label = label[0].to(device1).float()
                auc_ytrue.append(label[0].float())
            
                data = data.squeeze(0).float()
                #accumulate embedding
                embed = []
                for minibatch_ind in range(0,len(data),30):
                    data0 = data[minibatch_ind:min(len(data),minibatch_ind+30),...]
                    data0 = data0.to(device0)
                    embed0 = model0(data0) # extract patch level feature
                    embed.append(embed0)
                embed = torch.cat(embed,dim=0)
            # calculate loss and metrics
                ypred, yhat, _ = model1(embed, temp=temp, pooling=args.pooling)
                loss = Models.Loss(y_pred=ypred,y_true=bag_label,balance=args.balance).to(device1)
                vali_loss += loss.item()
                acc= Models.ACC(y_pred=yhat, y_true=bag_label)
                vali_acc += acc
                auc_ypred.append(ypred)
                
        # calculate loss and error for epoch
        vali_loss /= len(val_loader)
        vali_acc /= len(val_loader)
        vali_auc = sklearn.metrics.roc_auc_score(y_true=auc_ytrue,y_score=auc_ypred)
        print('Epoch: {}, Validation Loss: {:.4f}, Validation accuracy: {:.4f}, Validation auc: {:.4f}'.format(epoch, vali_loss, vali_acc, vali_auc))
        return vali_loss, vali_acc, vali_auc

#freeze batchnorm
def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
            
            
if __name__ == "__main__":
    main()
