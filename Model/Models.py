import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class DenseNet(nn.Module):
    def __init__(self,Freeze_Num=0):
        super(DenseNet, self).__init__()
        self.Freeze_Num=Freeze_Num
        self.extractor = self.densenet_extractor()
    def densenet_extractor(self):
        densenet = torchvision.models.densenet121(pretrained=True)
        densenet.classifier = nn.Identity()
        ct = 0
        for child in densenet.children():
            for child1 in child.children():
                ct += 1
                if ct <= self.Freeze_Num: # determines the amount of models to freeze
                    for param in child1.parameters():
                        param.requires_grad = False
        return densenet
    def forward(self, x):
        H = self.extractor(x)  # Nx1024
        return H

    
class ResNet18(nn.Module):
    def __init__(self,Freeze_Num=0):
        super(ResNet18,self).__init__()
        self.Freeze_Num=Freeze_Num
        self.extractor=self.resnet_extractor()
    def resnet_extractor(self):
        resnet=torchvision.models.resnet18(pretrained=True)
        resnet.Linear=nn.Identity()
        ct=0
        for child in resnet.children():
            ct+=1
            if ct<=self.Freeze_Num:
                for param in child1.parameters():
                    param.requires_grad=False
        return resnet
    def forward(self,x):
        H=self.extractor(x)
        return H



class attention(nn.Module):
    def __init__(self,D=16,K=1,L=1000):
        super(attention, self).__init__()
        self.L = L  # output of cnn. 1000 for resnet18, 1024 for densenet121 
        self.D = D  # nodes
        self.K = K  # weights
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )
    def forward(self, H, inf_mode='full', pooling='attention'):#pooling: mean,max,attention
        if inf_mode=='full':
            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N
            if pooling=='mean':
                M = torch.mean(H, dim=0, keepdim=True)
            elif pooling=='max':
                M = torch.max(H, dim=0, keepdim=True)[0]
            elif pooling=='attention':
                M = torch.mm(A, H)  # (K*N)X(N*1024) Kx1024
            else:
                print('pooling method not implemented!')      
            Y_prob = self.classifier(M)
            Y_hat = torch.ge(Y_prob, 0.5).float()
            return Y_prob, Y_hat, A
        elif inf_mode=='weight':
            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            return A
        elif inf_mode=='classify':
            Y_prob = self.classifier(H)
            Y_hat = torch.ge(Y_prob, 0.5).float()
            return Y_prob, Y_hat 
        else:
            raise Exception("Inference model not defined!")


    
def Loss(y_pred,y_true,balance=0.5):
    y_true=y_true.float()
    w = torch.Tensor([1 - balance, balance])
    y_pred = torch.clamp(y_pred, min=1e-5, max=1. - 1e-5)
    neg_log_likelihood = -1. * (2 * w[0] * y_true * torch.log(y_pred) + 2 * w[1] * (1. - y_true) * torch.log(
        1. - y_pred))  # negative log bernoulli
    return neg_log_likelihood


def ACC(y_pred,y_true):
    #y_pred: binary prediction
    y_true = y_true.float()
    ACC = y_pred.eq(y_true).float().diag().mean().item()
    return ACC
