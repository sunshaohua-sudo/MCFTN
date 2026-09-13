import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import re
import numpy as np
import json
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt
import random

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Get_Data:
    def get_result(self):
        with open('##############','r') as f:#target esm2 vector load, which can be found in data availability
            target_vec_1=json.load(f)['2']
        f.close()

        target_vec_2=np.load('################')['fps'][1]#target grover vector load, which can be found in data availability
        
        with open('##################','r') as f:#target unimol vector load, which can be found in data availability
            target_vec_3=json.load(f)['1']
        f.close()

        with open('##################','r') as f:#screen peptide esm2 vector load, which can be found in data availability
            org_vec_1=json.load(f)
        f.close()
        vec_1=[]
        for key,value in org_vec_1.items():
            vec_1.append(value)
        
        vec_2=np.load('##############')['fps']#screen peptide grover vector load, which can be found in data availability

        with open('##################','r') as f:#screen peptide unimol vector load, which can be found in data availability
            org_vec_3=json.load(f)
        f.close()
        vec_3=[]
        for key,value in org_vec_3.items():
            vec_3.append(value)

        pre_model=torch.load('../1.Pre_train/result_model.pth').to(device)
        pre_model.eval()

        # print(torch.Tensor(target_vec_1).shape,torch.Tensor(target_vec_2).shape,torch.Tensor(target_vec_3).shape)
        target_out_put=pre_model(torch.Tensor(target_vec_1).to(device),torch.Tensor(target_vec_2).to(device),torch.Tensor(target_vec_3).to(device))
        # print(target_out_put.shape)

        fine_model=torch.load('../2.Fine_tune/result_model.pth').to(device)
        fine_model.eval()

        pre_train_vec={}
        for i in range(0,len(vec_1)):
            print(i)
            # print(torch.Tensor(vec_1[i]).shape,torch.Tensor(vec_2[i]).shape,torch.Tensor(vec_3[i]).shape)
            out_put=pre_model(torch.Tensor(vec_1[i]).to(device),torch.Tensor(vec_2[i]).to(device),torch.Tensor(vec_3[i]).to(device))
            result=fine_model(target_out_put,out_put)
            pre_train_vec[i]=torch.round(F.softmax(result,dim=1),decimals=4).cpu().detach().numpy().tolist()[0][1]

        with open('###############', 'w') as f:#save result
            json.dump(pre_train_vec, f)
        f.close()

class Multi_CrossAttention(nn.Module):
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size      
        self.all_head_size  = all_head_size   
        self.num_heads      = head_num          
        self.h_size         = all_head_size // head_num
        assert all_head_size % head_num == 0
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)
        self.norm = sqrt(all_head_size)
    
    def forward(self,x,y):
        batch_size = x.size(0)
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)
        attention = F.scaled_dot_product_attention(q_s,k_s,v_s)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        output = self.linear_output(attention)
        return output

class F1_Model(nn.Module):
    def __init__(self):
        super(F1_Model,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(1280,960),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(960,640),nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(640,320),nn.ReLU(True))

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return(x)

class F2_Model(nn.Module):
    def __init__(self):
        super(F2_Model,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(5000,4064),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(4064,3128),nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(3128,2192),nn.ReLU(True))
        self.layer4=nn.Sequential(nn.Linear(2192,1295),nn.ReLU(True))
        self.layer5=nn.Sequential(nn.Linear(1295,320),nn.ReLU(True))

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        return(x)
    
class F3_Model(nn.Module):
    def __init__(self):
        super(F3_Model,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(512,458),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(458,394),nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(394,320),nn.ReLU(True))

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return(x)
    
class Out_Model(nn.Module):
    def __init__(self):
        super(Out_Model,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(320,80),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(80,20),nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(20,4),nn.ReLU(True))

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return(x)

class My_Model(nn.Module):
    def __init__(self):
        super(My_Model,self).__init__()
        self.trans_1=F1_Model()
        self.trans_2=F2_Model()
        self.trans_3=F3_Model()
        self.mix_layer = Multi_CrossAttention(320,320,64)
        self.end_layer=Out_Model()

    def forward(self,x1,x2,x3):
        x_1=self.trans_1(x1)
        x_2=self.trans_2(x2)
        x_3=self.trans_3(x3)
        # print(x_1.shape)
        # print(x_2.shape)
        # print(x_3.shape)
        x_1=torch.unsqueeze(x_1,0)
        x_2=torch.unsqueeze(x_2,0)
        x_3=torch.unsqueeze(x_3,0)
        # print(x_1.shape)

        y_1=self.mix_layer(x_1,x_2)
        y_2=self.mix_layer(y_1,x_3)
        # y_3=self.mix_layer(y_1,y_2)
        # print(y_1.shape)
        # print(y_2.shape)

        y=self.end_layer(y_2.squeeze())
        # print(y.shape)

        return(y_2)

class Fine_tune_Model(nn.Module):
    def __init__(self):
        super(Fine_tune_Model,self).__init__()
        self.liner1=nn.Sequential(nn.Linear(640,160),nn.ReLU(True))
        self.liner2=nn.Sequential(nn.Linear(160,40),nn.ReLU(True))
        self.liner3=nn.Sequential(nn.Linear(40,10),nn.ReLU(True))
        self.liner4=nn.Sequential(nn.Linear(10,2),nn.ReLU(True))
    def forward(self,x_1,x_2):
        x=torch.cat((x_1,x_2),dim=2)
        # print(x.shape)
        x=self.liner1(x)
        x=self.liner2(x)
        x=self.liner3(x)
        x=self.liner4(x)
        # print(x.shape)
        return x.squeeze(1,2)


if __name__=='__main__':
    Get_Data().get_result()
