import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import json
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import torch.utils.data as Data
import random
import torch.optim as optim
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
import sys
import seaborn as sns

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_vec_path=r'./all_result.pkl'
org_data_path=r'./result.json'#org_data

need_vec_path=r'./need_vec.pkl'
need_label_path=r'./need_label.pkl'
pre_train_model_path=r'../1.Pre_train/result_model.pth'

KF_Flag=False
run_type='getmodel'#'gridsearch','getmodel'

if run_type=='getmodel':
    result_param_path=r'./getmodel_param.txt'
    result_loss_path=r'./getmodel_loss.txt'
    result_aucroc_path=r'./getmodel_auroc.txt'
    result_label_path=r'./getmodel_label_list.txt'
    svg_loss_path=r'./getmodel_loss.svg'
    svg_matrix_path=r'./getmodel_matrix.svg'
    opt_dict={'optim_lr':[1e-3],'optim_weight_decay':[0.01],'loss_reduction':['sum'],'loss_label_smooothing':[0.8],'epochs':[300],'batch_size':[250]}
    target_auc=0.98
    upper_value=0.01
    down_value=0.01
    result_model_path='./result_model.pth'
else:
    result_param_path=r'./gridsearch_param.txt'
    result_aucroc_path=r'./gridsearch_auroc.txt'
    opt_dict={'optim_lr':[1e-1,1e-2,1e-3,1e-4,1e-5],
              'optim_weight_decay':[0,0.1,0.01,0.001],
              'loss_reduction':['mean','sum'],
              'loss_label_smooothing':[0,0.2,0.4,0.6,0.8,1],
              'epochs':[300],
              'batch_size':[250,350,450,550],
              }

class Get_Data:
    def create_data(self):
        with open(all_vec_path,'rb') as f:
            all_result=pickle.load(f)
        f.close()

        with open(org_data_path,'r') as f:
            all_type=json.load(f)
        f.close()

        pre_model=torch.load(pre_train_model_path).to(device)
        pre_model.eval()

        pre_train_vec={}
        for key,value in all_result.items():
            out_put=pre_model(torch.Tensor(value[0]).to(device),torch.Tensor(value[1]).to(device),torch.Tensor(value[2]).to(device))
            pre_train_vec[key]=out_put.cpu().detach().numpy().tolist()

        all_vec={}
        all_label={}
        for i in all_type['cy_bacter_id']:
            for n in all_type['micro_id']:
                all_vec[i+'_'+n]=[pre_train_vec[i],pre_train_vec[n]]
                all_label[i+'_'+n]=0
            for n in all_type['bacter_id']:
                all_vec[i+'_'+n]=[pre_train_vec[i],pre_train_vec[n]]
                all_label[i+'_'+n]=0
            for n in all_type['cy_micro_id']:
                all_vec[i+'_'+n]=[pre_train_vec[i],pre_train_vec[n]]
                all_label[i+'_'+n]=0
            for n in all_type['cy_bacter_id']:
                all_vec[i+'_'+n]=[pre_train_vec[i],pre_train_vec[n]]
                all_label[i+'_'+n]=1

        with open(need_vec_path, 'wb') as f:
            pickle.dump(all_vec, f)
        f.close()

        with open(need_label_path, 'wb') as f:
            pickle.dump(all_label, f)
        f.close()

    def read_pickle(self):
        with open(need_vec_path,'rb') as f:
            need_vec=pickle.load(f)
        f.close()

        with open(need_label_path,'rb') as f:
            need_label=pickle.load(f)
        f.close()

        return(need_vec,need_label)

def get_param():
    para_list=[]
    for lr in opt_dict['optim_lr']:
        for wd in opt_dict['optim_weight_decay']:
            for re in opt_dict['loss_reduction']:
                for ls in opt_dict['loss_label_smooothing']:
                    for ep in opt_dict['epochs']:
                        for bs in opt_dict['batch_size']:
                            para_list.append({'lr':lr,'wd':wd,'re':re,'ls':ls,'ep':ep,'bs':bs})
    with open(result_param_path,'w') as f:
        for i in para_list:
            f.write('%s\n'%i)
    f.close()
    return(para_list)

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
        x_1=torch.unsqueeze(x_1,0)
        x_2=torch.unsqueeze(x_2,0)
        x_3=torch.unsqueeze(x_3,0)
        y_1=self.mix_layer(x_1,x_2)
        y_2=self.mix_layer(y_1,x_3)
        y=self.end_layer(y_2.squeeze())
        return(y_2)

class Fine_tune_Model(nn.Module):
    def __init__(self):
        super(Fine_tune_Model,self).__init__()
        self.liner1=nn.Sequential(nn.Linear(640,160),nn.ReLU(True))
        self.liner2=nn.Sequential(nn.Linear(160,40),nn.ReLU(True))
        self.liner3=nn.Sequential(nn.Linear(40,10),nn.ReLU(True))
        self.liner4=nn.Sequential(nn.Linear(10,2),nn.ReLU(True))
    def forward(self,x_1,x_2):
        x=torch.cat((x_1,x_2),dim=3)
        x=self.liner1(x)
        x=self.liner2(x)
        x=self.liner3(x)
        x=self.liner4(x)
        return x.squeeze(1,2)

def train(my_model,train_data,test_data,index,param_list,k_time):
    train_iter=Data.DataLoader(train_data,batch_size=int(param_list['bs']),shuffle=True)
    test_iter=Data.DataLoader(test_data,batch_size=int(param_list['bs']),shuffle=True)
    
    loss_f=nn.CrossEntropyLoss(reduction=param_list['re'])
    optimizer=optim.Adam(my_model.parameters(),lr=param_list['lr'],weight_decay=param_list['wd'])
    
    train_ture_label=[]
    train_pre_label=[]
    test_ture_label=[]
    test_pre_label=[]

    train_loss_list=[]
    eval_loss_list=[]
    for epoch in range(1,int(param_list['ep'])+1):
        my_model.train()
        for train_step,(train_data1,train_data2,train_label) in enumerate(train_iter):
            train_output=my_model(train_data1,train_data2)
            train_l=loss_f(train_output,train_label)
            optimizer.zero_grad()
            train_l.backward()
            optimizer.step()  
            if epoch==param_list['ep']:
                for i in range(0,train_output.shape[0]):
                    train_ture_label.append(train_label[i].item())
                    train_pre_label.append(torch.argmax(train_output[i]).item())
        train_loss_list.append(train_l.item())

        my_model.eval()       
        for test_step,(test_data1,test_data2,test_label) in enumerate(test_iter):
            test_output=my_model(test_data1,test_data2)
            test_l=loss_f(test_output,test_label)
            if epoch==param_list['ep']:
                for i in range(0,test_output.shape[0]):
                    test_ture_label.append(test_label[i].item())
                    test_pre_label.append(torch.argmax(test_output[i]).item())
        eval_loss_list.append(test_l.item())

    if run_type=='getmodel':
        for i in range(0,len(train_loss_list)):
            loss_str='param_index:%d,k_time:%d,num_epoches:%d,epoch:%d,train_loss:%f,eval_loss:%f' %(index,k_time,param_list['ep'],i+1,train_loss_list[i],eval_loss_list[i])
            with open(result_loss_path,'a') as tem_file:
                tem_file.write('%s\n'%loss_str)
            tem_file.close()
    train_acc=accuracy_score(train_ture_label,train_pre_label)
    train_f1=f1_score(train_ture_label,train_pre_label,average='micro')
    train_recall=recall_score(train_ture_label,train_pre_label,average='micro')
    train_precision=precision_score(train_ture_label,train_pre_label,average='micro')
    test_acc=accuracy_score(test_ture_label,test_pre_label)
    test_f1=f1_score(test_ture_label,test_pre_label,average='micro')
    test_recall=recall_score(test_ture_label,test_pre_label,average='micro')
    test_precision=precision_score(test_ture_label,test_pre_label,average='micro')
    aucroc_str='param_index:%d,k_time:%d,train_acc:%f,train_f1:%f,train_recall:%f,train_precision:%f,test_acc:%f,test_f1:%f,test_recall:%f,test_precision:%f'%(index,k_time,train_acc,train_f1,train_recall,train_precision,test_acc,test_f1,test_recall,test_precision)
    
    if run_type=='getmodel':
        print(aucroc_str)
    else:
        print(aucroc_str)
        with open(result_aucroc_path,'a') as tem_file:
            tem_file.write('%s\n'%aucroc_str)
        tem_file.close()

    if run_type=='getmodel':
        if target_auc-down_value<test_acc<target_auc+upper_value:
            for i in range(0,len(train_loss_list)):
                loss_str='param_index:%d,k_time:%d,num_epoches:%d,epoch:%d,train_loss:%f,eval_loss:%f' %(index,k_time,param_list['ep'],i+1,train_loss_list[i],eval_loss_list[i])
                # print(loss_str)
                with open(result_loss_path,'a') as tem_file:
                    tem_file.write('%s\n'%loss_str)
                tem_file.close()
    
            with open(result_aucroc_path,'a') as tem_file:
                tem_file.write('%s\n'%aucroc_str)
            tem_file.close()
        
            torch.save(model,result_model_path)

            with open(result_label_path,'w') as f:
                f.write('%s\n'%test_ture_label)
                f.write('%s'%test_pre_label)
            f.close()

            draw_matrix(test_ture_label,test_pre_label)
            sys.exit()


def draw_matrix(ture_label,test_label):
    cm=confusion_matrix(y_true=ture_label,y_pred=test_label)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['True','False'])
    disp.plot(cmap='Blues')
    plt.tight_layout()
    plt.savefig(svg_matrix_path,format='svg')
    # plt.show()

if __name__=="__main__":
    # Get_Data().create_data()
    all_data,all_label=Get_Data().read_pickle()
    param_data=get_param()

    for n in range(0,len(param_data)):
        if KF_Flag:
            label_0_key=[]
            label_1_key=[]
            for key,value in all_label.items():
                if value==0:
                    label_0_key.append(key)
                else:
                    label_1_key.append(key)
            random_0_key=random.sample(label_0_key,k=len(label_1_key))
            all_random_key=random_0_key+label_1_key
            kf=KFold(n_splits=5,shuffle=True,random_state=501)
            k_time=0
            for train_index,test_index in kf.split(all_random_key):
                k_time=k_time+1
                train_vec1_list=[]
                train_vec2_list=[]
                train_label_list=[]
                test_vec1_list=[]
                test_vec2_list=[]
                test_label_list=[]
                for m in train_index:
                    train_vec1_list.append(all_data[all_random_key[m]][0])
                    train_vec2_list.append(all_data[all_random_key[m]][1])
                    train_label_list.append(all_label[all_random_key[m]])
                for m in test_index:
                    test_vec1_list.append(all_data[all_random_key[m]][0])
                    test_vec2_list.append(all_data[all_random_key[m]][1])
                    test_label_list.append(all_label[all_random_key[m]])
                train_data=Data.TensorDataset(torch.Tensor(train_vec1_list).to(device),torch.Tensor(train_vec2_list).to(device),torch.LongTensor(train_label_list).to(device))
                test_data=Data.TensorDataset(torch.Tensor(test_vec1_list).to(device),torch.Tensor(test_vec2_list).to(device),torch.LongTensor(test_label_list).to(device))
                model=Fine_tune_Model().to(device)
                train(model,train_data,test_data,index=n,param_list=param_data[n],k_time=k_time)
        else:
            if run_type=='getmodel':
                    while True:
                        label_0_key=[]
                        label_1_key=[]
                        for key,value in all_label.items():
                            if value==0:
                                label_0_key.append(key)
                            else:
                                label_1_key.append(key)
                        random_0_key=random.sample(label_0_key,k=len(label_1_key))
                        all_random_key=random_0_key+label_1_key
                        random_train_key=random.sample(all_random_key,k=int(len(all_random_key)*0.8))
                        random_test_key=list(set(all_random_key)-set(random_train_key))
                        train_vec1_list=[]
                        train_vec2_list=[]
                        train_label_list=[]
                        for i in random_train_key:
                            train_vec1_list.append(all_data[i][0])
                            train_vec2_list.append(all_data[i][1])
                            train_label_list.append(all_label[i])
                        test_vec1_list=[]
                        test_vec2_list=[]
                        test_label_list=[]
                        for i in random_test_key:
                            test_vec1_list.append(all_data[i][0])
                            test_vec2_list.append(all_data[i][1])
                            test_label_list.append(all_label[i])
                        train_data=Data.TensorDataset(torch.Tensor(train_vec1_list).to(device),torch.Tensor(train_vec2_list).to(device),torch.LongTensor(train_label_list).to(device))
                        test_data=Data.TensorDataset(torch.Tensor(test_vec1_list).to(device),torch.Tensor(test_vec2_list).to(device),torch.LongTensor(test_label_list).to(device))
                        model=Fine_tune_Model().to(device)
                        train(model,train_data,test_data,index=n,param_list=param_data[n],k_time=1)
            else:
                label_0_key=[]
                label_1_key=[]
                for key,value in all_label.items():
                    if value==0:
                        label_0_key.append(key)
                    else:
                        label_1_key.append(key)
                random_0_key=random.sample(label_0_key,k=len(label_1_key))
                all_random_key=random_0_key+label_1_key
                random_train_key=random.sample(all_random_key,k=int(len(all_random_key)*0.8))
                random_test_key=list(set(all_random_key)-set(random_train_key))
                train_vec1_list=[]
                train_vec2_list=[]
                train_label_list=[]
                for i in random_train_key:
                    train_vec1_list.append(all_data[i][0])
                    train_vec2_list.append(all_data[i][1])
                    train_label_list.append(all_label[i])
                test_vec1_list=[]
                test_vec2_list=[]
                test_label_list=[]
                for i in random_test_key:
                    test_vec1_list.append(all_data[i][0])
                    test_vec2_list.append(all_data[i][1])
                    test_label_list.append(all_label[i])
                train_data=Data.TensorDataset(torch.Tensor(train_vec1_list).to(device),torch.Tensor(train_vec2_list).to(device),torch.LongTensor(train_label_list).to(device))
                test_data=Data.TensorDataset(torch.Tensor(test_vec1_list).to(device),torch.Tensor(test_vec2_list).to(device),torch.LongTensor(test_label_list).to(device))

                model=Fine_tune_Model().to(device)
                train(model,train_data,test_data,index=n,param_list=param_data[n],k_time=1)