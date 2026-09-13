import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import re
import numpy as np

class Get_data():
    def get_con(self,result_path):
        file=open(result_path)
        org_list=file.readlines()

        train_acc=[]
        train_f1=[]
        train_recall=[]
        train_precision=[]
    
        test_acc=[]
        test_f1=[]
        test_recall=[]
        test_precision=[]

        result_dic={}

        for i in range(0,len(org_list)):
            time_num=int(re.search(r'k_time:\d+',org_list[i]).group().split(':')[1])

            train_acc.append(float(re.search(r'train_acc:\d+.\d+',org_list[i]).group().split(':')[1]))
            train_f1.append(float(re.search(r'train_f1:\d+.\d+',org_list[i]).group().split(':')[1]))
            train_recall.append(float(re.search(r'train_recall:\d+.\d+',org_list[i]).group().split(':')[1]))
            train_precision.append(float(re.search(r'train_recall:\d+.\d+',org_list[i]).group().split(':')[1]))

            test_acc.append(float(re.search(r'test_acc:\d+.\d+',org_list[i]).group().split(':')[1]))
            test_f1.append(float(re.search(r'test_f1:\d+.\d+',org_list[i]).group().split(':')[1]))
            test_recall.append(float(re.search(r'test_recall:\d+.\d+',org_list[i]).group().split(':')[1]))
            test_precision.append(float(re.search(r'test_precision:\d+.\d+',org_list[i]).group().split(':')[1]))

            if time_num==5:
                index=int(re.search(r'param_index:\d+',org_list[i]).group().split(':')[1])
                result_dic[index]={'train_acc':{'avg':float(np.mean(train_acc)),'std':float(np.std(train_acc))},
                               'train_f1':{'avg':float(np.mean(train_f1)),'std':float(np.std(train_f1))},
                               'train_recall':{'avg':float(np.mean(train_recall)),'std':float(np.std(train_recall))},
                               'train_precision':{'avg':float(np.mean(train_precision)),'std':float(np.std(train_precision))},
                               'test_acc':{'avg':float(np.mean(test_acc)),'std':float(np.std(test_acc))},
                               'test_f1':{'avg':float(np.mean(test_f1)),'std':float(np.std(test_f1))},
                               'test_recall':{'avg':float(np.mean(test_recall)),'std':float(np.std(test_recall))},
                               'test_precision':{'avg':float(np.mean(test_precision)),'std':float(np.std(test_precision))},
                               }

                train_acc=[]
                train_f1=[]
                train_recall=[]
                train_precision=[]
    
                test_acc=[]
                test_f1=[]
                test_recall=[]
                test_precision=[]   
        return(result_dic)

def Draw_scatter(all_data):
    auc_list=[]
    std_list=[]
    for key,value in all_data.items():
        auc_list.append(value['test_acc']['avg'])
        std_list.append(value['test_acc']['std'])
    
    ax=plt.subplot()
    ax.scatter(auc_list,std_list,marker='.',s=10,alpha=0.5,c='green')
    ax.set_xlabel('auc')
    ax.set_ylabel('std')
    plt.tight_layout()
    plt.savefig('./auroc_std.svg',format='svg')
    # plt.show()

class Choice_parmerter():
    def Chemprop_parmerter(self,all_data):
        auc_list=[]
        std_list=[]
        for key,value in all_data.items():
            auc_list.append([value['test_acc']['avg']])
            std_list.append([value['test_acc']['std']])
        auc_scaler=MinMaxScaler(feature_range=(0,1))
        auc_scaler.fit(auc_list)
        auc_result=auc_scaler.transform(auc_list)
    
        std_scaler=MinMaxScaler(feature_range=(0,1))
        std_scaler.fit(std_list)
        std_result=std_scaler.transform(std_list)

        result_list=[]
        for i in range(0,len(auc_result)):
            result_list.append(float(auc_result[i]))
        index_num=result_list.index(max(result_list))
        print('best_num：'+str(index_num))
        print('best_acc：'+str(auc_list[index_num])+'best_std：'+str(std_list[index_num]))

if __name__=='__main__':
    all_data=Get_data().get_con('./gridsearch_auroc.txt')
    Draw_scatter(all_data)
    # Choice_parmerter().Chemprop_parmerter(all_data)