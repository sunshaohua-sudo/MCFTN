import numpy as np
import json
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
from rdkit.Chem import Lipinski,GraphDescriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from Bio import SeqIO
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import codecs
from shutil import copyfile
from rdkit.Chem import Draw

class Get_C_Antimicro():
    def main(self,result):
        content=pd.read_csv('../3.antimicro/c_result.csv',header=None)
        file_path='../3.antimicro/c_pdb/'
        step_1=[]
        for i in result['C_Antimicro']:
            if Analysis_rules().calculate_instability(content.loc[int(i)][0]):
                step_1.append(i)
        # print(step_1)
        step_2=[]
        for i in step_1:
            if Analysis_rules().calculate_gravy(content.loc[int(i)][0]):
                step_2.append(i)
        # print(step_2)
        step_3=[]
        for i in step_2:
            if Analysis_rules().calculate_netcharge(content.loc[int(i)][0]):
                step_3.append(i)
        # print(step_3)
        step_4=[]
        for i in step_3:
            if Analysis_rules().calculate_isoelectricpoint(content.loc[int(i)][0]):
                step_4.append(i)
        print(step_4)

        alpha_list=[]
        for i in step_4:
            alpha_list.append(Analysis_rules().calculate_alpha(Chem.MolFromPDBFile(file_path+i+'.pdb')))
        # print(alpha_list)

        rot_list=[]
        for i in step_4:
            rot_list.append(Analysis_rules().calculate_rot(Chem.MolFromPDBFile(file_path+i+'.pdb')))
        # print(rot_list)

        rgyr_list=[]
        for i in step_4:
            rgyr_list.append(Analysis_rules().calculate_rgyr(codecs.open(file_path+i+'.pdb',mode='r',encoding='utf-8')))
        # print(rgyr_list)

        return(list(zip(step_4,alpha_list,rot_list,rgyr_list)))
    
    def draw_hydrophobicity(self,id_list,window_size):
        content=pd.read_csv('../3.antimicro/c_result.csv',header=None)
        for i in id_list:
            save_path='./c_antimicro_result/hydrophobicity_'+str(window_size)+'_'+str(i)+'.svg'
            Analysis_rules().pro_scale(content.loc[int(i)][0],window_size,save_path)

    def draw_isoelectricpoint(self,id_list):
        content=pd.read_csv('../3.antimicro/c_result.csv',header=None)
        for i in id_list:
            save_path='./c_antimicro_result/isoelectricpoint_'+str(i)+'.svg'
            Analysis_rules().get_isopic(content.loc[int(i)][0],save_path)

    def draw_smiles(self,id_list):
        content=pd.read_csv('../3.antimicro/c_result.csv',header=None)
        mol=Chem.MolFromSmiles(content.loc[int(id_list[0])][1])
        Draw.MolToFile(mol,'./temp.svg')

    def save_result(self,id_list):
        content=pd.read_csv('../3.antimicro/c_result.csv',header=None)
        file_path='../3.antimicro/c_pdb/'
        result_dict={}
        for i in id_list:
            org_file_path=file_path+str(i)+'.pdb'
            target_file_path='./c_antimicro_result/'+str(i)+'.pdb'
            copyfile(org_file_path,target_file_path)
            result_dict[i]=[content.loc[int(i)][0],content.loc[int(i)][1]]
        result=json.dumps(result_dict)
        f=open(r'./c_antimicro_result/c_antimicro_result.json','w')
        f.write(result)
        f.close()

class Get_TE_Antimicro():
    def main(self,result):
        content=pd.read_csv('../3.antimicro/te_result.csv',header=None)
        file_path='../3.antimicro/te_pdb/'
        step_1=[]
        for i in result['TE_Antimicro']:
            if Analysis_rules().calculate_instability(content.loc[int(i)][0]):
                step_1.append(i)
        # print(step_1)
        step_2=[]
        for i in step_1:
            if Analysis_rules().calculate_gravy(content.loc[int(i)][0]):
                step_2.append(i)
        # print(step_2)
        step_3=[]
        for i in step_2:
            if Analysis_rules().calculate_netcharge(content.loc[int(i)][0]):
                step_3.append(i)
        # print(step_3)
        step_4=[]
        for i in step_3:
            if Analysis_rules().calculate_isoelectricpoint(content.loc[int(i)][0]):
                step_4.append(i)
        print(step_4)
        alpha_list=[]
        for i in step_4:
            alpha_list.append(Analysis_rules().calculate_alpha(Chem.MolFromPDBFile(file_path+i+'.pdb')))
        # print(alpha_list)

        rot_list=[]
        for i in step_4:
            rot_list.append(Analysis_rules().calculate_rot(Chem.MolFromPDBFile(file_path+i+'.pdb')))
        # print(rot_list)

        rgyr_list=[]
        for i in step_4:
            rgyr_list.append(Analysis_rules().calculate_rgyr(codecs.open(file_path+i+'.pdb',mode='r',encoding='utf-8')))
        # print(rgyr_list)

        return(list(zip(step_4,alpha_list,rot_list,rgyr_list)))
    
    def draw_hydrophobicity(self,id_list,window_size):
        content=pd.read_csv('../3.antimicro/te_result.csv',header=None)
        for i in id_list:
            save_path='./te_antimicro_result/hydrophobicity_'+str(window_size)+'_'+str(i)+'.svg'
            Analysis_rules().pro_scale(content.loc[int(i)][0],window_size,save_path)

    def draw_isoelectricpoint(self,id_list):
        content=pd.read_csv('../3.antimicro/te_result.csv',header=None)
        for i in id_list:
            save_path='./te_antimicro_result/isoelectricpoint_'+str(i)+'.svg'
            Analysis_rules().get_isopic(content.loc[int(i)][0],save_path)

    def save_result(self,id_list):
        content=pd.read_csv('../3.antimicro/te_result.csv',header=None)
        file_path='../3.antimicro/te_pdb/'
        result_dict={}
        for i in id_list:
            org_file_path=file_path+str(i)+'.pdb'
            target_file_path='./te_antimicro_result/'+str(i)+'.pdb'
            copyfile(org_file_path,target_file_path)
            result_dict[i]=[content.loc[int(i)][0],content.loc[int(i)][1]]
        result=json.dumps(result_dict)
        f=open(r'./te_antimicro_result/te_antimicro_result.json','w')
        f.write(result)
        f.close()
    
class Get_David():
    def main(self,result):
        file_list=os.listdir(r'../4.random_david_baker/right_pdb/')
        step_1=[]
        for i in result['David']:
            if Analysis_rules().calculate_instability(file_list[int(i)].split('.')[0]):
                step_1.append(i)
        # print(step_1)
        step_2=[]
        for i in step_1:
            if Analysis_rules().calculate_gravy(file_list[int(i)].split('.')[0]):
                step_2.append(i)
        # print(step_2)
        step_3=[]
        for i in step_2:
            if Analysis_rules().calculate_netcharge(file_list[int(i)].split('.')[0]):
                step_3.append(i)
        # print(step_3)
        step_4=[]
        for i in step_3:
            if Analysis_rules().calculate_isoelectricpoint(file_list[int(i)].split('.')[0]):
                step_4.append(i)
        print(step_4)
    
class Get_C_Self():
    def main(self,result):
        content=pd.read_csv('../1.self_org/2.select/c_result.csv',header=None)
        file_path='../1.self_org/2.select/c_pdb/'
        step_1=[]
        for i in result['C_Antimicro']:
            if Analysis_rules().calculate_instability(content.loc[int(i)][0]):
                step_1.append(i)
        # print(step_1)
        step_2=[]
        for i in step_1:
            if Analysis_rules().calculate_gravy(content.loc[int(i)][0]):
                step_2.append(i)
        # print(step_2)
        step_3=[]
        for i in step_2:
            if Analysis_rules().calculate_netcharge(content.loc[int(i)][0]):
                step_3.append(i)
        # print(step_3)
        step_4=[]
        for i in step_3:
            if Analysis_rules().calculate_isoelectricpoint(content.loc[int(i)][0]):
                step_4.append(i)
        print(step_4)
        alpha_list=[]
        for i in step_4:
            alpha_list.append(Analysis_rules().calculate_alpha(Chem.MolFromPDBFile(file_path+i+'.pdb')))
        # print(alpha_list)
        rot_list=[]
        for i in step_4:
            rot_list.append(Analysis_rules().calculate_rot(Chem.MolFromPDBFile(file_path+i+'.pdb')))
        # print(rot_list)

        rgyr_list=[]
        for i in step_4:
            rgyr_list.append(Analysis_rules().calculate_rgyr(codecs.open(file_path+i+'.pdb',mode='r',encoding='utf-8')))
        # print(rgyr_list)

        return(list(zip(step_4,alpha_list,rot_list,rgyr_list)))
    
    def draw_hydrophobicity(self,id_list,window_size):
        content=pd.read_csv('../1.self_org/2.select/c_result.csv',header=None)
        for i in id_list:
            save_path='./c_self_result/hydrophobicity_'+str(window_size)+'_'+str(i)+'.svg'
            Analysis_rules().pro_scale(content.loc[int(i)][0],window_size,save_path)

    def draw_isoelectricpoint(self,id_list):
        content=pd.read_csv('../1.self_org/2.select/c_result.csv',header=None)
        for i in id_list:
            save_path='./c_self_result/isoelectricpoint_'+str(i)+'.svg'
            Analysis_rules().get_isopic(content.loc[int(i)][0],save_path)
    
    def draw_smiles(self,id_list):
        content=pd.read_csv('../1.self_org/2.select/c_result.csv',header=None)
        for i in id_list:
            mol=Chem.MolFromSmiles(content.loc[int(i)][1])
            path='./smiles_'+str(i)+'.svg'
            Draw.MolToFile(mol,path)

    def save_result(self,id_list):
        content=pd.read_csv('../1.self_org/2.select/c_result.csv',header=None)
        file_path='../1.self_org/2.select/c_pdb/'
        result_dict={}
        for i in id_list:
            org_file_path=file_path+str(i)+'.pdb'
            target_file_path='./c_self_result/'+str(i)+'.pdb'
            copyfile(org_file_path,target_file_path)
            result_dict[i]=[content.loc[int(i)][0],content.loc[int(i)][1]]
        result=json.dumps(result_dict)
        f=open(r'./c_self_result/c_self_result.json','w')
        f.write(result)
        f.close()
    
class Get_TE_Self():
    def main(self,result):
        content=pd.read_csv('../1.self_org/2.select/te_result.csv',header=None)
        file_path='../1.self_org/2.select/te_pdb/'
        step_1=[]
        for i in result['TE_Antimicro']:
            if Analysis_rules().calculate_instability(content.loc[int(i)][0]):
                step_1.append(i)
        # print(step_1)
        step_2=[]
        for i in step_1:
            if Analysis_rules().calculate_gravy(content.loc[int(i)][0]):
                step_2.append(i)
        # print(step_2)
        step_3=[]
        for i in step_2:
            if Analysis_rules().calculate_netcharge(content.loc[int(i)][0]):
                step_3.append(i)
        # print(step_3)
        step_4=[]
        for i in step_3:
            if Analysis_rules().calculate_isoelectricpoint(content.loc[int(i)][0]):
                step_4.append(i)
        print(step_4)
        alpha_list=[]
        for i in step_4:
            alpha_list.append(Analysis_rules().calculate_alpha(Chem.MolFromPDBFile(file_path+i+'.pdb')))
        # print(alpha_list)

        rot_list=[]
        for i in step_4:
            rot_list.append(Analysis_rules().calculate_rot(Chem.MolFromPDBFile(file_path+i+'.pdb')))
        # print(rot_list)

        rgyr_list=[]
        for i in step_4:
            rgyr_list.append(Analysis_rules().calculate_rgyr(codecs.open(file_path+i+'.pdb',mode='r',encoding='utf-8')))
        # print(rgyr_list)

        return(list(zip(step_4,alpha_list,rot_list,rgyr_list)))

def read_result():
    with open('../7.model_predict/result.json','r') as f:
        result=json.load(f)
    f.close()
    return(result)

class Analysis_rules():
    def calculate_weight(self,content):
        if Descriptors.MolWt(content)<1000:
            return(True)
    def calculate_hg(self,content):
        if Lipinski.NumHAcceptors(content)<15:
            return(True)
    def calculate_hs(self,content):
        if Lipinski.NumHDonors(content)<6:
            return(True)
    def calculate_rotate(self,content):
        if Lipinski.NumRotatableBonds(content)<20:
            return(True)
    def calculate_tpsa(self,content):
        if Descriptors.TPSA(content)<250:
            return(True)
    def calculate_logp(self,content):
        print(Descriptors.MolLogP(content))
        if 7.5<Descriptors.MolLogP(content)<10:
            return(True)
    def calculate_rgyr(self,A):
        line=A.readline()
        list1=[]
        list2=[]

        while line:
            a=line.split()
            if a[11:12]!=[]:
                b=a[6:7]
                c=a[7:8]
                d=a[8:9]
                e=a[11:12]
            elif a[6:7]!=[]:
                if len(a[9:10][0])<5:
                    b=a[5:6]
                    c=a[6:7]
                    d=a[7:8]
                    e=a[10:11]
                if len(a[9:10][0])>5:
                    b=a[6:7]
                    c=a[7:8]
                    d=a[8:9]
                    e=a[10:11]
            list1.append(b)
            list1.append(c)
            list1.append(d)
            list2.append(e)
            line=A.readline()
    
        A.close()
        dim=int(len(list1)/3-2)
        atom_M=np.array([0.0]*dim)
        atom_x=np.array([0.0]*dim)
        atom_y=np.array([0.0]*dim)
        atom_z=np.array([0.0]*dim)

        for i in range(dim):
            atom_x[i]=float(list1[i*3+0][0])
            atom_y[i]=float(list1[i*3+1][0])
            atom_z[i]=float(list1[i*3+2][0])
        total_M=0
        for j in range(dim):
            atom_M[j]=float(self.calcum(list2[j][0]))
            total_M+=atom_M[j]

        rg=self.calcuRg(atom_x,atom_y,atom_y,atom_M,total_M,dim)

        return(rg)
    def calculate_alpha(self,content):
        return(GraphDescriptors._pyHallKierAlpha(content))
    def calculate_rot(self,content):
        return(Lipinski.NumRotatableBonds(content))
    def calculate_isoelectricpoint(self,content):
        if 8.9<IsoelectricPoint(content).pi()<10.7:
            return(True)
    def calculate_netcharge(self,content):
        if IsoelectricPoint(content).charge_at_pH(7.4)>0:
            return(True)
    def calculate_instability(self,content):
        if ProteinAnalysis(content).instability_index()<40:
            return(True)
    def calculate_gravy(self,content):
        if ProteinAnalysis(content).gravy(scale='KyteDoolitle')<1:
            return(True)
    def get_isopic(self,content,path):
        pi_list=[]
        for i in range(0,15):
            pi_list.append(IsoelectricPoint(content).charge_at_pH(i))
        self.draw_pi(pi_list,path)
    def draw_pi(self,data,path):
        label_list=[]
        for i in range(0,15):
            label_list.append(i)
        plt.plot(label_list, data, linestyle="-", marker=".", label="IsoelectricPoint")
        plt.xlabel('pH')
        plt.xticks(np.arange(0,15,1))
        plt.ylabel('Polarity')
        plt.legend()
        plt.grid(color='grey', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.savefig(path,format='svg')
        plt.cla()
        # plt.show()
    def pro_scale(self,content,window_size,path):
        analysed_seq = ProteinAnalysis(content)
        hydrophobicity_dict={'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5}
        scale = analysed_seq.protein_scale(param_dict=hydrophobicity_dict, window=window_size,edge=1)
        self.draw_scale(scale,path)
    def draw_scale(self,data,path):
        label_list=[]
        x_list=[]
        for i in range(0,len(data)):
            label_list.append(i+1)
            x_list.append(i+1)
        plt.plot(label_list, data, linestyle="-", marker=".", label="Hydropath./Kyte Doolittle")
        plt.xticks(x_list)
        plt.xlabel('Position')
        plt.yticks(np.arange(-3,3,0.5))
        plt.ylabel('Hydrophobicity Score')
        plt.legend()
        plt.grid(color='grey', linestyle='--', linewidth=1)
        plt.tight_layout()
        plt.savefig(path,format='svg')
        plt.cla()
        # plt.show()
    def calcum(self,A):
        if(A=='C'):
            m=12
        elif(A=='O'):
            m=16
        elif(A=='N'):
            m=14
        elif(A=='H'):
            m=1
        elif(A=='FE'):
            m=56
        elif(A=='P'):
            m=31
        elif(A=='S'):
            m=32
        elif(A=='CU'):
            m=64
        elif(A=='CA'):
            m=40
        else:
            m=0
        return m
    def calcuRc(self,x,m,M,dim):
        X=0
        for i in range(dim):
            X+=m[i]*x[i]/M
        return X
    def calcuRg(self,x,y,z,m,M,dim):
        X=self.calcuRc(x,m,M,dim)
        Y=self.calcuRc(y,m,M,dim)
        Z=self.calcuRc(z,m,M,dim)
        Rg2=0
        for i in range(dim):
            Rg2+=m[i]*((x[i]-X)**2+(y[i]-Y)**2+(z[i]-Z)**2)
        return np.sqrt(Rg2/M)



if __name__=='__main__':
    result=read_result()
    c_antimicro_list=Get_C_Antimicro().main(result)
    te_antimicro_list=Get_TE_Antimicro().main(result)
    Get_David().main(result)
    c_self_list=Get_C_Self().main(result)
    te_self_list=Get_TE_Self().main(result)