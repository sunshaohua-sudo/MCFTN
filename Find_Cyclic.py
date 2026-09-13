import pandas as pd
from rdkit import Chem
import os
import time
from rdkit.Chem import Draw
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
import numpy as np

IPythonConsole.ipython_useSVG=True
# np.set_printoptions(threshold=np.inf)

class Find_Cyclic:   
    def del_con(self):
        content=pd.read_csv(r'################')#read the cyclic_peptide_detection_dataset.csv file.
        id_list=content['id'].tolist()
        smiles_list=content['smiles'].tolist()
        type_list=content['type'].tolist()
        content_dict={}
        for i in range(0,len(id_list)):
            content_dict[id_list[i]]=[smiles_list[i],type_list[i]]
        result_dict={}
        for key,value in content_dict.items():
            network=nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(value[0])))
            flag=nx.is_connected(network)
            if flag==True:
                result_dict[key]=[network,value[0],value[1]]
            # else:
            #     print(key)
        return(result_dict)

    def draw_num(self,connected):
        for key,value in connected.items():
            diclofenac=Chem.MolFromSmiles(value[1])
            pirc=self.mol_with_atom_index(diclofenac)
            img=Draw.MolToImage(pirc,size=(1000,1000))
            save_path=r'./num_pic/'+str(key)+'_'+str(value[2])+'.png'
            img.save(save_path)

    def mol_with_atom_index(self,mol):
        atoms=mol.GetNumAtoms()
        for idx in range(atoms):
            mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber',str(mol.GetAtomWithIdx(idx).GetIdx()))
        return(mol)
    
    def get_label(self,connected):
        result_dict={}
        for key,value in connected.items():
            try:
                num_dict=self.get_dict(value[1])
                cycle = nx.cycle_basis(value[0])
                lin_list=[]
                for i in cycle:
                    if len(i)>6:
                        _=[]
                        for n in i:
                            _.append(num_dict[int(n)])
                        flag1=True
                        if len(list(set(_)))==1 and list(set(_))[0]=='C':
                            flag1=False
                        flag2=False
                        for n in _:
                            if n=='N':
                                flag2=True
                        if flag1 and flag2:
                            lin_list.append(i)
                result_dict[key]=[lin_list,len(lin_list),value[1],value[2]]                             
            except:
                continue
        return(result_dict)

    def get_dict(self,smiles):
        mol=Chem.MolFromSmiles(smiles)
        atoms=[]
        for atom in mol.GetAtoms():
            atoms.append(atom.GetSymbol())
        result_dict={}
        for i in range(0,len(atoms)):
            result_dict[i]=atoms[i]
        return(result_dict)

def save_result(result_dict):
    id_list=[]
    smiles_list=[]
    num_list=[]
    for key,value in result_dict.items():
        id_list.append(key)
        smiles_list.append(value[2])
        num_list.append(value[1])
    result_dic={'id':id_list,'smiles':smiles_list,'num':num_list}
    df_swiss=pd.DataFrame.from_dict(result_dic)
    df_swiss.to_csv(r'./result.csv',encoding='utf8',index=False)


if __name__=="__main__":
    connected=Find_Cyclic().del_con()
    result_dict=Find_Cyclic().get_label(connected)
    save_result(result_dict)
    Find_Cyclic().draw_num(connected)