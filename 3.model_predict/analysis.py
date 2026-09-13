import numpy as np
import json

scope=0.999999

class Get_C_Antimicro():
    def mix_target(self):
        target_1=self.get_target_1()
        target_2=self.get_target_2()
        mix_result={}
        for key,value in target_1.items():
            if key in target_2:
                mix_result[key]=[value,target_2[key]]
        return(mix_result)
    
    def get_target_1(self):
        with open('./antimicro_c_vec_target_1.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
    def get_target_2(self):
        with open('./antimicro_c_vec_target_2.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
  
class Get_TE_Antimicro():
    def mix_target(self):
        target_1=self.get_target_1()
        target_2=self.get_target_2()
        mix_result={}
        for key,value in target_1.items():
            if key in target_2:
                mix_result[key]=[value,target_2[key]]
        return(mix_result)
    
    def get_target_1(self):
        with open('./antimicro_te_vec_target_1.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
    def get_target_2(self):
        with open('./antimicro_te_vec_target_2.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
class Get_Cycpept():
    def mix_target(self):
        target_1=self.get_target_1()
        target_2=self.get_target_2()
        mix_result={}
        for key,value in target_1.items():
            if key in target_2:
                mix_result[key]=[value,target_2[key]]
        return(mix_result)
    
    def get_target_1(self):
        with open('./cycpept_vec_target_1.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
    def get_target_2(self):
        with open('./cycpept_vec_target_2.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
class Get_David():
    def mix_target(self):
        target_1=self.get_target_1()
        target_2=self.get_target_2()
        mix_result={}
        for key,value in target_1.items():
            if key in target_2:
                mix_result[key]=[value,target_2[key]]
        return(mix_result)
    
    def get_target_1(self):
        with open('./david_baker_vec_target_1.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
    def get_target_2(self):
        with open('./david_baker_vec_target_2.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
class Get_C_Self():
    def mix_target(self):
        target_1=self.get_target_1()
        target_2=self.get_target_2()
        mix_result={}
        for key,value in target_1.items():
            if key in target_2:
                mix_result[key]=[value,target_2[key]]
        return(mix_result)
    
    def get_target_1(self):
        with open('./self_org_c_vec_target_1.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
    def get_target_2(self):
        with open('./self_org_c_vec_target_2.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
class Get_TE_Self():
    def mix_target(self):
        target_1=self.get_target_1()
        target_2=self.get_target_2()
        mix_result={}
        for key,value in target_1.items():
            if key in target_2:
                mix_result[key]=[value,target_2[key]]
        return(mix_result)
    
    def get_target_1(self):
        with open('./self_org_te_vec_target_1.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)
    
    def get_target_2(self):
        with open('./self_org_te_vec_target_2.json') as f:
            data = json.load(f)
        f.close()
        result={}
        for key,value in data.items():
            if value >scope:
                result[key]=value
        return(result)

def save_result(result):
    with open('./result.json','w') as f:
        json.dump(result,f)
    f.close()

if __name__=='__main__':
    result_dict={}
    _=[]
    for key,value in Get_C_Antimicro().mix_target().items():
        _.append(key)
    result_dict['C_Antimicro']=_
    _=[]
    for key,value in Get_TE_Antimicro().mix_target().items():
        _.append(key)
    result_dict['TE_Antimicro']=_
    _=[]
    for key,value in Get_Cycpept().mix_target().items():
        _.append(key)
    result_dict['Cycpept']=_
    _=[]
    for key,value in Get_David().mix_target().items():
        _.append(key)
    result_dict['David']=_
    _=[]
    for key,value in Get_C_Self().mix_target().items():
        _.append(key)
    result_dict['C_Self']=_
    _=[]
    for key,value in Get_TE_Self().mix_target().items():
        _.append(key)
    result_dict['TE_Self']=_
    print(len(result_dict['TE_Self']))
    # save_result(result_dict)
