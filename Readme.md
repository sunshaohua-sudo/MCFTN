The methods used in Article “Multimodal cross-attention mechanism twin networks for virtual screening of antibacterial cyclic peptides” will be briefly described in this paper.
# 1.Pre_train
This file contains the pre-training code and results of the model.
## pre_train.py
This is pre-training code. If you need to perform the fusion of feature vectors yourself, please download the feature vectors from the data source and replace the code path below.
```
esm_data_path=r'#########'#esm2_vec
grover_data_path=r'#########'#grover_vec
unimol_data_path=r'#########'#unimol_vec
```
And remove annotations `# Get_Data().main()`
## analyse_var.py
The document is used to analyze the optimal parameters after five-fold cross-validation and grid search.
# 2.Fine_tune
This folder contains the code and results related to model fine-tuning. The file structure is similar to [1.Pre_train](#1pre_train) and can be used as a reference.
# 3.model_predict
This file contains the code and conclusions used to filter the application model.
## predict.py
Main File for User Cyclic Peptide Prediction. During use, please replace the vector representation path in the code. This file can be downloaded from the data source.
```
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
```
This code is used to save the results.
```
with open('###############', 'w') as f:#save result
    json.dump(pre_train_vec, f)
f.close()
```
## *****1.json && ***2.json
These files contain the filtered results, where number 1 is the screening carried out targeting colistin, and number 2 is the screening carried out targeting colistimethate.
## analysis.py
This code is for analyzing the filtering results.
## result.json
This file contains the results after being analyzed and filtered by the model.
# 4.screen_rules
The folder contains code that performs screening based on expert knowledge.
# Find_Cyclic.py
The code in this file is used to screen cyclic peptide structures from peptides. Please replace the following code path during use.`content=pd.read_csv(r'################')#read the cyclic_peptide_detection_dataset.csv file.`
# requirements.txt
This project uses Python version 3.12.0, environment configuration files [requirements.txt](#requirements.txt), and other environment configuration files, such as: [esm2](https://github.com/cyan741/esm2), [grover](https://github.com/tencent-ailab/grover), and [unimol](https://github.com/deepmodeling/Uni-Mol).

**Note: If you have any questions, you can contact iamyzzhang@nwpu.edu.cn or sunshaohua@mail.nwpu.edu.cn**
