import pandas as pd
import numpy as np

import argparse
from yaml import parse
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from lightgbm import LGBMClassifier
from collections import Counter

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import KFold

import sys
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
parser = argparse.ArgumentParser()
parser.add_argument('--input-size', type=int, default=2083)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=150)

args = parser.parse_args()

#smile 차원조정
def smiles2morgan(s, nBits=64, radius=2):
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
        features = np.zeros((nBits,))
    return features

def tanimoto_similarity(vector1, vector2):
    min_length = min(len(vector1), len(vector2))
    vector1 = vector1[:min_length]
    vector2 = vector2[:min_length]
    intersection = np.sum(vector1 * vector2)
    union = np.sum(vector1 + vector2) - intersection
    return intersection / union

def cal_russel(fp1, fp2):
    fp1 = ''.join(fp1.astype(str))
    fp1 = DataStructs.CreateFromBitString(fp1)
    fp2 = ''.join(fp2.astype(str))
    fp2 = DataStructs.CreateFromBitString(fp2)
    similarity = DataStructs.RusselSimilarity(fp1, fp2)
    return similarity

def euclidean_distance_similarity(vector1, vector2):
    distance = np.linalg.norm(vector1 - vector2)
    dimension = len(vector1)
    similarity = 1 - (distance / np.sqrt(dimension))
    return similarity

# similarity matrix 생성
def create_similarity_matrix(one_hot_vectors1, one_hot_vectors2, criterion):
    num_vectors1 = len(one_hot_vectors1)
    num_vectors2 = len(one_hot_vectors2)
    similarity_matrix = np.zeros((num_vectors1, num_vectors2))

    for i in tqdm(range(num_vectors1), desc="Calculating Similarity"):
        for j in range(num_vectors2):
            similarity_matrix[i, j] = criterion(one_hot_vectors1[i], one_hot_vectors2[j])

    return similarity_matrix

def calculate_base_acc(df, adr_name):
    base_acc=0.0
    for i in df[adr_name]:
        base_acc+=i
    base_acc/=len(df)
    base_acc = max(base_acc, 1-base_acc)
    
    return base_acc

class medicineDataset(Dataset):
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        x = self.x[i,:]
        return x

class AUTO_ENCODER(torch.nn.Module):
    def __init__(self,args):
        super(AUTO_ENCODER, self).__init__()

        self.args = args

        self.encoder = nn.Sequential(
            nn.Linear(350, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 64),
            )
        
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(64, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 350),
        )
    
    def forward(self, x):
        eco = self.encoder(x)
        dco = self.decoder(eco)

        return eco, dco
    
def train_model_auto(data_loader, model, criterion, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X in tqdm(data_loader):
        eco_output, dco_output = model(X)

        loss = criterion(dco_output, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches

    print(f"Train Loss: {avg_loss}")

loss_mse = torch.nn.MSELoss()
# loss_bce = torch.nn.BCELoss()
# loss_bce = torch.nn.BCEWithLogitsLoss()

dataset = pd. read_csv('dataset(2)_0629.csv')
selected_label_data = dataset.loc[:,['drug_id','Name','THROMBOCYTOPENIA']]
selected_label_data = pd.DataFrame(selected_label_data)

smile_data_year=dataset.loc[:,['drug_id','ApprovalDate','SMILES']]

# total_data = pd.read_csv("./smile_data_0610.csv")
# label_data = total_data
# selected_label_data = pd.read_csv('./adr_selected.csv')
# smile_data_year = pd.read_csv('./smile_data_v3.csv')
smile_data_year = smile_data_year.sort_values(by='ApprovalDate')
smile_data_year = smile_data_year.head(350)

label_data = selected_label_data

print(smile_data_year)

filtered_smile_data = smile_data_year[smile_data_year['drug_id'].isin(selected_label_data['drug_id'])]
smile_year = filtered_smile_data['SMILES']
 
smile_total_set = dataset['SMILES']

morgan_year = pd.DataFrame([smiles2morgan(s) for s in tqdm(smile_year)])
data_filtered = np.array(morgan_year)
#data_400 = data_400[:400,:]

morgan_total = pd.DataFrame([smiles2morgan(s) for s in tqdm(smile_total_set)])
data_all = np.array(morgan_total)





print("creating russel similarity matrix")
similarity_matrix = create_similarity_matrix(data_all, data_filtered, tanimoto_similarity)
print("Similarity Matrix:")
print(similarity_matrix)

X = similarity_matrix
X = torch.FloatTensor(X)


model_auto_fp = AUTO_ENCODER(args)
optimizer_auto_fp = torch.optim.Adam(model_auto_fp.parameters(), lr=1e-5)
dataset_auto_fp = medicineDataset(X)
dataloader_auto_fp = DataLoader(dataset_auto_fp, batch_size=args.batch_size, shuffle = False)

for epoch in range(50):
    print(f"Epoch {epoch}\n ---------")
    train_model_auto(dataloader_auto_fp, model_auto_fp, loss_mse, optimizer_auto_fp)
    print("")

X_eco, X_dco = model_auto_fp(X)

adr_name_list = ["THROMBOCYTOPENIA"]
print(adr_name_list)


from sklearn.ensemble import RandomForestClassifier

#n_est_list = [3, 4, 5, 6, 7]
n_est_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# n_est_list = [1]
# criterion_list = ["gini"]
criterion_list = ["gini", "entropy", "log_loss"]
# max_depth_list = [13]
max_depth_list = [None, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# min_smaple_split_list = [6]
min_smaple_split_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# min_sample_leaf_list = [1]
min_sample_leaf_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


base_acc_list = []

f1_best_f1_list = []
f1_best_test_acc_list = []
f1_best_rs_list = []
f1_best_n_list = []
f1_best_j_list = []
f1_best_d_list = []
f1_best_c_list = []
f1_best_l_list = []
f1_best_matrix_list = []

acc_best_f1_list = []
acc_best_test_acc_list = []
acc_best_rs_list = []
acc_best_n_list = []
acc_best_j_list = []
acc_best_d_list = []
acc_best_c_list = []
acc_best_l_list = []
acc_best_matrix_list = []

rs_best_f1_list = []
rs_best_test_acc_list = []
rs_best_rs_list = []
rs_best_n_list = []
rs_best_j_list = []
rs_best_d_list = []
rs_best_c_list = []
rs_best_l_list = []
rs_best_matrix_list = []

total_best_f1_list = []
total_best_test_acc_list = []
total_best_rs_list = []
total_best_n_list = []
total_best_j_list = []
total_best_d_list = []
total_best_c_list = []
total_best_l_list = []
total_best_matrix_list = []

from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2024)

X = X_eco.detach().numpy()
X = pd.DataFrame(X)

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
for adr_name in adr_name_list:
    n_list = []
    j_list = []
    d_list = []
    c_list = []
    l_list = []
    test_acc_list = []
    f1_list = []
    matrix_list = []
    rs_list = []
    total_score_list = []
    
    base_acc = calculate_base_acc(label_data, adr_name)
    base_acc_list.append(base_acc)
    y = label_data[adr_name].to_numpy()
    
    X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(X, y, stratify=y, test_size=0.3)
    
    for n in n_est_list:
        for j in min_smaple_split_list:
            for d in max_depth_list:
                for c in criterion_list:
                    for l in min_sample_leaf_list:
                        y=pd.DataFrame(y)
                        
                        fold_result_f1 = []
                        fold_result_acc = []
                        fold_result_rs = []
                        fold_result_total = []
                        
                        for fold, (train_idx, valid_idx) in enumerate(k_fold.split(X_final_train, y_final_train)):
                            X_train, X_valid = X.iloc[train_idx,:],X.iloc[valid_idx,:]
                            y_train, y_valid = y.iloc[train_idx,:],y.iloc[valid_idx,:]
                            sm=SMOTE(random_state=2024)
                            #ada=ADASYN(random_state=2024)
                            X_train, y_train = sm.fit_resample(X_train, y_train)
                            #X_train, y_train = ada.fit_resample(X_train, y_train)

                            model = RandomForestClassifier(n_estimators=n, min_samples_split=j, max_depth=d, criterion=c, min_samples_leaf=l, random_state=0)
                            model.fit(X_train, y_train)
                
                            y_pred = model.predict(X_valid)
                
                            print("////////////////////////")
                            print("adr_name =", adr_name)
                            print("n = ", n)
                            print("j = ", j)
                            print("d = ", d)
                            print("c = ", c)
                            print("l = ", l)
                            print(f"train accuracy: {model.score(X_train, y_train)}")
                            print(f"test accuracy: {model.score(X_valid, y_valid)}")
                            print(f"f1 score: {f1_score(y_valid, y_pred)}")
                            print(f"confusion matrix: {confusion_matrix(y_valid, y_pred)}")
                            print(f"recall_score: {recall_score(y_valid, y_pred)}")
                            print(f"total score: {f1_score(y_valid, y_pred) + model.score(X_valid, y_valid) + recall_score(y_valid, y_pred)}")
                            
                            fold_result_f1.append(f1_score(y_valid, y_pred))
                            fold_result_acc.append(model.score(X_valid, y_valid))
                            fold_result_rs.append(recall_score(y_valid, y_pred))
                            fold_result_total.append(f1_score(y_valid, y_pred) + model.score(X_valid, y_valid) + recall_score(y_valid, y_pred))

                            print(f"base accuracy: {base_acc}")
                        
                        rs_list.append(sum(fold_result_rs)/len(fold_result_rs))
                        test_acc_list.append(sum(fold_result_acc)/len(fold_result_acc))
                        f1_list.append(sum(fold_result_f1)/len(fold_result_f1))
                        total_score_list.append(sum(fold_result_total)/len(fold_result_total))
                        n_list.append(n)
                        j_list.append(j)
                        d_list.append(d)
                        c_list.append(c)
                        l_list.append(l)
                        print(fold_result_f1)
                        print(fold_result_acc)
                        print(fold_result_rs)
                        print(fold_result_total)
    
    arg_max_f1 = np.argmax(f1_list)
    arg_max_acc = np.argmax(test_acc_list)
    arg_max_rs = np.argmax(rs_list)
    arg_max_total = np.argmax(total_score_list)
    result_list = []
    
    
    f1_n = n_list[arg_max_f1]
    f1_j = j_list[arg_max_f1]
    f1_d = d_list[arg_max_f1]
    f1_c = c_list[arg_max_f1]
    f1_l = l_list[arg_max_f1]
    
    acc_n = n_list[arg_max_acc]
    acc_j = j_list[arg_max_acc]
    acc_d = d_list[arg_max_acc]
    acc_c = c_list[arg_max_acc]
    acc_l = l_list[arg_max_acc]
    
    rs_n = n_list[arg_max_rs]
    rs_j = j_list[arg_max_rs]
    rs_d = d_list[arg_max_rs]
    rs_c = c_list[arg_max_rs]
    rs_l = l_list[arg_max_rs]

    total_n = n_list[arg_max_total]
    total_j = j_list[arg_max_total]
    total_d = d_list[arg_max_total]
    total_c = c_list[arg_max_total]
    total_l = l_list[arg_max_total]

    
    sm=SMOTE(random_state=2024)
    #ada = ADASYN(random_state=2024)
    X_final_train, y_final_train = sm.fit_resample(X_final_train, y_final_train)
    #X_final_train, y_final_train = ada.fit_resample(X_final_train, y_final_train)

    final_model_f1 = RandomForestClassifier(n_estimators=f1_n, min_samples_split=f1_j, max_depth=f1_d, criterion=f1_c, min_samples_leaf=f1_l, random_state=0)
    final_model_f1.fit(X_final_train, y_final_train)
    final_model_acc = RandomForestClassifier(n_estimators=acc_n, min_samples_split=acc_j, max_depth=acc_d, criterion=acc_c, min_samples_leaf=acc_l, random_state=0)
    final_model_acc.fit(X_final_train, y_final_train)
    final_model_rs = RandomForestClassifier(n_estimators=rs_n, min_samples_split=rs_j, max_depth=rs_d, criterion=rs_c, min_samples_leaf=rs_l, random_state=0)
    final_model_rs.fit(X_final_train, y_final_train)
    final_model_total = RandomForestClassifier(n_estimators=total_n, min_samples_split=total_j, max_depth=total_d, criterion=total_c, min_samples_leaf=total_l, random_state=0)
    final_model_total.fit(X_final_train, y_final_train)
    
    y_final_pred_f1 = final_model_f1.predict(X_final_test)
    y_final_pred_acc = final_model_acc.predict(X_final_test)
    y_final_pred_rs = final_model_rs.predict(X_final_test)
    y_final_pred_total = final_model_total.predict(X_final_test)
    
    
    # for i in range(len(f1_list)):
    #     if(f1_list[i]>=0.4 and test_acc_list[i]>=base_acc):
    #         result_list.append(i)
    
    # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # for j in result_list:
    #     print(f"best accuracy:\n{test_acc_list[j]}")
    #     print(f"best_f1:\n{f1_list[j]}")
    #     print(f"best_cm:\n{matrix_list[j]}")

    # print("f1: ")
    # print(f"best accuracy:\n{test_acc_list[arg_max_f1]}")
    # print(f"best_f1:\n{f1_list[arg_max_f1]}")
    # print(f"best_cm:\n{matrix_list[arg_max_f1]}")
    # print("acc")
    # print(f"best accuracy:\n{test_acc_list[arg_max_acc]}")
    # print(f"best_f1:\n{f1_list[arg_max_acc]}")
    # print(f"best_cm:\n{matrix_list[arg_max_acc]}")
    # print("rs")
    # print(f"best accuracy:\n{test_acc_list[arg_max_rs]}")
    # print(f"best_f1:\n{f1_list[arg_max_rs]}")
    # print(f"best_cm:\n{matrix_list[arg_max_rs]}")
    
    
    f1_best_f1_list.append(f1_score(y_final_test, y_final_pred_f1))
    f1_best_test_acc_list.append(final_model_f1.score(X_final_test, y_final_test))
    f1_best_rs_list.append(recall_score(y_final_test, y_final_pred_f1))
    f1_best_matrix_list.append(confusion_matrix(y_final_test, y_final_pred_f1))
    f1_best_n_list.append(f1_n)
    f1_best_j_list.append(f1_j)
    f1_best_d_list.append(f1_d)
    f1_best_c_list.append(f1_c)
    f1_best_l_list.append(f1_l)
    
    acc_best_f1_list.append(f1_score(y_final_test, y_final_pred_acc))
    acc_best_test_acc_list.append(final_model_acc.score(X_final_test, y_final_test))
    acc_best_rs_list.append(recall_score(y_final_test, y_final_pred_acc))
    acc_best_matrix_list.append(confusion_matrix(y_final_test, y_final_pred_acc))
    acc_best_n_list.append(acc_n)
    acc_best_j_list.append(acc_j)
    acc_best_d_list.append(acc_d)
    acc_best_c_list.append(acc_c)
    acc_best_l_list.append(acc_l)
    
    rs_best_f1_list.append(f1_score(y_final_test, y_final_pred_rs))
    rs_best_test_acc_list.append(final_model_rs.score(X_final_test, y_final_test))
    rs_best_rs_list.append(recall_score(y_final_test, y_final_pred_rs))
    rs_best_matrix_list.append(confusion_matrix(y_final_test, y_final_pred_rs))
    rs_best_n_list.append(rs_n)
    rs_best_j_list.append(rs_j)
    rs_best_d_list.append(rs_d)
    rs_best_c_list.append(rs_c)
    rs_best_l_list.append(rs_l)

    total_best_f1_list.append(f1_score(y_final_test, y_final_pred_total))
    total_best_test_acc_list.append(final_model_total.score(X_final_test, y_final_test))
    total_best_rs_list.append(recall_score(y_final_test, y_final_pred_total))
    total_best_matrix_list.append(confusion_matrix(y_final_test, y_final_pred_total))
    total_best_n_list.append(total_n)
    total_best_j_list.append(total_j)
    total_best_d_list.append(total_d)
    total_best_c_list.append(total_c)
    total_best_l_list.append(total_l)

datadatadata = {'adr_name': adr_name_list,
                'base_acc': base_acc_list,
                'f1_test_acc': f1_best_test_acc_list,
                'f1_f1_score': f1_best_f1_list,
                'f1_recall_scaore' : f1_best_rs_list,
                'f1_confusion_matrix' : f1_best_matrix_list,
                'f1_n_estimators': f1_best_n_list,
                'f1_min_sample_split': f1_best_j_list,
                'f1_max_depth': f1_best_d_list,
                'f1_criterion': f1_best_c_list,
                'f1_min_sample_leaf': f1_best_l_list,
                'base_acc': base_acc_list,
                'acc_test_acc': acc_best_test_acc_list,
                'acc_f1_score': acc_best_f1_list,
                'acc_recall_scaore' : acc_best_rs_list,
                'acc_confusion_matrix' : acc_best_matrix_list,
                'acc_n_estimators': acc_best_n_list,
                'acc_min_sample_split': acc_best_j_list,
                'acc_max_depth': acc_best_d_list,
                'acc_criterion': acc_best_c_list,
                'acc_min_sample_leaf': acc_best_l_list,
                'base_acc': base_acc_list,
                'rs_test_acc': rs_best_test_acc_list,
                'rs_f1_score': rs_best_f1_list,
                'rs_recall_scaore' : rs_best_rs_list,
                'rs_confusion_matrix' : rs_best_matrix_list,
                'rs_n_estimators': rs_best_n_list,
                'rs_min_sample_split': rs_best_j_list,
                'rs_max_depth': rs_best_d_list,
                'rs_criterion': rs_best_c_list,
                'rs_min_sample_leaf': rs_best_l_list,
                'base_acc': base_acc_list,
                'total_test_acc': total_best_test_acc_list,
                'total_f1_score': total_best_f1_list,
                'total_recall_scaore' : total_best_rs_list,
                'total_confusion_matrix' : total_best_matrix_list,
                'total_n_estimators': total_best_n_list,
                'total_min_sample_split': total_best_j_list,
                'total_max_depth': total_best_d_list,
                'total_criterion': total_best_c_list,
                'total_min_sample_leaf': total_best_l_list,
    }

dfdfdf = pd.DataFrame(datadatadata)

file_name = 'test_ranfo_0701_result_4fold_smote_testsize3_' + adr_name + '.csv'

dfdfdf.to_csv(file_name, index=False)