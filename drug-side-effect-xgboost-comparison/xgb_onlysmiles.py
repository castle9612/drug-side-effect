import pandas as pd
import numpy as np

import argparse
from yaml import parse
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from collections import Counter

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
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

class target_actionDataset(Dataset):
    def __init__(self, dataframe):
        # dataframe.values는 호출이 아닌 속성 접근
        self.data = dataframe.values
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx, :], dtype=torch.float32)

class AUTO_ENCODER(torch.nn.Module):
    def __init__(self,args):
        super(AUTO_ENCODER, self).__init__()

        self.args = args

        self.encoder = nn.Sequential(
            nn.Linear(128, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            # nn.LeakyReLU(),
            # nn.Linear(128, 64),
            )
        
        self.decoder = nn.Sequential(
            # nn.LeakyReLU(),
            # nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 128),
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

dataset = pd. read_csv('../drug-side-effect-shared-datasets/datasets_smiles_approval.csv')
print(dataset)
selected_label_data = dataset.loc[:,['DRUGBANK_ID','Name','THROMBOCYTOPENIA']]
selected_label_data = pd.DataFrame(selected_label_data)

smile_data_year=dataset.loc[:,['DRUGBANK_ID','ApprovalDate','SMILES']]

# total_data = pd.read_csv("./smile_data_0610.csv")
# label_data = total_data
# selected_label_data = pd.read_csv('./adr_selected.csv')
# smile_data_year = pd.read_csv('./smile_data_v3.csv')
smile_data_year = smile_data_year.head(128)

label_data = selected_label_data

print(smile_data_year)

filtered_smile_data = smile_data_year[smile_data_year['DRUGBANK_ID'].isin(selected_label_data['DRUGBANK_ID'])]
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
# print(X)


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

learning_rate = [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
max_depth = [1,3,5,7,9,11,13,15,30]
colsample_bytree = [0.3,0.5,0.7,1.0]
lambda_l1 = [0.0, 0.1, 0.3, 0.5, 0.8]

base_acc_list = []

f1_best_f1_list = []
f1_best_test_acc_list = []
f1_best_rs_list = []
f1_best_auc_list = []
f1_best_lr_list = []
f1_best_md_list = []
f1_best_cb_list = []
f1_best_l1_list = []
f1_best_matrix_list = []

auc_best_f1_list = []
auc_best_test_acc_list = []
auc_best_rs_list = []
auc_best_auc_list = []
auc_best_lr_list = []
auc_best_md_list = []
auc_best_cb_list = []
auc_best_l1_list = []
auc_best_matrix_list = []


acc_best_f1_list = []
acc_best_test_acc_list = []
acc_best_rs_list = []
acc_best_auc_list = []
acc_best_lr_list = []
acc_best_md_list = []
acc_best_cb_list = []
acc_best_l1_list = []
acc_best_matrix_list = []

rs_best_f1_list = []
rs_best_test_acc_list = []
rs_best_rs_list = []
rs_best_auc_list = []
rs_best_lr_list = []
rs_best_md_list = []
rs_best_cb_list = []
rs_best_l1_list = []
rs_best_matrix_list = []

total_best_f1_list = []
total_best_test_acc_list = []
total_best_rs_list = []
total_best_auc_list = []
total_best_lr_list = []
total_best_md_list = []
total_best_cb_list = []
total_best_l1_list = []
total_best_matrix_list = []

from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)


X_eco = X_eco.detach().numpy()
X_eco = pd.DataFrame(X_eco)
X_eco.columns = [f"encoded_{i}" for i in range(X_eco.shape[1])]

# X=pd.concat([X_eco,X_eco1],axis=1)
X=pd.DataFrame(X_eco)
# X=X.tail(len(X)-128)
# print(X)
# print(X)
# import sys
# sys.exit()
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
for adr_name in adr_name_list:
    lr_list = []
    md_list = []
    cb_list = []
    l1_list = []
    test_acc_list = []
    f1_list = []
    auc_list = []
    matrix_list = []
    rs_list = []
    total_score_list = []
    
    base_acc = calculate_base_acc(label_data, adr_name)
    base_acc_list.append(base_acc)
    y = label_data[adr_name]
    # y = y.tail(len(y)-128)
    y = y.to_numpy()
    X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(X, y, stratify=y, test_size=0.2)
    print(len(X_final_train))
    print(len(X_final_test))
    # import sys
    # sys.exit()
    # print(X_final_train)
    # print('test')
    # print(X_final_test)
    for lr in learning_rate:
        for md in max_depth:
            for cb in colsample_bytree:
                for l1 in lambda_l1:
                    y=pd.DataFrame(y)
                    
                    fold_result_f1 = []
                    fold_result_acc = []
                    fold_result_rs = []
                    fold_result_total = []
                    fold_result_auc = []
                    
                    for fold, (train_idx, valid_idx) in enumerate(k_fold.split(X_final_train, y_final_train)):
                        X_train, X_valid = X.iloc[train_idx,:],X.iloc[valid_idx,:]
                        y_train, y_valid = y.iloc[train_idx,:],y.iloc[valid_idx,:]
                        sm=SMOTE(random_state=2024)
                        #ada=ADASYN(random_state=2024)
                        X_train, y_train = sm.fit_resample(X_train, y_train)
                        # print(X_train)
                        # print(y_train)
                        #X_train, y_train = ada.fit_resample(X_train, y_train)

                        model = XGBClassifier(learning_rate=lr, max_depth=md, colsample_bytree=cb, alpha=l1, objective='binary:logistic',random_state=2024)
                        model.fit(X_train, y_train)
            
                        y_pred = model.predict(X_valid)
            
                        print("////////////////////////")
                        print("adr_name =", adr_name)
                        print("learning_rate = ", lr)
                        print("max_depth = ", md)
                        print("colsample_bytree = ", cb)
                        print("lambda_l1 = ", l1)
                        print(f"train accuracy: {model.score(X_train, y_train)}")
                        print(f"test accuracy: {model.score(X_valid, y_valid)}")
                        print(f"f1 score: {f1_score(y_valid, y_pred)}")
                        print(f"confusion matrix: {confusion_matrix(y_valid, y_pred)}")
                        print(f"recall_score: {recall_score(y_valid, y_pred)}")
                        print(f"total score: {f1_score(y_valid, y_pred) + model.score(X_valid, y_valid) + recall_score(y_valid, y_pred)}")
                        
                        fold_result_f1.append(f1_score(y_valid, y_pred))
                        fold_result_acc.append(model.score(X_valid, y_valid))
                        fold_result_rs.append(recall_score(y_valid, y_pred))
                        fold_result_auc.append(roc_auc_score(y_valid, y_pred))
                        fold_result_total.append(f1_score(y_valid, y_pred) + model.score(X_valid, y_valid) + recall_score(y_valid, y_pred))

                        print(f"base accuracy: {base_acc}")
                    
                    rs_list.append(sum(fold_result_rs)/len(fold_result_rs))
                    test_acc_list.append(sum(fold_result_acc)/len(fold_result_acc))
                    f1_list.append(sum(fold_result_f1)/len(fold_result_f1))
                    total_score_list.append(sum(fold_result_total)/len(fold_result_total))
                    auc_list.append(sum(fold_result_auc) / len(fold_result_auc))
                    lr_list.append(lr)
                    md_list.append(md)
                    cb_list.append(cb)
                    l1_list.append(l1)
                    print(fold_result_f1)
                    print(fold_result_acc)
                    print(fold_result_rs)
                    print(fold_result_total)
                    print(fold_result_auc)
    
    arg_max_f1 = np.argmax(f1_list)
    arg_max_acc = np.argmax(test_acc_list)
    arg_max_rs = np.argmax(rs_list)
    arg_max_auc = np.argmax(auc_list)
    arg_max_total = np.argmax(total_score_list)
    
    result_list = []
    
    
    f1_lr = lr_list[arg_max_f1]
    f1_md = md_list[arg_max_f1]
    f1_cb = cb_list[arg_max_f1]
    f1_l1 = l1_list[arg_max_f1]
    
    auc_lr = lr_list[arg_max_auc]
    auc_md = md_list[arg_max_auc]
    auc_cb = cb_list[arg_max_auc]
    auc_l1 = l1_list[arg_max_auc]

    acc_lr = lr_list[arg_max_acc]
    acc_md = md_list[arg_max_acc]
    acc_cb = cb_list[arg_max_acc]
    acc_l1 = l1_list[arg_max_acc]
    
    rs_lr = lr_list[arg_max_rs]
    rs_md = md_list[arg_max_rs]
    rs_cb = cb_list[arg_max_rs]
    rs_l1 = l1_list[arg_max_rs]

    total_lr = lr_list[arg_max_total]
    total_md = md_list[arg_max_total]
    total_cb = cb_list[arg_max_total]
    total_l1 = l1_list[arg_max_total]

    
    sm=SMOTE(random_state=2024)
    #ada = ADASYN(random_state=2024)
    X_final_train, y_final_train = sm.fit_resample(X_final_train, y_final_train)
    #X_final_train, y_final_train = ada.fit_resample(X_final_train, y_final_train)

    final_model_f1 = XGBClassifier(learning_rate=f1_lr, max_depth=f1_md, colsample_bytree=f1_cb, alpha=f1_l1, objective='binary:logistic',random_state=2024)
    final_model_f1.fit(X_final_train, y_final_train)
    final_model_auc = XGBClassifier(learning_rate=auc_lr, max_depth=auc_md, colsample_bytree=auc_cb, alpha=auc_l1, objective='binary:logistic',random_state=2024)
    final_model_auc.fit(X_final_train, y_final_train)
    final_model_acc = XGBClassifier(learning_rate=acc_lr, max_depth=acc_md, colsample_bytree=acc_cb, alpha=acc_l1, objective='binary:logistic',random_state=2024)
    final_model_acc.fit(X_final_train, y_final_train)
    final_model_rs = XGBClassifier(learning_rate=rs_lr, max_depth=rs_md, colsample_bytree=rs_cb, alpha=rs_l1, objective='binary:logistic',random_state=2024)
    final_model_rs.fit(X_final_train, y_final_train)
    final_model_total = XGBClassifier(learning_rate=total_lr, max_depth=total_md, colsample_bytree=total_cb, alpha=total_l1, objective='binary:logistic',random_state=2024)
    final_model_total.fit(X_final_train, y_final_train)
    
    y_final_pred_f1 = final_model_f1.predict(X_final_test)
    y_final_pred_auc = final_model_auc.predict(X_final_test)
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
    f1_best_auc_list.append(roc_auc_score(y_final_test,y_final_pred_f1))
    f1_best_test_acc_list.append(final_model_f1.score(X_final_test, y_final_test))
    f1_best_rs_list.append(recall_score(y_final_test, y_final_pred_f1))
    f1_best_matrix_list.append(confusion_matrix(y_final_test, y_final_pred_f1))
    f1_best_lr_list.append(f1_lr)
    f1_best_md_list.append(f1_md)
    f1_best_cb_list.append(f1_cb)
    f1_best_l1_list.append(f1_l1)

    auc_best_f1_list.append(f1_score(y_final_test, y_final_pred_auc))
    auc_best_auc_list.append(roc_auc_score(y_final_test, y_final_pred_auc))
    auc_best_test_acc_list.append(final_model_auc.score(X_final_test, y_final_test))
    auc_best_rs_list.append(recall_score(y_final_test, y_final_pred_auc))
    auc_best_matrix_list.append(confusion_matrix(y_final_test, y_final_pred_auc))
    auc_best_lr_list.append(auc_lr)
    auc_best_md_list.append(auc_md)
    auc_best_cb_list.append(auc_cb)
    auc_best_l1_list.append(auc_l1)
    
    acc_best_f1_list.append(f1_score(y_final_test, y_final_pred_acc))
    acc_best_auc_list.append(roc_auc_score(y_final_test, y_final_pred_acc))
    acc_best_test_acc_list.append(final_model_acc.score(X_final_test, y_final_test))
    acc_best_rs_list.append(recall_score(y_final_test, y_final_pred_acc))
    acc_best_matrix_list.append(confusion_matrix(y_final_test, y_final_pred_acc))
    acc_best_lr_list.append(acc_lr)
    acc_best_md_list.append(acc_md)
    acc_best_cb_list.append(acc_cb)
    acc_best_l1_list.append(acc_l1)
    
    rs_best_f1_list.append(f1_score(y_final_test, y_final_pred_rs))
    rs_best_auc_list.append(roc_auc_score(y_final_test, y_final_pred_rs))
    rs_best_test_acc_list.append(final_model_rs.score(X_final_test, y_final_test))
    rs_best_rs_list.append(recall_score(y_final_test, y_final_pred_rs))
    rs_best_matrix_list.append(confusion_matrix(y_final_test, y_final_pred_rs))
    rs_best_lr_list.append(rs_lr)
    rs_best_md_list.append(rs_md)
    rs_best_cb_list.append(rs_cb)
    rs_best_l1_list.append(rs_l1)

    total_best_f1_list.append(f1_score(y_final_test, y_final_pred_total))
    total_best_auc_list.append(roc_auc_score(y_final_test, y_final_pred_total))
    total_best_test_acc_list.append(final_model_total.score(X_final_test, y_final_test))
    total_best_rs_list.append(recall_score(y_final_test, y_final_pred_total))
    total_best_matrix_list.append(confusion_matrix(y_final_test, y_final_pred_total))
    total_best_lr_list.append(total_lr)
    total_best_md_list.append(total_md)
    total_best_cb_list.append(total_cb)
    total_best_l1_list.append(total_l1)

datadatadata = {'adr_name': adr_name_list,
                'base_acc': base_acc_list,
                'f1_test_acc': f1_best_test_acc_list,
                'f1_f1_score': f1_best_f1_list,
                'f1_auc_score' : f1_best_auc_list ,
                'f1_recall_scaore' : f1_best_rs_list,
                'f1_confusion_matrix' : f1_best_matrix_list,
                'f1_learning_rate': f1_best_lr_list,
                'f1_max_depth': f1_best_md_list,
                'f1_colsample_bytree': f1_best_cb_list,
                'f1_lambdal1': f1_best_l1_list,
                'base_acc': base_acc_list,
                'auc_test_acc': auc_best_test_acc_list,
                'auc_f1_score': auc_best_f1_list,
                'auc_auc_score' : auc_best_auc_list ,
                'auc_recall_scaore' : auc_best_rs_list,
                'auc_confusion_matrix' : auc_best_matrix_list,
                'auc_learning_rate': auc_best_lr_list,
                'auc_max_depth': auc_best_md_list,
                'auc_colsample_bytree': auc_best_cb_list,
                'auc_lambdal1': auc_best_l1_list,
                'base_acc': base_acc_list,
                'acc_test_acc': acc_best_test_acc_list,
                'acc_f1_score': acc_best_f1_list,
                'acc_auc_score' : acc_best_auc_list ,
                'acc_recall_scaore' : acc_best_rs_list,
                'acc_confusion_matrix' : acc_best_matrix_list,
                'acc_learning_rate': acc_best_lr_list,
                'acc_max_depth': acc_best_md_list,
                'acc_colsample_bytree': acc_best_cb_list,
                'acc_lambdal1': acc_best_l1_list,
                'base_acc': base_acc_list,
                'rs_test_acc': rs_best_test_acc_list,
                'rs_f1_score': rs_best_f1_list,
                'rs_auc_score' : rs_best_auc_list ,
                'rs_recall_scaore' : rs_best_rs_list,
                'rs_confusion_matrix' : rs_best_matrix_list,
                'rs_learning_rate': rs_best_lr_list,
                'rs_max_depth': rs_best_md_list,
                'rs_colsample_bytree': rs_best_cb_list,
                'rs_lambdal1': rs_best_l1_list,
                'base_acc': base_acc_list,
                'total_test_acc': total_best_test_acc_list,
                'total_f1_score': total_best_f1_list,
                'total_auc_score' : total_best_auc_list ,
                'total_recall_scaore' : total_best_rs_list,
                'total_confusion_matrix' : total_best_matrix_list,
                'total_learning_rate': total_best_lr_list,
                'total_max_depth': total_best_md_list,
                'total_colsample_bytree': total_best_cb_list,
                'total_lambdal1': total_best_l1_list,
    }

file_name = 'test_xgboost_0715_result_5fold_smote_smiles_only_32' + adr_name + '.csv'

for key, value in datadatadata.items():
    print(f"Length of {key}: {len(value)}")
# datadatadata.to_csv(file_name, index=False)
dfdfdf = pd.DataFrame(datadatadata)


dfdfdf.to_csv(file_name, index=False)
