import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from xgboost import XGBRFClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


from collections import Counter

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import argparse
from yaml import parse
from tqdm import tqdm
z
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

# input_data1 = pd.read_csv("C:\\Users\\Juno Kwon\\OneDrive\\바탕 화면\\v2_data(타니모토+타겟 같이 넣을 때)_약물개수809\\additional_adrs_with_fp.csv")
# input_data2 = pd.read_csv("C:\\Users\\Juno Kwon\\OneDrive\\바탕 화면\\v2_data(타니모토+타겟 같이 넣을 때)_약물개수809\\target_data_v2.csv")
output_data = pd.read_csv('label.csv')  # 부작용 선택

def cal_tanimoto(smile1, smile2):

    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    return similarity

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
smile_data = pd.read_csv('smlies_data_v2.csv')
smile = smile_data['smiles']
morgan_all = pd.DataFrame([smiles2morgan(s) for s in smile])
print(morgan_all)

for drug1 in smile:
        for drug2 in smile:
            criterion = cal

def smiles2morgan(s, radius=4, nBits=80):

    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
        features = np.zeros((nBits,))

    return features
# data1을 반으로 나눕니다.
half_length = len(data1) // 2
first_half_data1 = data1[:half_length, :]
second_half_data1 = data1[half_length:, :]

# 첫 번째 반에 대한 Tanimoto 유사도를 계산합니다.
first_half_similarity_matrix = create_similarity_matrix(first_half_data1)

# 두 번째 반은 그대로 두기 때문에 그 부분은 원래 데이터 그대로 사용합니다.
second_half_similarity_matrix = second_half_data1

# 두 부분을 concatenate하여 전체 유사도 행렬을 만듭니다.
print("First Half Similarity Matrix Shape:", first_half_similarity_matrix.shape)
print("Second Half Similarity Matrix Shape:", second_half_similarity_matrix.shape)
similarity_matrix = np.concatenate((first_half_similarity_matrix, second_half_similarity_matrix), axis=0)

data2 = pd.read_csv('target_action_data222.csv')
data2 = data2.drop("Unnamed: 0", axis = 1)
data2 = data2.to_numpy()

data2_1 = ((data2 * data2) + data2)/2 #라그랑주 보간법: 1은1, 나머지 0
data2_2 = ((data2 * data2) - data2)/2 #-1은1, 나머지 0
data2_3 = 1 - (data2 * data2) #0은1, 나머지 0

# data = np.concatenate((data1, data2_1, data2_2, data2_3), axis=1)

# one_hot_vectors = create_similarity_matrix(data1)
one_hot_vectors=np.concatenate((data2_1,data2_2,data2_3),axis=1)
# half_length = len(data1) // 2
# similarity_matrix = create_similarity_matrix(data1[:half_length  ,:])
# similarity_matrix = one_hot_vectors

print("Similarity Matrix:")
print(similarity_matrix)

output_data = output_data.drop("drug_id", axis=1)

output_data.columns = ["".join(e for e in col if e.isalnum()) for col in output_data.columns]
label = output_data.columns




parser = argparse.ArgumentParser()
parser.add_argument('--input-size', type=int, default=2083)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--target', type=str, default="ratio", choices=["ratio", "top1", "top10", "bot50"])
parser.add_argument('--model', type=str, default="trans", choices=["trans", "lstm", "gru"])
parser.add_argument('--prediction-years', type=int, default=1)
parser.add_argument('--train-years', type=int, default=15)

# Transformer Hyper Parameter
parser.add_argument('--trans-head', type=int, default=2)
parser.add_argument('--trans-layer', type=int, default=1)
parser.add_argument('--trans-dropout', type=float, default=0.1)
parser.add_argument('--trans-dim', type=int, default=128)

args = parser.parse_args()

class medicineDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        x = self.x[i,:]
        y = self.y[i] # i번째 약물
        return x, y

X = one_hot_vectors
print(X)
y = output_data.to_numpy()[:,114] #0번째 부작용=Anaemia
print(y)


X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

dataset_auto = medicineDataset(X, y)

class AUTO_ENCODER1(torch.nn.Module):
    def __init__(self,args):
        super(AUTO_ENCODER1, self).__init__()

        self.args = args

        self.encoder = nn.Sequential(
            nn.Linear(1024, 516),
            nn.LeakyReLU(),
            nn.Linear(516, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            # nn.LeakyReLU(),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(),
            # nn.Linear(64, 32),
            # nn.LeakyReLU(),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(),
            # nn.Linear(16, 8),
            # nn.LeakyReLU(),
            # nn.Linear(8, 4),
            # nn.LeakyReLU(),
            # nn.Linear(4, 2),
            # nn.LeakyReLU(),
            # nn.Linear(128, 64),
            )
        
        self.decoder = nn.Sequential(
            # nn.LeakyReLU(),
            # nn.Linear(2, 4),
            # nn.LeakyReLU(),
            # nn.Linear(64, 128),
            # nn.Linear(4, 8),
            # nn.LeakyReLU(),
            # nn.Linear(8, 16),
            # nn.LeakyReLU(),
            # nn.Linear(16, 32),
            # nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 516),
            # nn.Linear(404, 512),
            nn.LeakyReLU(),
            nn.Linear(516, 1024),
            # nn.LeakyReLU(),
            # nn.Linear(750, 1500),
            # nn.LeakyReLU(),
            # nn.Linear(1500, 404),
        )
    
    def forward(self, x):
        eco = self.encoder(x.float())
        dco = self.decoder(eco)

        return eco, dco

class AUTO_ENCODER(torch.nn.Module):
    def __init__(self,args):
        super(AUTO_ENCODER, self).__init__()

        self.args = args

        self.encoder = nn.Sequential(
            nn.Linear(3435, 1500),
            nn.LeakyReLU(),
            nn.Linear(1500, 750),
            nn.LeakyReLU(),
            nn.Linear(750, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            # nn.LeakyReLU(),
            # nn.Linear(32, 16),
            # nn.LeakyReLU(),
            # nn.Linear(16, 8),
            # nn.LeakyReLU(),
            # nn.Linear(8, 4),
            # nn.LeakyReLU(),
            # nn.Linear(4, 2),
            # nn.LeakyReLU(),
            # nn.Linear(128, 64),
            )
        
        self.decoder = nn.Sequential(
            # nn.LeakyReLU(),
            # nn.Linear(2, 4),
            # nn.LeakyReLU(),
            # nn.Linear(64, 128),
            # nn.Linear(4, 8),
            # nn.LeakyReLU(),
            # nn.Linear(8, 16),
            # nn.LeakyReLU(),
            # nn.Linear(16, 32),
            # nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 750),
            nn.LeakyReLU(),
            nn.Linear(750, 1500),
            nn.LeakyReLU(),
            nn.Linear(1500, 3435),
        )
    
    def forward(self, x):
        eco = self.encoder(x)
        dco = self.decoder(eco)

        return eco, dco
    
model_auto = AUTO_ENCODER(args)

optimizer_auto = torch.optim.Adam(model_auto.parameters(), lr=1e-5)

dataloader_auto = DataLoader(dataset_auto, batch_size=args.batch_size, shuffle = False)

train_loss = []

def train_model_auto(data_loader, model, criterion, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in tqdm(data_loader):
        eco_output, dco_output = model(X)

        loss = criterion(dco_output, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches

    train_loss.append(avg_loss)


    print(f"Train Loss: {avg_loss}")

loss_mse = torch.nn.MSELoss()
# loss_bce = torch.nn.BCELoss()
loss_bce = torch.nn.BCEWithLogitsLoss()

#target_action_ae
for epoch in range(100):
    print(f"Epoch {epoch}\n ---------")
    train_model_auto(dataloader_auto, model_auto, loss_mse, optimizer_auto)
    print("")

half_length+=1
print("similarity_matrix shape:", similarity_matrix.shape)
print("data1[half_length:, :] shape:", data1[half_length:, :].shape)
X1 = np.concatenate((similarity_matrix, data1[half_length:, :]), axis=0)
X1=similarity_matrix
y1=output_data.to_numpy()[:,114]
X1 = torch.FloatTensor(X1)
dataset_auto1=medicineDataset(X1,y1)

model_auto1 = AUTO_ENCODER1(args)

optimizer_auto1 = torch.optim.Adam(model_auto1.parameters(), lr=1e-5)

dataloader_auto1 = DataLoader(dataset_auto1, batch_size=args.batch_size, shuffle = False)

train_loss = []
#simlarity_ae
for epoch in range(100):
    print(f"Epoch {epoch}\n ---------")
    train_model_auto(dataloader_auto1, model_auto1, loss_mse, optimizer_auto1)
    print("")

X_eco1,X_dco1 = model_auto1(X1)
X_eco, X_dco = model_auto(X)
# X_eco1 = np.concatenate((X_eco1.detach().numpy(), data1[half_length:, :]), axis=0)
X_concat = np.concatenate((X_eco.detach().numpy(),X_eco1.detach().numpy()),axis=0)
Y = pd.DataFrame(output_data[label[114]])
X = pd.DataFrame(X_concat)

#print(X)
#print(Y)

idx = list(range(X.shape[0]))
train_idx, valid_idx = train_test_split(idx, test_size=0.3, random_state=2023)
print(">>>> # of Train data : {}".format(len(train_idx)))
print(">>>> # of valid data : {}".format(len(valid_idx)))
# Ensure train_idx is within the bounds of Y DataFrame
train_idx = [i for i in train_idx if i < len(Y)]
valid_idx = [i for i in valid_idx if i < len(Y)]
print(">>>> # of Train data Y : {}".format(Counter(Y.iloc[train_idx])))
print(">>>> # of valid data Y : {}".format(Counter(Y.iloc[valid_idx])))

## SMOTE 구현 -----------구글링 코드
# from imblearn.under_sampling import AllKNN
# from imblearn.over_sampling import SMOTE

# AK = AllKNN(allow_minority=True)
# sm = SMOTE(random_state=42)

# 중요한 파라미터 위주로 조절한다

# n_estimators
n_tree = [20, 30, 40, 50, 60, 80]
# n_tree = [20, 30, 40, 50]
# n_tree = [44,46,45]
# n_tree = [24]
# n_tree = np.arange(10,1000,10)

# learning_rate
l_rate = [0.05,0.01]
# l_rate = [1.1,1.2,1.3,1]
# l_rate = [0.35]

# l_rate = np.arange(0.1, 100, 0.1)

# max_depth
m_depth = [3, 5, 7, 10, 20]
# m_depth = [1,2,3,4]
# m_depth = [2]
# m_depth = [8]
# m_depth = np.arange(1,100,1)

# reg_alpha
L1_norm = [0.1, 0.3, 0.5, 0.8]
# L1_norm = [0.49]
# L1_norm = [0.11,0.1,0.09]
# L1_norm=np.arange(0,1,0.05,0.001)

#nothing    = learning_rate=0.1 LGBMClassifier(max_depth=3, n_estimators=30, objective='cross_entropy',random_state=119, reg_alpha=0.5)                         Train:83% Test:84%
#under sm   = learning_rate=0.2 LGBMClassifier(learning_rate=0.2, max_depth=2, n_estimators=24, objective='cross_entropy', random_state=119, reg_alpha=0.49)    Train:82%  Test:83%
#over sm    = learning_rate=0.4 LGBMClassifier(learning_rate=0.4, max_depth=3, n_estimators=20,objective='cross_entropy', random_state=119, reg_alpha=0.3)      Train:84%  Test:81%

##################

# Modeling
save_n = []
save_l = []
save_m = []
save_L1 = []
f1_score_ = []

save_n1 = []
save_la = []
save_m1 = []
save_L11 = []
f1_score_1 = []

cnt = 0

from tqdm import tqdm

for n in n_tree:
    for l in l_rate:
        for m in m_depth:
            for L1 in tqdm(L1_norm):
                print(">>> {} <<<".format(cnt))
                cnt += 1
                print("n_estimators : {}, learning_rate : {}, max_depth : {}, reg_alpha : {}".format(n, l, m, L1))
                # X_train_over, y_train_over = sm.fit_resample(X.iloc[train_idx, :], Y.iloc[train_idx])
                model = xgb.XGBRFClassifier(n_estimators=n, learning_rate=l, max_depth=m, reg_alpha=L1, n_jobs=-1)
                model.fit(X.iloc[train_idx,:], Y.iloc[train_idx])
                # print(X.iloc[train_idx,:])
                # print("\n\n")
                # print(Y.iloc[train_idx])
                # model.fit(X_train_over, y_train_over)

                # Train Acc
                y_pre_train = model.predict(X.iloc[train_idx, :])
                # print(y_pre_train)
                cm_train = confusion_matrix(Y.iloc[train_idx], y_pre_train)
                # print("Train Confusion Matrix")
                # print(cm_train)
                # print("Train Acc : {}".format((cm_train[0, 0] + cm_train[1, 1]) / cm_train.sum()))
                # print("Train F1-Score : {}".format(f1_score(Y.iloc[train_idx], y_pre_train)))
                # print("Train AUROC : {}".format(roc_auc_score(Y.iloc[train_idx], y_pre_train)))

                # Test Acc
                y_pre_test = model.predict(X.iloc[valid_idx, :])
                cm_test = confusion_matrix(Y.iloc[valid_idx], y_pre_test)
                print("Test Confusion Matrix")
                print(cm_test)
                print("TesT Acc : {}".format((cm_test[0, 0] + cm_test[1, 1]) / cm_test.sum()))
                print("Test F1-Score : {}".format(f1_score(Y.iloc[valid_idx], y_pre_test)))
                print("Test AUROC : {}".format(roc_auc_score(Y.iloc[valid_idx], y_pre_test)))
                print("-----------------------------------------------------------------------")
                print("-----------------------------------------------------------------------")
                save_n.append(n)
                save_l.append(l)
                save_m.append(m)
                save_L1.append(L1)
                f1_score_.append(f1_score(Y.iloc[valid_idx], y_pre_test))

                # joblib.dump(model, './LightGBM_model/Result_{}_{}_{}_{}_{}.pkl'.format(n, l, m, L1, round(save_acc[-1], 4)))
                # gc.collect()

best_model = model = xgb.XGBRFClassifier(n_estimators=save_n[np.argmax(f1_score_)], learning_rate=save_l[np.argmax(f1_score_)],
                            max_depth=save_m[np.argmax(f1_score_)], reg_alpha=save_L1[np.argmax(f1_score_)], n_jobs=-1)

print(f'learning_rate={save_l[np.argmax(f1_score_)]}')
print(best_model)
best_model.fit(X.iloc[train_idx, :], Y.iloc[train_idx])

# Train Acc
y_pre_train = best_model.predict(X.iloc[train_idx, :])
cm_train = confusion_matrix(Y.iloc[train_idx], y_pre_train)
print("Train Confusion Matrix")
print(cm_train)
print("Train Acc : {}".format((cm_train[0, 0] + cm_train[1, 1]) / cm_train.sum()))
print("Train F1-Score : {}".format(f1_score(Y.iloc[train_idx], y_pre_train)))
print("Train AUROC : {}".format(roc_auc_score(Y.iloc[train_idx], y_pre_train)))

# Test Acc
y_pre_test = best_model.predict(X.iloc[valid_idx, :])
cm_test = confusion_matrix(Y.iloc[valid_idx], y_pre_test)
print("Test Confusion Matrix")
print(cm_test)
print("TesT Acc : {}".format((cm_test[0, 0] + cm_test[1, 1]) / cm_test.sum()))
print("Test F1-Score : {}".format(f1_score(Y.iloc[valid_idx], y_pre_test)))
print("Test AUROC : {}".format(roc_auc_score(Y.iloc[valid_idx], y_pre_test)))