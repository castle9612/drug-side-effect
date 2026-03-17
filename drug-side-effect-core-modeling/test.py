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
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 32),
            )
        
        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(32, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 400),
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
smile_data_year = smile_data_year.head(400)

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

from sklearn.model_selection import StratifiedKFold
# k_fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2024)

from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

X = X_eco.detach().numpy()
X = pd.DataFrame(X)

for adr_name in adr_name_list:
    base_acc = calculate_base_acc(label_data, adr_name)
    y = label_data[adr_name].to_numpy()

    X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(X, y, stratify=y, test_size=0.3)

    sm = SMOTE(random_state=2024)
    X_final_train, y_final_train = sm.fit_resample(X_final_train, y_final_train)

    clf = TabNetClassifier(
        n_steps=3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 2e-2},
        scheduler_fn=None,
        clip_value=2.0,
        momentum=0.3,
        scheduler_params={"step_size": 50, "gamma": 0.9},
        mask_type='sparsemax',
        device_name='cpu',
        n_d=8,
        n_a=8,
        seed=100,
        lambda_sparse=1e-3,
        epsilon=1e-15
    )

    clf.fit(
        X_final_train.values, y_final_train.ravel(),
        eval_set=[(X_final_train.values, y_final_train.ravel()), (X_final_test.values, y_final_test.ravel())],
        eval_name=['train', 'val'],
        eval_metric=['auc', 'logloss'],
        max_epochs=10000,
        patience=200,
        batch_size=16,
        virtual_batch_size=16,
        num_workers=0,
        drop_last=False
    )

    # 모델 학습 결과 시각화
    plt.plot(clf.history['loss'], marker='o', label='train')
    plt.plot(clf.history['val_logloss'], marker='x', label='val')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    plt.plot(clf.history['train_auc'], marker='o', label='train')
    plt.plot(clf.history['val_auc'], marker='x', label='val')
    plt.title('AUC per epoch')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    best_epoch = np.argmax(clf.history['val_auc'])
    best_val_auc = clf.history['val_auc'][best_epoch]
    best_train_auc = clf.history['train_auc'][best_epoch]

    print(f"Best Epoch: {best_epoch+1}, Best train AUC: {best_train_auc:.4f}, Best Val AUC: {best_val_auc:.4f}")

    # F1 Score 및 혼동 행렬 출력
    y_pred = clf.predict(X_final_test.values)

    f1 = f1_score(y_final_test, y_pred, average='macro')
    cm = confusion_matrix(y_final_test, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    f1_scores_train = []
    f1_scores_val = []

    for epoch in range(len(clf.history['train_auc'])):
        clf.fit(
            X_final_train.values, y_final_train.ravel(),
            eval_set=[(X_final_train.values, y_final_train.ravel()), (X_final_test.values, y_final_test.ravel())],
            eval_name=['train', 'val'],
            eval_metric=['auc', 'logloss'],
            max_epochs=epoch + 1,
            patience=10,
            batch_size=16,
            virtual_batch_size=16,
            num_workers=0,
            drop_last=False
        )
        
        y_pred_train = clf.predict(X_final_train.values)
        y_pred_val = clf.predict(X_final_test.values)
        
        f1_train = f1_score(y_final_train, y_pred_train, average='macro')
        f1_val = f1_score(y_final_test, y_pred_val, average='macro')
        
        f1_scores_train.append(f1_train)
        f1_scores_val.append(f1_val)

    plt.plot(f1_scores_train, marker='o', label='train')
    plt.plot(f1_scores_val, marker='x', label='val')
    plt.title('F1 Score per epoch')
    plt.ylabel('F1 Score')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()