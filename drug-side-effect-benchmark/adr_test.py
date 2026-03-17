import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from collections import Counter

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold

import sys
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)


df = pd. read_csv('./dataset_0722.csv')

print(df.columns)

smiles_list = df['SMILES'].tolist()

# Morgan fingerprints로 변환

# Morgan fingerprints로 변환 (유효하지 않은 SMILES 구조 처리)
fingerprints = []
invalid_smiles = []  # 유효하지 않은 SMILES 기록
for smile in smiles_list:
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fingerprints.append(fp)
    else:
        invalid_smiles.append(smile)

# 유효하지 않은 SMILES가 있는지 확인
if invalid_smiles:
    print(f"유효하지 않은 SMILES: {invalid_smiles}")

# 타니모토 유사도 계산 함수
def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# 타니모토 유사도 계산
similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
for i in range(len(fingerprints)):
    for j in range(i, len(fingerprints)):  # 대칭행렬이므로 절반만 계산
        sim = tanimoto_similarity(fingerprints[i], fingerprints[j])
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim

# 유사도 행렬을 데이터프레임으로 변환
similarity_df = pd.DataFrame(similarity_matrix)

print(similarity_df)

# 단백질 처리

exclude_columns =['Name', 'SMILES', 'DRUGBANK_ID', 'ApprovalDate', 'THROMBOCYTOPENIA']
df_protein = [col for col in df.columns if col not in exclude_columns]

print(df_protein)

target_data = df[df_protein]

# 스케일링
scaler = StandardScaler()
target_data_scaled = scaler.fit_transform(target_data)

# PCA를 이용한 차원 축소 (2D 임베딩)
pca = PCA(n_components=2)
target_embeddings = pca.fit_transform(target_data_scaled)

# 임베딩 결과를 데이터프레임에 추가
df['target_embedding_x'] = target_embeddings[:, 0]
df['target_embedding_y'] = target_embeddings[:, 1]

print(df[['target_embedding_x', 'target_embedding_y']])

# similarity_df의 행과 열의 이름을 df['Name'] 값으로 설정
similarity_df.columns = df['Name'].values
similarity_df.index = df['Name'].values

# df에서 'ApprovalDate', 'target_embedding_x', 'target_embedding_y', 'THROMBOCYTOPENIA' 컬럼만 선택
selected_columns = df[['ApprovalDate', 'target_embedding_x', 'target_embedding_y', 'THROMBOCYTOPENIA']]

# df의 'Name' 컬럼을 인덱스로 설정하여 similarity_df와 병합할 준비
selected_columns.index = df['Name']

# similarity_df에 새로운 컬럼을 붙임
final_df = pd.concat([similarity_df, selected_columns], axis=1)

# 결과 출력
print(final_df.columns)

# 'ApprovalDate' 컬럼을 datetime 형식으로 변환
df['ApprovalDate'] = pd.to_datetime(df['ApprovalDate'], errors='coerce')

# 'ApprovalDate' 컬럼을 기준으로 연도 순서로 정렬
final_df_sorted = final_df.sort_values(by='ApprovalDate')

# 결과 출력
print(final_df_sorted.columns)

#final_df_sorted.to_csv('./final_df_sorted.csv', index=False)





