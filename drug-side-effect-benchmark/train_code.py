import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, confusion_matrix, f1_score

import matplotlib.pyplot as plt
from datetime import datetime

# 분류 알고리즘 라이브러리
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LassoCV, RidgeClassifierCV, ElasticNetCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


import warnings
warnings.filterwarnings('ignore')

final_df_sorted = pd.read_csv('./final_df_sorted_factor_lda.csv')

# 1. 데이터 준비
# 'ApprovalDate' 컬럼 삭제
final_df_sorted = final_df_sorted.drop(columns=['ApprovalDate'])

# 'THROMBOCYTOPENIA'를 라벨로 분리하고 나머지 데이터를 입력으로 사용
labels = final_df_sorted['THROMBOCYTOPENIA'].values

features = final_df_sorted.drop(columns=['THROMBOCYTOPENIA']).values

# 데이터를 텐서로 변환
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# 데이터셋 및 DataLoader 생성
dataset = TensorDataset(features_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 2. Autoencoder 모델 정의
class Autoencoder(nn.Module):

    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            # nn.ReLU(),
            # nn.Linear(4, 2),
        )

        # 디코더
        self.decoder = nn.Sequential(
            # nn.Linear(2, 4),
            # nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 3. 모델 초기화 및 학습 준비
input_dim = features.shape[1]  # 입력 데이터의 차원
autoencoder = Autoencoder(input_dim)

criterion = nn.MSELoss()  # 손실 함수는 MSE 사용
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)  # Adam optimizer 사용

# 4. Autoencoder 학습
num_epochs = 100 ###########################################테스트
for epoch in range(num_epochs):
    for data, _ in dataloader:
        # Forward pass
        reconstructed = autoencoder(data)
        loss = criterion(reconstructed, data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 인코딩된 결과 추출 (압축된 데이터)
with torch.no_grad():
    compressed_features = autoencoder.encoder(features_tensor).numpy()

# 압축된 데이터 출력
print(compressed_features)

# 데이터를 학습 세트와 테스트 세트로 나눔
X_train, X_test, y_train, y_test = train_test_split(compressed_features, labels, test_size=0.2, random_state=42)


# 2. 하이퍼파라미터 공간 및 최적화 정의

def objective_xgb(params):
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss',verbosity  =0)
    auc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return {'loss': -auc, 'status': STATUS_OK}


space_xgb = {
    'max_depth': hp.choice('max_depth', range(3, 100)),
    'learning_rate': hp.uniform('learning_rate', 0.00001, 0.9),
    'n_estimators': hp.choice('n_estimators', range(10, 1000))
}


def objective_lgbm(params):
    model = LGBMClassifier(**params,verbose = -1)
    auc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return {'loss': -auc, 'status': STATUS_OK}


space_lgbm = {
    'num_leaves': hp.choice('num_leaves', range(5, 100)),
    'learning_rate': hp.uniform('learning_rate', 0.00001, 0.9),
    'n_estimators': hp.choice('n_estimators', range(10, 1000))
}


def objective_dt(params):
    model = DecisionTreeClassifier(**params)
    auc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return {'loss': -auc, 'status': STATUS_OK}


space_dt = {
    'max_depth': hp.choice('max_depth', range(3, 100)),
    'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 1.0)
}


def objective_adaboost(params):
    model = AdaBoostClassifier(**params)
    auc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return {'loss': -auc, 'status': STATUS_OK}


space_adaboost = {
    'n_estimators': hp.choice('n_estimators', range(10, 1000)),
    'learning_rate': hp.uniform('learning_rate', 0.00001, 0.9)
}

def objective_elastic_net(params):
    model = ElasticNetCV(l1_ratio=params['l1_ratio'], alphas=[params['alpha']], cv=5,verbose = 0)
    auc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return {'loss': -auc, 'status': STATUS_OK}

space_elastic_net = {
    'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
    'alpha': hp.uniform('alpha', 0.0001, 1.0)
}

def objective_catboost(params):
    model = CatBoostClassifier(**params)
    auc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return {'loss': -auc, 'status': STATUS_OK}


space_catboost = {
    'iterations': hp.choice('iterations', range(10, 1000)),
    'learning_rate': hp.uniform('learning_rate', 0.00001, 0.9),
    'depth': hp.choice('depth', range(3, 16))
}


def objective_gbm(params):
    model = GradientBoostingClassifier(**params)
    auc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return {'loss': -auc, 'status': STATUS_OK}


space_gbm = {
    'n_estimators': hp.choice('n_estimators', range(10, 1000)),
    'learning_rate': hp.uniform('learning_rate', 0.00001, 0.9),
    'max_depth': hp.choice('max_depth', range(3, 100))
}

# 3. 베이지안 최적화 실행

trials_xgb = Trials()
best_xgb = fmin(fn=objective_xgb, space=space_xgb, algo=tpe.suggest, max_evals=1000, trials=trials_xgb)

trials_lgbm = Trials()
best_lgbm = fmin(fn=objective_lgbm, space=space_lgbm, algo=tpe.suggest, max_evals=1000, trials=trials_lgbm)

trials_dt = Trials()
best_dt = fmin(fn=objective_dt, space=space_dt, algo=tpe.suggest, max_evals=1000, trials=trials_dt)

trials_elastic_net = Trials()
best_elastic_net = fmin(fn=objective_elastic_net, space=space_elastic_net, algo=tpe.suggest, max_evals=1000,
                        trials=trials_elastic_net)

trials_adaboost = Trials()
best_adaboost = fmin(fn=objective_adaboost, space=space_adaboost, algo=tpe.suggest, max_evals=1000, trials=trials_adaboost)

trials_catboost = Trials()
best_catboost = fmin(fn=objective_catboost, space=space_catboost, algo=tpe.suggest, max_evals=1000, trials=trials_catboost)

trials_gbm = Trials()
best_gbm = fmin(fn=objective_gbm, space=space_gbm, algo=tpe.suggest, max_evals=1000, trials=trials_gbm)


# 4. 최적화된 모델로 학습 및 ROC-AUC 계산
models = {
    'XGBoost': (XGBClassifier(**best_xgb, use_label_encoder=False, eval_metric='logloss'), best_xgb),
    'LightGBM': (LGBMClassifier(**best_lgbm, verbose=-1), best_lgbm),
    'Decision Tree': (DecisionTreeClassifier(**best_dt), best_dt),
    'Elastic Net': (ElasticNetCV(l1_ratio=best_elastic_net['l1_ratio'], alphas=[best_elastic_net['alpha']], cv=5), best_elastic_net),
    'AdaBoost': (AdaBoostClassifier(**best_adaboost), best_adaboost),
    'CatBoost': (CatBoostClassifier(**best_catboost), best_catboost),
    'Gradient Boosting': (GradientBoostingClassifier(**best_gbm), best_gbm)
}
# 5. 모델 학습 및 성능 평가
model_results = {}

for model_name, (model, best_params) in models.items():
    model.fit(X_train, y_train)

    if model_name == 'Elastic Net':
        y_prob = model.predict(X_test)
        y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    # 성능 평가
    auc_score = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # 결과 저장
    model_results[model_name] = {
        'AUC': auc_score,
        'Accuracy': accuracy,
        'AUPRC': auprc,
        'Confusion Matrix': cm,
        'f1score': f1,
        'Best Parameters': best_params
    }

    # 결과 출력
    print(f'{model_name} AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, AUPRC: {auprc:.4f}, F1_score: {f1:.4f}')
    print(f'{model_name} Confusion Matrix:\n{cm}\n')
    print(f'{model_name} Best Parameters: {best_params}\n')


results_df = pd.DataFrame(model_results).T
# results_df = results_df.drop(columns=['Confusion Matrix'])  # 혼동 행렬은 CSV에 포함하지 않음

# 날짜 기반 파일명 생성
current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f"model_results_{current_date}.csv"
results_df.to_csv(file_name)

print(f'Results saved to {file_name}')

# 6. ROC-AUC 그래프 시각화
plt.figure(figsize=(10, 8))

for model_name, (model, best_params) in models.items():
    if model_name == 'Elastic Net':
        y_prob = model.predict(X_test)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Comparison of Different Models (Optimized)')
plt.legend(loc='lower right')
plt.savefig(f'ROC-AUC_{current_date}.jpg', format='jpeg')
plt.show()

plt.figure(figsize=(10, 8))

for model_name, (model, best_params) in models.items():
    if model_name == 'Elastic Net':
        y_prob = model.predict(X_test)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f'{model_name} (AUPRC = {model_results[model_name]["AUPRC"]:.4f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison of Different Models (Optimized)')
plt.legend(loc='lower left')
plt.savefig(f'AUPRC_{current_date}.jpg', format='jpeg')
plt.show()
