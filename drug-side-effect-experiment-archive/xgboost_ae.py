import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 데이터 불러오기
data = pd.read_csv('simliarity_action.csv')
data = data.iloc[:, 1:]
data.columns = ["".join(e for e in col if e.isalnum()) for col in data.columns]

label1 = pd.read_csv('lavel_data_with_target_action.csv')
label1 = label1.iloc[:, 1:]
label1.columns = ["".join(e for e in col if e.isalnum()) for col in label1.columns]

# 레이블 및 입력 데이터 설정
input_data = label1.columns[1]
Y = pd.DataFrame(label1[input_data])
X = data
X.columns = [f"feature_{i}" for i in range(X.shape[1])]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 오토인코더 모델 정의
autoencoder = Sequential([
    Dense(10, activation='relu', input_shape=(X.shape[1],)),  # 작은 차원으로 압축
    Dense(X.shape[1], activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

# 오토인코더 모델 학습
autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, shuffle=True)

# 인코더 모델 추출 (압축된 표현을 얻기 위해)
encoder = Sequential(autoencoder.layers[:1])

# 데이터를 인코더에 통과시켜 저차원 표현 얻기
compressed_train_data = encoder.predict(X_train)
compressed_test_data = encoder.predict(X_test)

# 압축된 데이터의 모양 확인
print('Shape of compressed train data:', compressed_train_data.shape)
print('Shape of compressed test data:', compressed_test_data.shape)

# XGBoost 데이터셋 생성
dtrain_compressed = xgb.DMatrix(compressed_train_data, label=y_train)
dtest_compressed = xgb.DMatrix(compressed_test_data, label=y_test)

# 파라미터 그리드 정의
param_grid_compressed = {
    'max_depth': [3, 5, 10, 15, 20, 25],
    'subsample': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}

best_accuracy_compressed = 0.0
best_params_compressed = {}

# 파라미터 그리드 순회하며 최적의 모델 찾기
for params in product(*param_grid_compressed.values()):
    params = dict(zip(param_grid_compressed.keys(), params))
    num_round = 100  # num_round를 정의

    # XGBoost 모델 설정
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'

    # 모델 학습
    bst_compressed = xgb.train(params, dtrain_compressed, num_round, [(dtrain_compressed, 'train'), (dtest_compressed, 'test')])

    # 예측
    pred_test_compressed = bst_compressed.predict(dtest_compressed)
    pred_test_binary_compressed = [1 if x >= 0.5 else 0 for x in pred_test_compressed]

    # 정확도 계산
    test_accuracy_compressed = accuracy_score(y_test, pred_test_binary_compressed)

    if test_accuracy_compressed > best_accuracy_compressed:
        best_accuracy_compressed = test_accuracy_compressed
        best_params_compressed = params

print('Best Test Accuracy (Compressed):', best_accuracy_compressed)
print('Best Parameters (Compressed):', best_params_compressed)
