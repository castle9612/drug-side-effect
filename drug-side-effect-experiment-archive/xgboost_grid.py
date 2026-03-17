import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product

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

# 파라미터 그리드 정의
param_grid = {
    'max_depth': [3,5,10,15,20,25],
    'subsample': [0.1,0.3,0.5,0.7,0.9, 1.0],
    'colsample_bytree': [0.1,0.3,0.5,0.7,0.9, 1.0]
}

best_accuracy = 0.0
best_params = {}

# 파라미터 그리드 순회하며 최적의 모델 찾기
for params in product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), params))
    num_round = 100  # num_round를 정의
     
    # XGBoost 데이터셋 생성
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost 모델 설정
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'

    # 모델 학습
    bst = xgb.train(params, dtrain, num_round, [(dtrain, 'train'), (dtest, 'test')])

    # 예측
    pred_test = bst.predict(dtest)
    pred_test_binary = [1 if x >= 0.5 else 0 for x in pred_test]

    # 정확도 계산
    test_accuracy = accuracy_score(y_test, pred_test_binary)

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_params = params

print('Best Test Accuracy:', best_accuracy)
print('Best Parameters:', best_params)
