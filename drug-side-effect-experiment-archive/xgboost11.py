import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# XGBoost 데이터셋 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost 모델 설정
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# 모델 학습
num_round = 100
bst = xgb.train(params, dtrain, num_round, [(dtrain, 'train'), (dtest, 'test')])

# 예측
pred_train = bst.predict(dtrain)
pred_train_binary = [1 if x >= 0.5 else 0 for x in pred_train]
pred_test = bst.predict(dtest)
pred_test_binary = [1 if x >= 0.5 else 0 for x in pred_test]

# 정확도 계산
train_accuracy = accuracy_score(y_train, pred_train_binary)
test_accuracy = accuracy_score(y_test, pred_test_binary)

print('Train Accuracy:', train_accuracy)
print('Test Accuracy:', test_accuracy)
