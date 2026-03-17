import numpy as np
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 알잘딱 데이터 넣으셈

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP 
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
mlp_model.fit(X_train, y_train)

# lgb
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# mlp & lgb 예측 
mlp_preds = mlp_model.predict(X_test)
lgb_preds = lgb_model.predict(X_test)

# twotower모델
stacked_preds = np.column_stack((mlp_preds, lgb_preds))

# Stacking 모델 
stacking_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000)
stacking_model.fit(stacked_preds, y_test)

# twotower 모델 예측
stacked_preds_test = np.column_stack((mlp_model.predict(X_test), lgb_model.predict(X_test)))
final_predictions = stacking_model.predict(stacked_preds_test)

# 성능 평가
accuracy = accuracy_score(y_test, final_predictions)
print("twotower accuracy:", accuracy)