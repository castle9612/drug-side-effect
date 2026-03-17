import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from lightgbm import LGBMClassifier
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from imblearn.under_sampling import AllKNN
from imblearn.over_sampling import SMOTE
from keras.models import Model
from keras.layers import Input, Dense

# 데이터 불러오기
data = pd.read_csv('simliarity_action.csv')
data = data.iloc[:, 1:]
data.columns = ["".join(e for e in col if e.isalnum()) for col in data.columns]
label1 = pd.read_csv('lavel_data_with_target_action.csv')
label1 = label1.iloc[:,1:]
label1.colums = ["".join(e for e in col if e.isalnum()) for col in label1.columns]
label = label1.columns
input_data = label[1]
Y = pd.DataFrame(label1[input_data])
X = data
X.columns = [f"feature_{i}" for i in range(X.shape[1])]

# Train, validation 데이터 분리
idx = list(range(X.shape[0]))
train_idx, valid_idx = train_test_split(idx, test_size=0.2, random_state=10000)

# Autoencoder를 통한 데이터 인코딩
autoencoder_data = X.values
input_dim = autoencoder_data.shape[1]
encoding_dim = 707  # 인코딩 차원

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(autoencoder_data, autoencoder_data, epochs=50, batch_size=256, shuffle=True, validation_split=0.1)

# 데이터 인코딩
encoded_features = autoencoder.predict(autoencoder_data)
X_encoded = pd.DataFrame(encoded_features, columns=[f"encoded_feature_{i}" for i in range(encoding_dim)])
X_encoded = X_encoded.iloc[train_idx, :]

# SMOTE 적용
AK = AllKNN(allow_minority=True)
sm = SMOTE(random_state=42)
# X_train_over, y_train_over = sm.fit_resample(X_encoded, Y.iloc[train_idx])

# 중요한 파라미터 위주로 조절한다

# n_estimators
# n_tree = [20, 30, 40, 50, 60]
n_tree = [10,15,20,25]
# n_tree = [24]
# n_tree = np.arange(10,1000,10)

# learning_rate
# l_rate = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
l_rate = [0.6,0.7,0.8]
# l_rate = [0.35]

# l_rate = np.arange(0.1, 100, 0.1)

# max_depth
# m_depth = [3, 5, 7, 10, 20, 30]
m_depth = [8,9,10]
# m_depth = [8]
# m_depth = np.arange(1,100,1)

# reg_alpha
# L1_norm = [0.1, 0.3, 0.5, 0.8]
L1_norm = [0.4,0.5,0.6,0.7]
# L1_norm = [0.15]
# L1_norm=np.arange(0,1,0.01)


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
                X_train_over, y_train_over = AK.fit_resample(X.iloc[train_idx, :], Y.iloc[train_idx])
                model = LGBMClassifier(n_estimators=n, learning_rate=l,
                                       max_depth=m, reg_alpha=L1,
                                       n_jobs=-1, objective='cross_entropy')
                model.fit(X.iloc[train_idx,:], Y.iloc[train_idx])
                # model.fit(X_train_over, y_train_over)

                # Train Acc
                y_pre_train = model.predict(X.iloc[train_idx, :])
                cm_train = confusion_matrix(Y.iloc[train_idx], y_pre_train)
                print("Train Confusion Matrix")
                print(cm_train)
                print("Train Acc : {}".format((cm_train[0, 0] + cm_train[1, 1]) / cm_train.sum()))
                print("Train F1-Score : {}".format(f1_score(Y.iloc[train_idx], y_pre_train)))

                # Test Acc
                y_pre_test = model.predict(X.iloc[valid_idx, :])
                cm_test = confusion_matrix(Y.iloc[valid_idx], y_pre_test)
                # print("Test Confusion Matrix")
                # print(cm_test)
                # print("TesT Acc : {}".format((cm_test[0, 0] + cm_test[1, 1]) / cm_test.sum()))
                # print("Test F1-Score : {}".format(f1_score(Y.iloc[valid_idx], y_pre_test)))
                cm_test = confusion_matrix(Y.iloc[valid_idx], y_pre_test)
                print("Test Confusion Matrix")
                print(cm_test)

                # Check if the confusion matrix is 1x1
                if cm_test.shape[0] == 1 and cm_test.shape[1] == 1:
                    true_positive = cm_test[0, 0]
                    total_samples = cm_test.sum()
                    accuracy = true_positive / total_samples
                    f1 = 2 * (true_positive / total_samples)  # F1-score for binary classification with one class

                    print("Test Acc : {}".format(accuracy))
                    print("Test F1-Score : {}".format(f1))
                else:
                    # Compute accuracy and F1-score for multi-class classification
                    accuracy = (cm_test.diagonal().sum()) / cm_test.sum()
                    f1 = f1_score(Y.iloc[valid_idx], y_pre_test, average='weighted')

                    print("Test Acc : {}".format(accuracy))
                    print("Test F1-Score : {}".format(f1))
                print("-----------------------------------------------------------------------")
                print("-----------------------------------------------------------------------")
                save_n.append(n)
                save_l.append(l)
                save_m.append(m)
                save_L1.append(L1)
                f1_score_.append(f1_score(Y.iloc[valid_idx], y_pre_test))

                # joblib.dump(model, './LightGBM_model/Result_{}_{}_{}_{}_{}.pkl'.format(n, l, m, L1, round(save_acc[-1], 4)))
                # gc.collect()

best_model = LGBMClassifier(n_estimators=save_n[np.argmax(f1_score_)], learning_rate=save_l[np.argmax(f1_score_)],
                            max_depth=save_m[np.argmax(f1_score_)], reg_alpha=save_L1[np.argmax(f1_score_)],
                            objective='cross_entropy',
                            random_state=119)
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

# Test Acc
y_pre_test = best_model.predict(X.iloc[valid_idx, :])
cm_test = confusion_matrix(Y.iloc[valid_idx], y_pre_test)
print("Test Confusion Matrix")
print(cm_test)
print("TesT Acc : {}".format((cm_test[0, 0] + cm_test[1, 1]) / cm_test.sum()))
print("Test F1-Score : {}".format(f1_score(Y.iloc[valid_idx], y_pre_test)))