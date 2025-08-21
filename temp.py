import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
test = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\test_transaction.csv")
identity = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\test_identity.csv")
# v_cols = [f'V{i}' for i in range(1, 340)]
# m_cols = [f'M{i}' for i in range(1, 10)]
# d_cols = [f'D{i}' for i in range(1, 16)]
# c_cols = [f'C{i}' for i in range(1, 15)]
# columns_to_del = ["TransactionID", "P_emaildomain", "R_emaildomain", "dist1", "dist2"]
# test.drop(columns=v_cols + m_cols + d_cols + c_cols + columns_to_del, inplace=True)
# test = test[~test['card6'].isin(['charge card', 'debit or credit'])]

# #ENCODING addr2
# def encode_addr2(x):
#     if pd.isna(x):
#         return 1
#     elif x == 87.0:
#         return 0
#     else:
#         return 2

# test['addr2_encoded'] = test['addr2'].apply(encode_addr2)

# for col in ['card2', 'card3', 'card5']:
#     test[col] = test[col].fillna(test[col].median())

# for col in ['addr1']:
#     test[col] = test[col].fillna(test[col].mode()[0]).astype(str)

# # ProductCD and card6 One-Hot Encoding 
# test = pd.get_dummies(test, columns=['ProductCD', 'card6'], dummy_na=True)

# df_train = test[test['card4'].notnull()].copy()
# df_pred = test[test['card4'].isnull()].copy()

# # Label encode card4 
# le_card4 = LabelEncoder()
# df_train['card4_encoded'] = le_card4.fit_transform(df_train['card4'])

# one_hot_cols = [col for col in test.columns if col.startswith('ProductCD_') or col.startswith('card6_')]

# features = ['TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2_encoded'] + one_hot_cols

# # Train model
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(df_train[features], df_train['card4_encoded'])

# # Guess missing card4 values
# predicted_labels = le_card4.inverse_transform(knn.predict(df_pred[features]))
# test.loc[test['card4'].isnull(), 'card4'] = predicted_labels

# # After knn card4  One-Hot Encoding 
# test = pd.get_dummies(test, columns=['card4'], dummy_na=True)

# col = 'TransactionAmt'
# data = test[col]

# # 2. Log transformation
# log_data = np.log1p(data)  # log(TransactionAmt + 1)

# # 4. Z-score on log-transformed data
# log_zscores = zscore(log_data)
# threshold = 3
# outliers_z = log_data[np.abs(log_zscores) > threshold]

# # 7. Final: Add log and z-score columns to dataframe
# test[f'{col}_log'] = log_data
# test[f'{col}_log_zscore'] = log_zscores


# #TransationDT preprocessing:
# test['Hour'] = (test['TransactionDT'] % 86400 // 3600).astype(int)

# cards = test[['card1','card2','card3','card5']]

# drop= ["TransactionDT", "card1", "card2", "TransactionAmt", "TransactionAmt_log", "card4_nan", "addr2","card6_nan","ProductCD_nan"]
# test.drop(columns=drop , inplace=True)

# # Frekansı 1 olan addr1 değerlerini bul
# addr1_counts = test['addr1'].value_counts()
# rare_addr1 = addr1_counts[addr1_counts == 1].index

# # Bu adresleri içeren satırları veriden çıkar
# test = test[~test['addr1'].isin(rare_addr1)]

# scaler = StandardScaler()

# test['addr1_standardized'] = scaler.fit_transform(test[['addr1']])
# standard_scaler = StandardScaler()
# test[['card3_standardized', 'card5_standardized']] = standard_scaler.fit_transform(test[['card3', 'card5']])
# ctd = ["addr1", "card3", "card5"]
# test.drop(columns=ctd, inplace=True)
# print(test.tail(50))
# print(test.columns)
# cols = ["TransactionID","id-01", "id-03", "id-04", "id-05", "id-06", "id-07", "DeviceType","DeviceInfo"]
# id.drop(columns=cols,inplace=True)
# print(id.sample(50))
from sklearn.preprocessing import LabelEncoder


from sklearn.preprocessing import LabelEncoder

label_cols = ["id-35", "id-36", "id-37", "id-38"]
le = LabelEncoder()

for col in label_cols:
    if col in identity.columns:
        # Eksik değer varsa en sık görülenle dolduralım
        mode_val = identity[col].mode()[0]
        identity[col] = identity[col].fillna(mode_val)

        # Label encoding
        identity[col] = le.fit_transform(identity[col].astype(str))

        # Mapping çıktısını yazdıralım
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"{col} mapping:")
        for k, v in mapping.items():
            print(f"  {k} -> {v}")
        print()
