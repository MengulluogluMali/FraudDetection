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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score



# Load datasets
transactions = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv")
identity = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_identity.csv")
idbg = identity
def qo(df):
    print("Columns:\n", df.columns)
    print("\nShape:", df.shape)
    print("\nSample 50 rows:\n", df.sample(50))
    print("\nMissing values per column:\n", df.isnull().sum())

# df_trans = pd.read_csv("train_transaction.csv")
# df_id = pd.read_csv("train_identity.csv")

# # TransactionID sütunlarını al
# trans_ids = set(df_trans['TransactionID'])
# id_ids = set(df_id['TransactionID'])

# # Eşleşen ve eşleşmeyen TransactionID'leri bul
# common_ids = trans_ids.intersection(id_ids)
# only_in_trans = trans_ids.difference(id_ids)
# only_in_id = id_ids.difference(trans_ids)

# # Eşleşen kayıtlar
# df_common = df_trans[df_trans['TransactionID'].isin(common_ids)]
# # Eşleşmeyen (sadece transaction'da olanlar)
# df_only_trans = df_trans[df_trans['TransactionID'].isin(only_in_trans)]

# # Kaç tanesi fraud?
# common_fraud_count = df_common['isFraud'].sum()
# only_trans_fraud_count = df_only_trans['isFraud'].sum()

# # Sonuçları yazdır
# print("Toplam eşleşen TransactionID sayısı:", len(common_ids))
# print("Toplam eşleşmeyen TransactionID sayısı:", len(only_in_trans))
# print("Eşleşenlerden kaç tanesi fraud:", common_fraud_count)
# print("Eşleşmeyenlerden kaç tanesi fraud:", only_trans_fraud_count)

unique_device_info = identity['DeviceInfo'].dropna().unique()

# Kaç farklı değer olduğunu yazdır
print("Farklı DeviceInfo değeri sayısı:", len(unique_device_info))
print("DeviceInfo değerleri:")
print(unique_device_info)

l_starting_devices = identity['DeviceInfo'].dropna()
l_starting_matches = l_starting_devices[l_starting_devices.str.startswith(('L', 'l'))]
print("\nL ile başlayan DeviceInfo değerleri:")
print(l_starting_matches.unique())

identity['DeviceCategory'] = identity['DeviceInfo'].str.lower().fillna('').apply(
    lambda x: 'iOS Device' if 'iphone' in x or 'ipad' in x or 'ios' in x else
              'MacOS' if 'mac' in x or 'macos' in x else
              'Windows' if 'windows' in x else
              'Android'
)
print(identity['DeviceCategory'].value_counts())


cols_to_drop = ["id_33","DeviceType", "DeviceInfo","id_30","id_31"]
identity.drop(columns=cols_to_drop, inplace=True)





for i in range(2, 30):
    col = f"id_{i:02d}"
    if col in identity.columns:
        print(f"\n{col} - Unique Values ({identity[col].nunique()}):")
        print(identity[col].unique())
    else:
        print(f"\n{col} sütunu identity tablosunda yok.")


# Her sütundaki eksik değerlerin oranını hesapla
missing_ratios = identity.isnull().mean().sort_values(ascending=False)

# Yüzde cinsinden ve ilk 30 sütunu gösterelim
missing_ratios_percent = (missing_ratios * 100).round(2)



for col, perc in missing_ratios_percent.items():
    print(f"{col}: {perc}%")

# %40'tan fazla eksik değere sahip sütunları bul
cols_to_drop = missing_ratios[missing_ratios > 0.40].index

# Bu sütunları identity tablosundan kaldıralım
identity = identity.drop(columns=cols_to_drop)

print(f"Silinen sütun sayısı: {len(cols_to_drop)}")
print(f"Kalan sütun sayısı: {identity.shape[1]}")

# 1. NaN'leri mod değeriyle doldur
mode_val = identity['id_15'].mode()[0]
identity['id_15'] = identity['id_15'].fillna(mode_val)

# 2. One-hot encoding
id15_dummies = pd.get_dummies(identity['id_15'], prefix='id_15')

# 3. Yeni sütunları ekle
identity = pd.concat([identity, id15_dummies], axis=1)

# 4. Orijinal sütunu kaldır (istersen)
identity.drop('id_15', axis=1, inplace=True)


        
        
label_cols = ["id_12", "id_16", "id_28", "id_29"]
le = LabelEncoder()

for col in label_cols:
    if col in identity.columns:
        # En sık görülen (mod) değeri al
        mode_val = identity[col].mode()[0]

        # Güvenli atama: doğrudan sütunu güncelle
        identity[col] = identity[col].fillna(mode_val)

        # Label encoding
        identity[col] = le.fit_transform(identity[col].astype(str))

identity = identity.dropna(thresh=identity.shape[1] - 14)
print(f"Remaining rows: {identity.shape[0]}")
cols = ["id_12", "id_15", "id_16", "id_28", "id_29"]

for col in cols:
    if col in identity.columns:
        print(f"\n--- {col} ---")
        print(identity[col].value_counts(dropna=False))
    else:
        print(f"\n{col} sütunu tabloda yok.")
        
        
print(identity.head(50))