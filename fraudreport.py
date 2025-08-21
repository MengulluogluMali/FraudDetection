import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from scipy.stats import zscore
import os
from sklearn.preprocessing import StandardScaler
identity = pandas.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_identity.csv")
transactions = pandas.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv")
v_cols = [f'V{i}' for i in range(1, 340)]  
m_cols = [f'M{i}' for i in range(1, 10)]  
d_cols = [f'D{i}' for i in range(1,16)]
c_cols = [f'C{i}' for i in range(1,15)]
columns_to_del = ["P_emaildomain", "R_emaildomain", "dist1", "dist2"]
transactions.drop(columns=v_cols, inplace=True)
transactions.drop(columns=m_cols, inplace=True)
transactions.drop(columns=d_cols, inplace=True)
transactions.drop(columns=c_cols, inplace=True)
transactions.drop(columns=columns_to_del, inplace=True)
print(transactions.head(50))
print(transactions.isnull().sum())
print(transactions.columns)

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Veri setini oku (merge edilmiş hali gerek yok, sadece transactions + identity)
transactions = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv")
identity = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_identity.csv")

# Merge
df = pd.merge(transactions, identity, on="TransactionID", how="left")

# HTML formda kullanacağımız numeric sütunlar
numeric_cols = [
    'TransactionAmt', 'TransactionDT', 'addr1', 'card3', 'card5',
    'id_01', 'id_02', 'id_05', 'id_06',
    'id_11', 'id_13', 'id_17', 'id_19', 'id_20',
    'addr2',
]


# Eksik değerleri doldur
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Scaler fit et
scaler = StandardScaler()
scaler.fit(df[numeric_cols])

# Kaydet
joblib.dump(scaler, "scaler_2.pkl")
print("✅ 22 sütun için scaler kaydedildi: scaler_22.pkl")

