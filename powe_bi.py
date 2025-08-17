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

transactions = pandas.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv")
identity = pandas.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_identity.csv")

# TransactionDT'yi 0-23 saat aralığına çevirme
transactions['hour'] = (transactions['TransactionDT'] // 3600) % 24

# String formatına çevirme
transactions['hour'] = transactions['hour'].astype(str)



powerdf = pandas.DataFrame()
powerdf["Hour"] = transactions["hour"]

transaction_amt = transactions["TransactionAmt"]

# IQR hesapla
Q1 = transaction_amt.quantile(0.25)
Q2 = transaction_amt.quantile(0.50)  # Medyan
Q3 = transaction_amt.quantile(0.75)
IQR = Q3 - Q1

# Outlier sınırları
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Kategorize etme fonksiyonu
def categorize_amt(x):
    if x < lower_bound:
        return "Lower Outlier"
    elif lower_bound <= x < Q1:
        return "Below Q1"
    elif Q1 <= x < Q2:
        return "Q1-Q2"
    elif Q2 <= x < Q3:
        return "Q2-Q3"
    elif Q3 <= x <= upper_bound:
        return "Above Q3"
    else:
        return "Upper Outlier"

# Yeni sütun ekle
transactions["TransactionAmt_Category"] = transaction_amt.apply(categorize_amt)

# Kontrol
print(transactions[["TransactionAmt", "TransactionAmt_Category"]].sample(20))
powerdf["TransactionAmt"] = transactions["TransactionAmt_Category"]
print(powerdf.head(50))
drop = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD","card1","card2","card3","card4","card5"]
transactions.drop(columns=drop, inplace=True)
print(transactions.sample(50))