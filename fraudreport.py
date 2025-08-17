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