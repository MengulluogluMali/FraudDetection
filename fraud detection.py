import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as matplot
import seaborn as seabrn
import os

transactions = pandas.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv")
identity = pandas.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_identity.csv")
#print(transactions.shape)
#print(identity.shape)
#print("\nTransaction columns:\n", transactions.columns.tolist()[:394])  
#print("\nIdentity columns:\n", identity.columns.tolist()[:41])
#print(transactions.head(12))
#print(identity.head(12))
#unique_products = transactions['ProductCD'].unique()
#print("ProductCD sütunundaki benzersiz değerler:")
#print(unique_products)
v_cols = [f'V{i}' for i in range(1, 340)]  
m_cols = [f'M{i}' for i in range(1, 10)]  
d_cols = [f'D{i}' for i in range(1,16)]

transactions.drop(columns=v_cols, inplace=True)
transactions.drop(columns=m_cols, inplace=True)
transactions.drop(columns=d_cols, inplace=True)
print(transactions.head(50))



#Adress information is masked because it's a personal infor, instead of deleting this column, i will try to use clustering algorithms and try to mine some useful data out of it. (Creating my own map with multiple clusters)
#most effective column for fraud marked data should be found using decision tree algorithms.

#fraudulent = transactions[transactions['isFraud'] == 1]
#print(fraudulent)
