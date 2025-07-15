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
c_cols = [f'C{i}' for i in range(1,15)]
columns_to_del = ["TransactionID", "P_emaildomain", "R_emaildomain", "dist1", "dist2"]
transactions.drop(columns=v_cols, inplace=True)
transactions.drop(columns=m_cols, inplace=True)
transactions.drop(columns=d_cols, inplace=True)
transactions.drop(columns=c_cols, inplace=True)
transactions.drop(columns=columns_to_del, inplace=True)
print(transactions.head(50))
print(transactions.isnull().sum())


#Adress information is masked because it's a personal infor, instead of deleting this column, i will try to use clustering algorithms and try to mine some useful data out of it. (Creating my own map with multiple clusters)
#most effective column for fraud marked data should be found using decision tree algorithms.

#fraudulent = transactions[transactions['isFraud'] == 1]
#print(fraudulent)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ON THIS SECTION I AM TRYING TO FILL MISSING DATA IN A COMPLEX WAY. INSTEAD OF FILLING WITH MOD I WILL TRY TO USE K-MEANS AND KNN TO FILL card4 COLUMN
for col in ['card2', 'card3', 'card5']:
    transactions[col].fillna(transactions[col].median(), inplace=True)
    
for col in ['addr1', 'addr2']:
    transactions[col].fillna(transactions[col].mode()[0], inplace=True)
    
transactions['card6'].fillna(transactions['card6'].mode()[0], inplace=True)

cat_cols = ['ProductCD', 'card6']
encoder = LabelEncoder()

for col in cat_cols:
    transactions[col] = encoder.fit_transform(transactions[col])
    
print(transactions.isnull().sum())

# # Sadece eksik olmayan satırlarla çalışıyoruz
# df_tree = transactions[transactions['card4'].notnull()].copy()

# # Encode target
# le_card4 = LabelEncoder()
# df_tree['card4_encoded'] = le_card4.fit_transform(df_tree['card4'])

# # Özellik sütunları (eksik olmayanlar + sayısallaştırılmış olanlar)
# features = ['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card5', 'card6', 'addr1', 'addr2']
# X = df_tree[features]
# y = df_tree['card4_encoded']

# model = DecisionTreeClassifier(max_depth=5, random_state=42)
# model.fit(X, y)

# importances = model.feature_importances_
# feature_importance_df = pandas.DataFrame({
#     'Feature': features,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# print(feature_importance_df)

# plt.figure(figsize=(10, 6))
# sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
# plt.title("card4 Importance")
# plt.tight_layout()
# plt.show()
# plt.figure(figsize=(20, 10))  # Geniş bir figür boyutu önerilir
# plot_tree(
#     model,
#     feature_names=features,
#     class_names=le_card4.classes_,  # orijinal card4 değerleri
#     filled=True,
#     rounded=True,
#     fontsize=10
# )
# plt.title("Decision Tree - card4 ")
# plt.show()


# Eksik olmayanları al
df_train = transactions[transactions['card4'].notnull()]
df_pred = transactions[transactions['card4'].isnull()]

# Özellik olarak kullanacağımız sütunlar
features = ['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card5', 'card6', 'addr1', 'addr2']

# Label encoding (target)
le_card4 = LabelEncoder()
df_train['card4_encoded'] = le_card4.fit_transform(df_train['card4'])

# KNN modeli
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(df_train[features], df_train['card4_encoded'])

# Tahmin et
predictions = knn.predict(df_pred[features])

# Tahmin edilenleri orijinal kategoriye çevir
predicted_labels = le_card4.inverse_transform(predictions)

# Eksik değerleri yerine yaz
transactions.loc[transactions['card4'].isnull(), 'card4'] = predicted_labels
print("Eksik card4 sayısı:", transactions['card4'].isnull().sum())

count_predicted = Counter(predicted_labels)
print("Tahmin edilen card4 dağılımı:")
for card_type, count in count_predicted.items():
    print(f"{card_type}: {count}")
    
    
#
# predicted_series = pandas.Series(predicted_labels)
# print(predicted_series.value_counts())
# print(transactions.isnull().sum())
#1577 MISSING card4 VALUE WAS PREDICTED BY K-NEARIST NEIGHBOR ALGORITHM. 
# 1188 WAS GUESSED VISA, 
# 250 WAS GUESSED MASTERCARD,
# 118 WAS GUESSED AMERICAN EXPRESS,
#21 WAS GUESSED DISCOVER