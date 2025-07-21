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
#ON THIS SECTION I AM TRYING TO FILL MISSING DATA IN A COMPLEX WAY. INSTEAD OF FILLING WITH MOD, I WILL TRY TO USE K-MEANS AND KNN TO FILL card4 COLUMN
for col in ['card2', 'card3', 'card5']:
    transactions[col].fillna(transactions[col].median(), inplace=True)
    
for col in ['addr1', 'addr2']:
    transactions[col].fillna(transactions[col].mode()[0], inplace=True)
    
transactions['card6'].fillna(transactions['card6'].mode()[0], inplace=True)

cat_cols = ['ProductCD', 'card6']
encoder = LabelEncoder()

for col in cat_cols:
    transactions[col] = encoder.fit_transform(transactions[col])
    
#print(transactions.isnull().sum())

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
    
#-------------------------------------------------------------------------------------------------------------------------------------    
#RESULTS ARE FOLLOWING
# predicted_series = pandas.Series(predicted_labels)
# print(predicted_series.value_counts())
#print(transactions.isnull().sum())
#1577 MISSING card4 VALUE WAS PREDICTED BY K-NEAREST NEIGHBOR ALGORITHM. 
# 1188 WAS GUESSED VISA, 
# 250 WAS GUESSED MASTERCARD,
# 118 WAS GUESSED AMERICAN EXPRESS,
#21 WAS GUESSED DISCOVER

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#Detecting and resolving outliers
print(transactions.head(50))
import matplotlib.pyplot as plt
import numpy as np

# İncelemek istediğin sütun
col = 'TransactionAmt'
data = transactions[col].dropna()
log_data = np.log1p(data)

# plt.figure(figsize=(10, 6))
# plt.hist(log_data, bins=100, color='orange', edgecolor='black')
# plt.title('Log-Transformed Transaction Amount Histogram')
# plt.xlabel('log(TransactionAmt + 1)')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# IQR yöntemi ile sınırlar
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Aykırı değerleri filtrele
outliers = data[(data < lower_bound) | (data > upper_bound)]
#ortalama
mean_val = data.mean()
# Boxplot + scatter (sadece aykırılar)
# plt.figure(figsize=(12, 6))
# plt.boxplot(data, vert=False)
# plt.scatter(outliers, np.ones_like(outliers), color='red', label='Outliers', zorder=3)

# plt.title(f'{col} Boxplot with Outliers Highlighted')
# plt.xlabel(col)
# plt.legend()
# plt.grid(True)
# plt.show()
print("Aykırı değer sayısı:", len(outliers))
print("Q1:", Q1)
print("Q3", Q3)
print("Mean:", mean_val)
negatives = transactions[transactions['TransactionAmt'] < 0]
print(negatives[['TransactionAmt']])
print(f"Negatif değer sayısı: {len(negatives)}")
print(transactions['TransactionAmt'].dtype)

col = 'TransactionAmt'
data = transactions[col].dropna()

# Z-score hesapla
z_scores = zscore(data)
threshold = 3  # Z-score eşiği

# Aykırı değerleri belirle
outliers = data[(np.abs(z_scores) > threshold)]
mean_val = data.mean()

# Görselleştir
# plt.figure(figsize=(12, 6))
# plt.boxplot(z_scores, vert=False)
# plt.scatter(z_scores[np.abs(z_scores) > threshold], 
#             np.ones_like(outliers), 
#             color='red', label='Z-score > 3', zorder=3)
# plt.title(f'Z-Score Normalized Boxplot for {col}')
# plt.xlabel('Z-score')
# plt.grid(True)
# plt.legend()
# plt.show()

# Sonuçları yazdır
print(f"Z-score ile aykırı değer sayısı: {len(outliers)}")
print(f"Ortalama (orijinal ölçekte): {mean_val:.2f}")
# # Histogram çizimi
# plt.figure(figsize=(10, 6))
# plt.hist(data, bins=100, color='skyblue', edgecolor='black')
# plt.title('Transaction Amount Histogram')
# plt.xlabel('TransactionAmt')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

log_data = np.log1p(data)
outliers = log_data[(log_data < lower_bound) | (log_data > upper_bound)]
# plt.figure(figsize=(10, 6))
# plt.hist(log_data, bins=100, color='orange', edgecolor='black')
# plt.title('Log-Transformed Transaction Amount Histogram')
# plt.xlabel('log(TransactionAmt + 1)')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()
# print(f"Log dönüşümü sonrası aykırı değer sayısı: {len(outliers)}")
# print("Bazı aykırı değerler örnekleri (log değerleri):")
# print(outliers.head(10))

#USING LOG TRANSFORMATION AND Z-SCORE TO HANDLE 66482 OUTLIERS
#RESULTS:
#IF ONLY Z-SCORE USED RESULTS ARE 10093 OUTLIERSLEFT
#LOG AND Z-SCORE GIVES US 0 OUTLIERS


print(transactions.isnull().sum())
card_columns_to_drop = ["card1", "card2"]
transactions.drop(columns=card_columns_to_drop, inplace=True)
print(transactions.head(20))