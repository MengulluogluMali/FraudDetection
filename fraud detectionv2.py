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
# Load datasets
transactions = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv")
identity = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_identity.csv")

# Drop high-dimensional or less useful columns
v_cols = [f'V{i}' for i in range(1, 340)]
m_cols = [f'M{i}' for i in range(1, 10)]
d_cols = [f'D{i}' for i in range(1, 16)]
c_cols = [f'C{i}' for i in range(1, 15)]
columns_to_del = ["TransactionID", "P_emaildomain", "R_emaildomain", "dist1", "dist2"]
#  
transactions.drop(columns=v_cols + m_cols + d_cols + c_cols + columns_to_del, inplace=True)
transactions = transactions[~transactions['card6'].isin(['charge card', 'debit or credit'])]
#print(transactions.head(20))
# missing_addr1_fraud_count = transactions[(transactions['addr1'].isnull()) & (transactions['addr2'].isnull()) & (transactions['isFraud'] == 1)].shape[0]



# print(f"addr1 ve addr2 sütununda eksik ve isFraud=1 olan satır sayısı: {missing_addr1_fraud_count}")
def encode_addr2(x):
    if pd.isna(x):
        return 1
    elif x == 87.0:
        return 0
    else:
        return 2

transactions['addr2_encoded'] = transactions['addr2'].apply(encode_addr2)

# Kontrol için dağılımı yazdıralım
#print(transactions['addr2_encoded'].value_counts())

for col in ['card2', 'card3', 'card5']:
    transactions[col] = transactions[col].fillna(transactions[col].median())
#print(transactions['addr2_encoded'].isnull().sum())
for col in ['addr1']:
    transactions[col] = transactions[col].fillna(transactions[col].mode()[0]).astype(str)

# ProductCD ve card6 için One-Hot Encoding (card4 hariç, çünkü onu tahmin edeceğiz)
transactions = pd.get_dummies(transactions, columns=['ProductCD', 'card6'], dummy_na=True)

# card4 eksiklerini tahmin için hazırlık
df_train = transactions[transactions['card4'].notnull()].copy()
df_pred = transactions[transactions['card4'].isnull()].copy()

# Label encode card4 hedef değişkenini
le_card4 = LabelEncoder()
df_train['card4_encoded'] = le_card4.fit_transform(df_train['card4'])

# Modelde kullanacağımız sütunlar - artık ProductCD ve card6 sütunları one-hot sütunlarına dönüşmüş olacak,
# onları features listesine manuel eklememiz gerekiyor:
one_hot_cols = [col for col in transactions.columns if col.startswith('ProductCD_') or col.startswith('card6_')]

features = ['TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2_encoded'] + one_hot_cols

# Modeli eğit
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(df_train[features], df_train['card4_encoded'])

# Eksik card4'leri tahmin et
predicted_labels = le_card4.inverse_transform(knn.predict(df_pred[features]))
transactions.loc[transactions['card4'].isnull(), 'card4'] = predicted_labels

#print("Remaining missing card4 values:", transactions['card4'].isnull().sum())
#print("Predicted card4 value distribution:")
for k, v in Counter(predicted_labels).items():
    print(f"{k}: {v}")

# Tahmin sonrası card4 için One-Hot Encoding uygula
transactions = pd.get_dummies(transactions, columns=['card4'], dummy_na=True)

# Outlier detection and handling
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

# IQR method on original data
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]

# plt.figure(figsize=(12, 6))
# plt.boxplot(data, vert=False)
# plt.scatter(outliers_iqr, np.ones_like(outliers_iqr), color='red', label='Outliers', zorder=3)
# plt.title(f'{col} Boxplot with Outliers Highlighted')
# plt.xlabel(col)
# plt.legend()
# plt.grid(True)
# plt.show()

#print("Number of IQR-based outliers:", len(outliers_iqr))

# Check for negative transaction amounts
negatives = transactions[transactions[col] < 0]
#print(f"Number of negative values: {len(negatives)}")

col = 'TransactionAmt'
data = transactions[col]

# # 1. Histogram of raw data
# plt.figure(figsize=(10, 6))
# plt.hist(data, bins=100, color='skyblue', edgecolor='black')
# plt.title('Raw Transaction Amount Histogram')
# plt.xlabel('TransactionAmt')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# 2. Log transformation
log_data = np.log1p(data)  # log(TransactionAmt + 1)

# # 3. Histogram after log transformation
# plt.figure(figsize=(10, 6))
# plt.hist(log_data, bins=100, color='orange', edgecolor='black')
# plt.title('Log-Transformed Transaction Amount Histogram')
# plt.xlabel('log(TransactionAmt + 1)')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# 4. Z-score on log-transformed data
log_zscores = zscore(log_data)
threshold = 3
outliers_z = log_data[np.abs(log_zscores) > threshold]

# # 5. Boxplot and outlier highlight
# plt.figure(figsize=(12, 6))
# plt.boxplot(log_zscores, vert=False)
# plt.scatter(
#     log_zscores[np.abs(log_zscores) > threshold],
#     np.ones_like(outliers_z),  # tek satırda göstermek için
#     color='red', label='|Z-score| > 3', zorder=3
# )
# plt.title(f'Z-Score Boxplot (Log-Transformed {col})')
# plt.xlabel('Z-score')
# plt.grid(True)
# plt.legend()
# plt.show()

# #print(f"Number of Z-score outliers (after log): {len(outliers_z)}")

# 6. IQR-based outlier detection on log-transformed data
Q1_log = log_data.quantile(0.25)
Q3_log = log_data.quantile(0.75)
IQR_log = Q3_log - Q1_log
lower_bound_log = Q1_log - 1.5 * IQR_log
upper_bound_log = Q3_log + 1.5 * IQR_log
log_outliers_iqr = log_data[(log_data < lower_bound_log) | (log_data > upper_bound_log)]

# #print(f"Number of IQR outliers after log transformation: {len(log_outliers_iqr)}")
# #print("Sample outliers (log-transformed, IQR method):")
# #print(log_outliers_iqr.head(10))

# 7. Final: Add log and z-score columns to dataframe
transactions[f'{col}_log'] = log_data
transactions[f'{col}_log_zscore'] = log_zscores

# # 8. Sample output
# #print(transactions[[col, f'{col}_log', f'{col}_log_zscore']].head(20))


#------------------------------------------------------------------------------------------------------------------------------------------------------------
#TransationDT preprocessing:
transactions['Hour'] = (transactions['TransactionDT'] % 86400 // 3600).astype(int)



# Gerekli sütunları seç
cards = transactions[['card1','card2','card3','card5']]

# Eksik veri kontrolü
#print(cards.isnull().sum())

# Basit istatistikler
#print(cards.describe())




# Kart markasına göre sayısal sütunların ortalaması
# #print(cards.groupby('card4')[['card1','card2','card3','card5','card6']].mean())

# Korelasyon matrisi
# corr = cards[['card1','card2','card3','card5','card6']].corr()
# #print(corr)

# Korelasyon matrisi görselleştirme
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title("Correlation Matrix of Card Features")
# plt.show()

# # Kart markasına göre card2 dağılımı
# sns.boxplot(x='card4', y='card2', data=cards)
# plt.title("card2 Distribution by Card Brand")
# plt.show()



# isFraud oranına göre gruplama (örneğin card5)
# transactions.groupby('card5')['isFraud'].mean().sort_values(ascending=False).head(10)
# transactions.groupby('card2')['isFraud'].mean()
# transactions.groupby('card3')['isFraud'].mean()
# transactions.groupby('card6')['isFraud'].mean()



#print(transactions.head(20))
drop= ["TransactionDT", "card1", "card2", "TransactionAmt", "TransactionAmt_log", "card4_nan", "addr2","card6_nan","ProductCD_nan"]
transactions.drop(columns=drop , inplace=True)
#print(transactions.head(50))
#unique_values = transactions['addr2_encoded'].unique()
#print("addr1 sütunundaki benzersiz değer sayısı:", len(unique_values))
#print("Benzersiz değerler:", unique_values)
#print(transactions['addr2_encoded'].value_counts())




# X = transactions[['addr1', 'addr2']].values

# kmeans = KMeans(n_clusters=5, random_state=42)
# clusters = kmeans.fit_predict(X)

# plt.figure(figsize=(12,8))
# scatter = plt.scatter(X[:,0], X[:,1], c=clust ers, cmap='tab10', alpha=0.6)
# plt.xlabel('addr1')
# plt.ylabel('addr2')
# plt.title('addr1 ve addr2 ile KMeans Kümeleme')
# plt.colorbar(scatter, ticks=range(10), label='Cluster')
# plt.show()



# addr1_counts = transactions['addr1'].value_counts()
# rare_addr1 = addr1_counts[addr1_counts == 1].index

# # Adım 2: Bu nadir değerleri içeren satırları filtrele
# rare_fraud = transactions[(transactions['addr1'].isin(rare_addr1)) & (transactions['isFraud'] == 1)]

# # Sonuçları göster
# print(rare_fraud[['addr1', 'isFraud']])
# print("1 defa geçen adres sayısı",rare_addr1)
# print(f"\nToplam: {len(rare_fraud)} satır bulundu.")
# Frekansı 1 olan addr1 değerlerini bul
addr1_counts = transactions['addr1'].value_counts()
rare_addr1 = addr1_counts[addr1_counts == 1].index

# Bu adresleri içeren satırları veriden çıkar
transactions = transactions[~transactions['addr1'].isin(rare_addr1)]

print(f"Kalan satır sayısı: {len(transactions)}")

print(transactions.tail(50))
print(transactions.columns)
print(transactions['addr1'].value_counts())

print(transactions['addr1'].value_counts().tail(100))
unique_count = transactions['addr1'].nunique()
print(f"'addr1' sütununda {unique_count} farklı değer var.")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
#adres 1 sütununu düzenlemeye devam et IQR mı artık bilmiyorum şu sütunu da düzenle de gidek