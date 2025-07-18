import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from scipy.stats import zscore

# Load datasets
transactions = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv")
identity = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_identity.csv")

# Drop high-dimensional or less useful columns
v_cols = [f'V{i}' for i in range(1, 340)]
m_cols = [f'M{i}' for i in range(1, 10)]
d_cols = [f'D{i}' for i in range(1, 16)]
c_cols = [f'C{i}' for i in range(1, 15)]
columns_to_del = ["TransactionID", "P_emaildomain", "R_emaildomain", "dist1", "dist2"]

transactions.drop(columns=v_cols + m_cols + d_cols + c_cols + columns_to_del, inplace=True)
print(transactions.head(20))

# Fill missing values for numerical features
for col in ['card2', 'card3', 'card5']:
    transactions[col] = transactions[col].fillna(transactions[col].median())

# Fill missing values for categorical features
for col in ['addr1', 'addr2', 'card6']:
    transactions[col] = transactions[col].fillna(transactions[col].mode()[0])

# Encode categorical columns with separate LabelEncoders
cat_cols = ['ProductCD', 'card6']
for col in cat_cols:
    le = LabelEncoder()
    transactions[col] = le.fit_transform(transactions[col])

# Predict missing values in 'card4' using KNN
df_train = transactions[transactions['card4'].notnull()].copy()
df_pred = transactions[transactions['card4'].isnull()].copy()
features = ['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card5', 'card6', 'addr1', 'addr2']

le_card4 = LabelEncoder()
df_train['card4_encoded'] = le_card4.fit_transform(df_train['card4'])

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(df_train[features], df_train['card4_encoded'])

predicted_labels = le_card4.inverse_transform(knn.predict(df_pred[features]))
transactions.loc[transactions['card4'].isnull(), 'card4'] = predicted_labels

print("Remaining missing card4 values:", transactions['card4'].isnull().sum())
print("Predicted card4 value distribution:")
for k, v in Counter(predicted_labels).items():
    print(f"{k}: {v}")

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

print("Number of IQR-based outliers:", len(outliers_iqr))

# Check for negative transaction amounts
negatives = transactions[transactions[col] < 0]
print(f"Number of negative values: {len(negatives)}")

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

# print(f"Number of Z-score outliers (after log): {len(outliers_z)}")

# 6. IQR-based outlier detection on log-transformed data
Q1_log = log_data.quantile(0.25)
Q3_log = log_data.quantile(0.75)
IQR_log = Q3_log - Q1_log
lower_bound_log = Q1_log - 1.5 * IQR_log
upper_bound_log = Q3_log + 1.5 * IQR_log
log_outliers_iqr = log_data[(log_data < lower_bound_log) | (log_data > upper_bound_log)]

# print(f"Number of IQR outliers after log transformation: {len(log_outliers_iqr)}")
# print("Sample outliers (log-transformed, IQR method):")
# print(log_outliers_iqr.head(10))

# 7. Final: Add log and z-score columns to dataframe
transactions[f'{col}_log'] = log_data
transactions[f'{col}_log_zscore'] = log_zscores

# # 8. Sample output
# print(transactions[[col, f'{col}_log', f'{col}_log_zscore']].head(20))


#------------------------------------------------------------------------------------------------------------------------------------------------------------
#TransationDT preprocessing:
transactions_hour_only = pd.DataFrame()
transactions_hour_only['Hour'] = (transactions['TransactionDT'] % 86400 // 3600).astype(int)

print(transactions_hour_only.head(20))
print(transactions.head(20))

# Gerekli sütunları seç
cards = transactions[['card1','card2','card3','card4','card5','card6']]

# Eksik veri kontrolü
print(cards.isnull().sum())

# Basit istatistikler
print(cards.describe())

# card4 (kart markası) kategorik olduğundan, eksik varsa doldurabilir veya çıkarabilirsin:
cards = cards.dropna(subset=['card4'])  # kart markası boş olanları çıkar

# Kart markasına göre sayısal sütunların ortalaması
print(cards.groupby('card4')[['card1','card2','card3','card5','card6']].mean())

# Korelasyon matrisi
corr = cards[['card1','card2','card3','card5','card6']].corr()
print(corr)

# Korelasyon matrisi görselleştirme
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Card Features")
plt.show()

# Kart markasına göre card2 dağılımı
sns.boxplot(x='card4', y='card2', data=cards)
plt.title("card2 Distribution by Card Brand")
plt.show()



# isFraud oranına göre gruplama (örneğin card5)
transactions.groupby('card5')['isFraud'].mean().sort_values(ascending=False).head(10)
transactions.groupby('card2')['isFraud'].mean()
transactions.groupby('card3')['isFraud'].mean()
transactions.groupby('card6')['isFraud'].mean()
