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

# Fill missing values for numerical and categorical features
for col in ['card2', 'card3', 'card5']:
    transactions[col] = transactions[col].fillna(transactions[col].median())

for col in ['addr1', 'addr2']:
    transactions[col] = transactions[col].fillna(transactions[col].mode()[0])

transactions['card6'] = transactions['card6'].fillna(transactions['card6'].mode()[0])

# Encode categorical columns
cat_cols = ['ProductCD', 'card6']
encoder = LabelEncoder()
for col in cat_cols:
    transactions[col] = encoder.fit_transform(transactions[col])

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

plt.figure(figsize=(10, 6))
plt.hist(log_data, bins=100, color='orange', edgecolor='black')
plt.title('Log-Transformed Transaction Amount Histogram')
plt.xlabel('log(TransactionAmt + 1)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]

plt.figure(figsize=(12, 6))
plt.boxplot(data, vert=False)
plt.scatter(outliers_iqr, np.ones_like(outliers_iqr), color='red', label='Outliers', zorder=3)
plt.title(f'{col} Boxplot with Outliers Highlighted')
plt.xlabel(col)
plt.legend()
plt.grid(True)
plt.show()

print("Number of IQR-based outliers:", len(outliers_iqr))

# Check for negative transaction amounts
negatives = transactions[transactions[col] < 0]
print(f"Number of negative values: {len(negatives)}")

# Z-score method
z_scores = zscore(data)
threshold = 3
outliers_z = data[np.abs(z_scores) > threshold]

plt.figure(figsize=(12, 6))
plt.boxplot(z_scores, vert=False)
plt.scatter(z_scores[np.abs(z_scores) > threshold],
            np.ones_like(outliers_z),
            color='red', label='Z-score > 3', zorder=3)
plt.title(f'Z-Score Normalized Boxplot for {col}')
plt.xlabel('Z-score')
plt.grid(True)
plt.legend()
plt.show()

print(f"Number of Z-score outliers: {len(outliers_z)}")

# Raw distribution histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=100, color='skyblue', edgecolor='black')
plt.title('Transaction Amount Histogram')
plt.xlabel('TransactionAmt')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Outlier detection on log-transformed data
log_outliers = log_data[(log_data < lower_bound) | (log_data > upper_bound)]

plt.figure(figsize=(10, 6))
plt.hist(log_data, bins=100, color='orange', edgecolor='black')
plt.title('Log-Transformed Transaction Amount Histogram')
plt.xlabel('log(TransactionAmt + 1)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print(f"Number of outliers after log transformation: {len(log_outliers)}")
print("Sample outliers (log-transformed):")
print(log_outliers.head(10))

# Final sample output
print(transactions.head(20))

#Transforming
# Örnek kolon adı
col = 'TransactionAmt'

# Log dönüşüm (önce +1 ekleyerek log alma)
transactions[f'{col}_log'] = np.log1p(transactions[col])

# Z-score standardizasyonu (log dönüşüm sonrası)
transactions[f'{col}_log_zscore'] = zscore(transactions[f'{col}_log'])

print(transactions.head(20))