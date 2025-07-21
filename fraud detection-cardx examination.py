# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.utils import resample
# import matplotlib.pyplot as plt

# # 1. Veriyi yükle
# df = pd.read_csv(
#     "C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv"
# )

# # 2. Kullanılacak sütunlar
# card_columns = ['card1', 'card2', 'card3', 'card5', 'card6']
# df_subset = df[card_columns + ['isFraud']].copy()

# # 3. Eksik verileri doldur
# df_subset['card6'] = df_subset['card6'].fillna('unknown')  # kategorik olan
# for col in ['card1', 'card2', 'card3', 'card5']:
#     df_subset[col] = df_subset[col].fillna(df_subset[col].mean())  # sayısal olanlar

# # 4. Kategorik veriyi sayısala çevir
# le = LabelEncoder()
# df_subset['card6'] = le.fit_transform(df_subset['card6'])

# # 5. Özellik ve hedef değişkenleri ayır
# X = df_subset.drop('isFraud', axis=1)
# y = df_subset['isFraud']

# # 6. Veriyi eğitim ve test olarak ayır (stratify kullanımı önemlidir)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # 7. Dengesiz veriyi dengele (downsampling)
# train_data = pd.concat([X_train, y_train], axis=1)
# fraud = train_data[train_data['isFraud'] == 1]
# non_fraud = train_data[train_data['isFraud'] == 0]

# non_fraud_downsampled = resample(
#     non_fraud,
#     replace=False,
#     n_samples=len(fraud),
#     random_state=42
# )

# balanced = pd.concat([fraud, non_fraud_downsampled])

# X_train_bal = balanced.drop('isFraud', axis=1)
# y_train_bal = balanced['isFraud']

# # 8. Modeli eğit (Decision Tree)
# model = DecisionTreeClassifier(max_depth=4, random_state=42)
# model.fit(X_train_bal, y_train_bal)

# # 9. Tahmin yap
# y_pred = model.predict(X_test)

# # 10. Değerlendirme
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # 11. Feature Importance
# plt.figure(figsize=(8,4))
# importances = pd.Series(model.feature_importances_, index=X.columns)
# importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importances')
# plt.tight_layout()
# plt.show()

# # 12. Mutual Information
# plt.figure(figsize=(8,4))
# mi = mutual_info_classif(X, y, discrete_features=[False, False, False, False, True])
# mi_series = pd.Series(mi, index=X.columns) 
# mi_series.sort_values(ascending=False).plot(kind='bar', title='Mutual Information Scores')
# plt.tight_layout()
# plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Veri yükleme
df = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv")

# Kart sütunları ve hedef
card_columns = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
df_subset = df[card_columns + ['isFraud']].copy()

# Eksik doldurma
df_subset['card4'].fillna('unknown', inplace=True)
df_subset['card6'].fillna('unknown', inplace=True)
for col in ['card1', 'card2', 'card3', 'card5']:
    df_subset[col].fillna(df_subset[col].mean(), inplace=True)

# Label Encoding for card4 and card6 (categorical)
le_card4 = LabelEncoder()
df_subset['card4'] = le_card4.fit_transform(df_subset['card4'])

le_card6 = LabelEncoder()
df_subset['card6'] = le_card6.fit_transform(df_subset['card6'])

# KDE plot for each card column
# for col in card_columns:
#     plt.figure(figsize=(10,4))
#     sns.kdeplot(df_subset.loc[df_subset['isFraud']==0, col], label='Non-Fraud', shade=True)
#     sns.kdeplot(df_subset.loc[df_subset['isFraud']==1, col], label='Fraud', shade=True)
#     plt.title(f'Distribution of {col} by Fraud Status')
#     plt.legend()
#     plt.show()
#RESULTS----------RESULTS-----------RESULTS--------------RESULTS-------RESULTS-------RESULTS-------RESULTS-------RESULTS-------RESULTS-------RESULTS-------RESULTS---

#card1 and card2 will be eliminated and other elements (card3,card4,card5,card6) will be kept for training the model.