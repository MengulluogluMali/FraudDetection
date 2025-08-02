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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score



# Load datasets
transactions = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\train_transaction.csv")
test = pd.read_csv("C:\\Users\\Monster\\Desktop\\Erasmus 2025\\Internship WSTI\\IEEE\\ieee-fraud-detection\\test_transaction.csv")

# Drop high-dimensional or less useful columns
v_cols = [f'V{i}' for i in range(1, 340)]
m_cols = [f'M{i}' for i in range(1, 10)]
d_cols = [f'D{i}' for i in range(1, 16)]
c_cols = [f'C{i}' for i in range(1, 15)]
columns_to_del = ["TransactionID", "P_emaildomain", "R_emaildomain", "dist1", "dist2"]
transactions.drop(columns=v_cols + m_cols + d_cols + c_cols + columns_to_del, inplace=True)
transactions = transactions[~transactions['card6'].isin(['charge card', 'debit or credit'])]

#ENCODING addr2
def encode_addr2(x):
    if pd.isna(x):
        return 1
    elif x == 87.0:
        return 0
    else:
        return 2

transactions['addr2_encoded'] = transactions['addr2'].apply(encode_addr2)

for col in ['card2', 'card3', 'card5']:
    transactions[col] = transactions[col].fillna(transactions[col].median())

for col in ['addr1']:
    transactions[col] = transactions[col].fillna(transactions[col].mode()[0]).astype(str)

# ProductCD and card6 One-Hot Encoding 
transactions = pd.get_dummies(transactions, columns=['ProductCD', 'card6'], dummy_na=True)

df_train = transactions[transactions['card4'].notnull()].copy()
df_pred = transactions[transactions['card4'].isnull()].copy()

# Label encode card4 
le_card4 = LabelEncoder()
df_train['card4_encoded'] = le_card4.fit_transform(df_train['card4'])

one_hot_cols = [col for col in transactions.columns if col.startswith('ProductCD_') or col.startswith('card6_')]

features = ['TransactionDT', 'TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2_encoded'] + one_hot_cols

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(df_train[features], df_train['card4_encoded'])

# Guess missing card4 values
predicted_labels = le_card4.inverse_transform(knn.predict(df_pred[features]))
transactions.loc[transactions['card4'].isnull(), 'card4'] = predicted_labels

# After knn card4  One-Hot Encoding 
transactions = pd.get_dummies(transactions, columns=['card4'], dummy_na=True)

col = 'TransactionAmt'
data = transactions[col]

# 2. Log transformation
log_data = np.log1p(data)  # log(TransactionAmt + 1)

# 4. Z-score on log-transformed data
log_zscores = zscore(log_data)
threshold = 3
outliers_z = log_data[np.abs(log_zscores) > threshold]

# 7. Final: Add log and z-score columns to dataframe
transactions[f'{col}_log'] = log_data
transactions[f'{col}_log_zscore'] = log_zscores


#TransationDT preprocessing:
transactions['Hour'] = (transactions['TransactionDT'] % 86400 // 3600).astype(int)

cards = transactions[['card1','card2','card3','card5']]

drop= ["TransactionDT", "card1", "card2", "TransactionAmt", "TransactionAmt_log", "card4_nan", "addr2","card6_nan","ProductCD_nan"]
transactions.drop(columns=drop , inplace=True)

# Frekansı 1 olan addr1 değerlerini bul
addr1_counts = transactions['addr1'].value_counts()
rare_addr1 = addr1_counts[addr1_counts == 1].index

# Bu adresleri içeren satırları veriden çıkar
transactions = transactions[~transactions['addr1'].isin(rare_addr1)]

scaler = StandardScaler()

transactions['addr1_standardized'] = scaler.fit_transform(transactions[['addr1']])
standard_scaler = StandardScaler()
transactions[['card3_standardized', 'card5_standardized']] = standard_scaler.fit_transform(transactions[['card3', 'card5']])
ctd = ["addr1", "card3", "card5"]
transactions.drop(columns=ctd, inplace=True)
print(transactions.tail(50))
print(transactions.columns)
X = transactions.drop(columns=['isFraud'])
y = transactions['isFraud']

# Stratify, fraud dağılımını korur
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modeli tanımla (class_weight='balanced' ile fraud dengesizliği dikkate alınır)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Eğitimi başlat
rf_model.fit(X_train, y_train)

# Tahmin
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Skorlar
print("📊 Classification Report of Random Forest Classifier:\n")
print(classification_report(y_test, y_pred))

print("🎯 ROC-AUC Skoru:", roc_auc_score(y_test, y_proba))

# Confusion matrix
print("🧩 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# plt.figure(figsize=(8, 5))
# plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})", color="darkorange")
# plt.plot([0, 1], [0, 1], 'k--')
# plt.title('ROC Curve – Random Forest')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

importances = rf_model.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# print("\n📌 En önemli 10 feature:\n")
# print(feature_importance_df.head(10))

# Modeli oluştur
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# Eğitimi yap
xgb_clf.fit(X_train, y_train)

# Tahmin yap
y_pred = xgb_clf.predict(X_test)
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]

# Sonuçları yazdır
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("🧩 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n🎯 ROC-AUC Skoru: {roc_auc}")

# Orijinal sınıf dağılımını göster
print("🔍 Orijinal sınıf dağılımı:", Counter(y_train))

# Dengeleme pipeline'ı: önce oversample sonra undersample
over = SMOTE(sampling_strategy=0.2, random_state=42)  # Azınlık sınıfını %10 olacak şekilde arttır
under = RandomUnderSampler(sampling_strategy=0.4, random_state=42)  # Azınlık:Çoğunluk = 1:2 olacak şekilde azalt

pipeline = Pipeline(steps=[('o', over), ('u', under)])

# Yeni dengelenmiş veriyi oluştur
X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

# Yeni sınıf dağılımını yazdır
print("🎯 Dengeleme sonrası sınıf dağılımı:", Counter(y_resampled))

# Dengeleme sonrası örnek sayıları
print(f"\n📈 Eğitim verisi boyutu önce: {X_train.shape[0]} → sonra: {X_resampled.shape[0]}")

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_resampled, y_resampled)
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred_rf))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_rf))

# 3. XGBoost Modeli Eğit ve Değerlendir
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_resampled, y_resampled)
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- XGBoost Classification Report ---")
print(classification_report(y_test, y_pred_xgb))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_xgb))