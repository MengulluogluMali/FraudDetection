from identity_preprocessing import processed_identity as identity
from fraud_detectionv3 import transactions_processed as transactions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_score, recall_score, f1_score, roc_curve)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
# INNER JOIN
data = pd.merge(transactions, identity, on="TransactionID", how="inner")

# Eksik veri kontrolü
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

print("Eksik veri içeren sütunlar (ilk 20):\n")
print(missing_values.head(20))
print(f"\nToplam {missing_values.shape[0]} sütunda eksik değer var.")

X = data.drop(columns=['isFraud', 'TransactionID'])
y = data['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelleri tanımla
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

for name, model in models.items():
    print(f"\n{name} Model Eğitiliyor...")
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    # Metrikler
    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # print(f"{name} AUC: {auc:.4f}")
    # print(f"{name} Precision: {precision:.4f}")
    # print(f"{name} Recall: {recall:.4f}")
    # print(f"{name} F1-Score: {f1:.4f}")
    
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))
    
    # # Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(5,4))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.title(f"{name} Confusion Matrix")
    # plt.xlabel("Tahmin")
    # plt.ylabel("Gerçek")
    # plt.show()
    
    # # Feature Importance
    # if name == "XGBoost":
    #     importance = model.get_booster().get_score(importance_type='weight')
    #     importance_df = pd.DataFrame({
    #         'Feature': list(importance.keys()),
    #         'Importance': list(importance.values())
    #     }).sort_values(by='Importance', ascending=False).head(20)
    # else:
    #     importance_df = pd.DataFrame({
    #         'Feature': X.columns,
    #         'Importance': model.feature_importances_
    #     }).sort_values(by='Importance', ascending=False).head(20)
    
    # plt.figure(figsize=(10,6))
    # sns.barplot(x='Importance', y='Feature', data=importance_df)
    # plt.title(f"{name} - En Önemli 20 Özellik")
    # plt.tight_layout()
    # plt.show()
    
    y_probs = model.predict_proba(X_test)[:, 1]

# ROC eğrisi verilerini al
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# En iyi eşik: TPR yüksek, FPR düşük (sol üst nokta)
optimal_idx = (tpr - fpr).argmax()
optimal_threshold_roc = thresholds[optimal_idx]
print("ROC Optimal Threshold:", optimal_threshold_roc)

# ROC Optimal Threshold'a göre yeniden sınıflandır
y_pred_optimal = (y_probs >= optimal_threshold_roc).astype(int)

# Yeni metriklerle yeniden değerlendirme
# print("\n*** ROC Optimal Threshold Kullanılarak Güncellenmiş Metrikler ***")
# print(f"Threshold: {optimal_threshold_roc:.4f}")
# print(classification_report(y_test, y_pred_optimal, digits=4))

# # Confusion Matrix
# cm_opt = confusion_matrix(y_test, y_pred_optimal)
# plt.figure(figsize=(5,4))
# sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Oranges")
# plt.title("ROC Eşiği ile Confusion Matrix")
# plt.xlabel("Tahmin")
# plt.ylabel("Gerçek")
# plt.show()


precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# F1 skoru ile en iyi eşik
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_thresh_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_thresh_idx]

print(f"PR Curve Optimal Threshold (F1 Max): {best_thresh:.4f}")

y_pred_pr = (y_probs >= 0.32).astype(int)




# print("PR Curve Optimal Threshold: 0.32\n")
# print(confusion_matrix(y_test, y_pred_pr))
# print("\n", classification_report(y_test, y_pred_pr, digits=4))

# cm_pr = confusion_matrix(y_test, y_pred_pr)

# Matrisin çıktısı
# print(cm_pr)
# optimal_threshold_pr = 0.32
# # Görselleştir
# plt.figure(figsize=(5,4))
# sns.heatmap(cm_pr, annot=True, fmt="d", cmap="Greens")
# plt.title(f"Confusion Matrix (PR Threshold = {optimal_threshold_pr})")
# plt.xlabel("Tahmin")
# plt.ylabel("Gerçek")
# plt.show()

X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Modeli tanımla
lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    objective='binary',
    class_weight='balanced',
    random_state=42
)

# Modeli eğit
lgbm_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc'
)

# Tahmin ve değerlendirme
y_pred = lgbm_model.predict(X_test)
y_proba = lgbm_model.predict_proba(X_test)[:, 1]

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin')
# plt.ylabel('Gerçek')
# plt.title('Confusion Matrix')
# plt.show()

bins = [0.0, 0.35, 0.70, 1.0]
labels = ['Low Risk (0-35%)', 'Medium Risk (35-70%)', 'High Risk (70-100%)']

risk_categories = pd.cut(y_probs, bins=bins, labels=labels, include_lowest=True)

# Sonuçları DataFrame'de topla
results_df = pd.DataFrame({
    'Probability': y_probs,
    'Risk Category': risk_categories,
    'True Label': y_test.reset_index(drop=True)
})

# Her risk kategorisi için istatistikler
summary = results_df.groupby('Risk Category')['True Label'].value_counts().unstack(fill_value=0)

# Ayrıca her kategoride toplam örnek sayısı
summary['Total'] = summary.sum(axis=1)

print(summary)

low_risk_max = 0.35
medium_risk_max = 0.70

# Öncelikle test setindeki indeksleri alalım
test_indices = y_test.index

# Tahmin edilen olasılıkları test seti indeksine göre DataFrame haline getir
prob_df = pd.DataFrame({
    'TransactionID': data.loc[test_indices, 'TransactionID'],
    'TrueLabel': y_test,
    'Fraud_Probability': y_probs
}, index=test_indices)

# Low Risk (0-35%) olanlar ve gerçek fraud olanlar
dfdf = prob_df[(prob_df['Fraud_Probability'] <= low_risk_max) & (prob_df['TrueLabel'] == 1)]

# Kaç satır olduğunu göster
print(f"Kaçırılan fraud sayısı (Low Risk içinde): {len(dfdf)}")

# İlk birkaç satırı göster
print(dfdf.head(50))

explainer = shap.TreeExplainer(lgbm_model)

# Test verisi üzerinde SHAP değerleri hesapla
shap_values = explainer.shap_values(X_test)

# Genel önemli özellikler
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Detaylı global özet grafik (renkli)
shap.summary_plot(shap_values, X_test)

# Belirli bir örnek için açıklama (mesela 5. satır)
idx = 5
shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx])