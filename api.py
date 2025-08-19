from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import pickle

app = Flask(__name__)

# Yükle
model = joblib.load("fraud_model.pkl")

scaler = joblib.load("scaler.pkl")

# 2️⃣ Modelin eğitimde kullandığı categorical sütun mapleri
productCD_map = ['C', 'H', 'R', 'S', 'W']
card4_map = ['american express', 'discover', 'mastercard', 'visa']
card6_map = ['credit', 'debit']

# 3️⃣ Numeric sütunlar (scaler uygulanacak)
numeric_cols = [
    'TransactionAmt_log_zscore', 'Hour', 'addr1_standardized', 'card3_standardized', 'card5_standardized',
    'id_12', 'id_16', 'id_28', 'id_29', 'id_35', 'id_36', 'id_37', 'id_38',
    'id_01_zscore', 'id_02_zscore', 'id_05_zscore', 'id_06_zscore',
    'id_11_zscore', 'id_13_zscore', 'id_17_zscore', 'id_19_zscore', 'id_20_zscore'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Numeric inputları al
            numeric_input = []
            for col in numeric_cols:
                val = float(request.form.get(col, 0))
                numeric_input.append(val)

            # Radio button inputları al ve one-hot encode
            productCD_input = request.form.get("ProductCD")
            productCD_encoded = [1 if productCD_input == val else 0 for val in productCD_map]

            card4_input = request.form.get("card4")
            card4_encoded = [1 if card4_input == val else 0 for val in card4_map]

            card6_input = request.form.get("card6")
            card6_encoded = [1 if card6_input == val else 0 for val in card6_map]

            # Tüm inputları birleştir
            X = numeric_input + productCD_encoded + card4_encoded + card6_encoded

            # Scaler uygula (numeric kısım)
            X_scaled_numeric = scaler.transform([X[:len(numeric_cols)]])
            X_final = np.concatenate([X_scaled_numeric[0], X[len(numeric_cols):]])

            # Tahmin
            pred = model.predict([X_final])[0]
            prediction = "FRAUD ✅" if pred == 1 else "SAFE ❌"

        except Exception as e:
            prediction = f"Hata: {str(e)}"

    return render_template("index.html", prediction=prediction, numeric_cols=numeric_cols, productCD_map=productCD_map,
                           card4_map=card4_map, card6_map=card6_map)

if __name__ == "__main__":
    app.run(debug=True)
