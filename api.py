from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Model ve scaler yükle
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler_2.pkl")

# Categorical sütun mapleri
productCD_map = ['C', 'H', 'R', 'S', 'W']
card4_map = ['american express', 'discover', 'mastercard', 'visa']
card6_map = ['credit', 'debit']

# Label encoded sütun mapping
label_mappings = {
    "id_12": {"Found": 0, "NotFound": 1},
    "id_16": {"Found": 0, "NotFound": 1},
    "id_28": {"Found": 0, "New": 1},
    "id_29": {"Found": 0, "NotFound": 1},
    "id_35": {"F": 0, "T": 1},
    "id_36": {"F": 0, "T": 1},
    "id_37": {"F": 0, "T": 1},
    "id_38": {"F": 0, "T": 1},
}

# Numeric sütunlar (scaler uygulanacak)
numeric_cols = [
    'TransactionAmt_log_zscore', 'Hour', 'addr1_standardized',
    'card3_standardized', 'card5_standardized',
    'id_01_zscore', 'id_02_zscore', 'id_05_zscore', 'id_06_zscore',
    'id_11_zscore', 'id_13_zscore', 'id_17_zscore',
    'id_19_zscore', 'id_20_zscore', 'addr2_encoded'
]

# ID_15 one-hot mapping
id15_map = ['Found', 'New', 'Unknown']

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

            # Label encoded sütunları al
            label_input = []
            for col, mapping in label_mappings.items():
                val = request.form.get(col)
                if val not in mapping:
                    raise ValueError(f"Geçersiz değer: {col} = {val}")
                label_input.append(mapping[val])

            # One-hot encode edilen sütunlar
            productCD_input = request.form.get("ProductCD")
            productCD_encoded = [1 if productCD_input == val else 0 for val in productCD_map]

            card4_input = request.form.get("card4")
            card4_encoded = [1 if card4_input == val else 0 for val in card4_map]

            card6_input = request.form.get("card6")
            card6_encoded = [1 if card6_input == val else 0 for val in card6_map]

            # ID_15 one-hot
            id15_input = request.form.get("id_15")
            id15_encoded = [1 if id15_input == val else 0 for val in id15_map]

            # Tüm inputları birleştir
            X = numeric_input + label_input + productCD_encoded + card4_encoded + card6_encoded + id15_encoded

            # Numeric kısma scaler uygula
            X_scaled_numeric = scaler.transform([numeric_input])
            X_final = np.concatenate([X_scaled_numeric[0], label_input, productCD_encoded, card4_encoded, card6_encoded, id15_encoded])

            # Tahmin
            pred = model.predict([X_final])[0]
            prediction = "FRAUD ❌" if pred == 1 else "SAFE ✅"

        except Exception as e:
            prediction = f"Hata: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        numeric_cols=numeric_cols,
        productCD_map=productCD_map,
        card4_map=card4_map,
        card6_map=card6_map,
        label_mappings=label_mappings,
        id15_map=id15_map
    )

if __name__ == "__main__":
    app.run(debug=True)
