import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

workspace_dir = os.path.dirname(os.path.abspath(__file__))
folder_names = {
    "ventas": "fact_ventas",
    "evaluacion": "fact_evaluacion_proveedores",
    "abastecimiento": "fact_abastecimiento_logistica",
    "inventario": "fact_inventario",
    "competencia": "fact_competencia"
}

# Pre-load models metadata and transformers
models = {}
for name, folder in folder_names.items():
    meta_path = os.path.join(workspace_dir, folder, "outputs", "inference_model.json")
    qt_path = os.path.join(workspace_dir, folder, "outputs", "quantile_transformer.joblib")
    
    if os.path.exists(meta_path) and os.path.exists(qt_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            qt = joblib.load(qt_path)
            models[name] = {
                "meta": meta,
                "transformer": qt
            }
            print(f"Loaded model {name} successfully.")
        except Exception as e:
            print(f"Error loading model {name}: {e}")
    else:
        print(f"Warning: Model paths not found for {name}. Meta: {os.path.exists(meta_path)}, QT: {os.path.exists(qt_path)}")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

@app.route("/")
def index():
    # Gather models list for UI sidebar
    available_models = []
    for k, v in models.items():
        available_models.append({
            "id": k,
            "name": k.replace("_", " ").title(),
            "target": v["meta"]["target_col"],
            "fields": v["meta"]["input_fields"]
        })
    return render_template("index.html", available_models=available_models)

@app.route("/plot/<model_name>/dispersion")
def get_dispersion_plot(model_name):
    if model_name not in folder_names:
        return "Model not found", 404
    folder = folder_names[model_name]
    img_path = os.path.join(workspace_dir, folder, "outputs", "tsne_dispersion.png")
    if os.path.exists(img_path):
        from flask import send_file
        return send_file(img_path, mimetype='image/png')
    else:
        return "Image not found", 404

@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if model_name not in models:
        return jsonify({"error": f"Model {model_name} not found or loaded."}), 404
        
    model_data = models[model_name]
    meta = model_data["meta"]
    qt = model_data["transformer"]
    
    try:
        input_data = request.json
        if not input_data:
            return jsonify({"error": "No input data provided."}), 400
            
        # Form input dataframe
        df_in = pd.DataFrame([input_data])
        
        # Fix for QuantileTransformer: it expects original feature shape, not just selected
        if "original_feature_names" in meta and "selected_feature_indices" in meta:
            feature_names = meta["original_feature_names"]
            X_encoded = pd.DataFrame(0.0, index=[0], columns=feature_names)
            
            for col in df_in.columns:
                val = df_in.loc[0, col]
                if col in meta["categorical_cols"]:
                    oh_col = f"{col}_{val}"
                    if oh_col in X_encoded.columns:
                        X_encoded.loc[0, oh_col] = 1.0
                else:
                    if col in X_encoded.columns:
                        try:
                            X_encoded.loc[0, col] = float(val)
                        except Exception as e:
                            pass
                            
            X_scaled_full = qt.transform(X_encoded.values.astype(float))
            X_scaled = X_scaled_full[:, meta["selected_feature_indices"]]
        else:
            # Fallback old behavior
            one_hot_cols = meta["one_hot_columns"]
            X_encoded = pd.DataFrame(0.0, index=[0], columns=one_hot_cols)
            
            for col in df_in.columns:
                val = df_in.loc[0, col]
                if col in meta["categorical_cols"]:
                    oh_col = f"{col}_{val}"
                    if oh_col in X_encoded.columns:
                        X_encoded.loc[0, oh_col] = 1.0
                else:
                    if col in X_encoded.columns:
                        try:
                            X_encoded.loc[0, col] = float(val)
                        except:
                            pass
                            
            X_scaled = qt.transform(X_encoded.values.astype(float))
        
        # Load weights
        W1 = np.array(meta["W1"])
        b1 = np.array(meta["b1"])
        W2 = np.array(meta["W2"])
        b2 = np.array(meta["b2"])
        W3 = np.array(meta["W3"])
        b3 = np.array(meta["b3"])
        
        # Feedforward
        Z1 = np.dot(X_scaled, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = softmax(Z3)
        
        # Extract probabilities and prediction
        probs = A3[0].tolist()
        pred_idx = int(np.argmax(probs))
        pred_class = meta["classes"][pred_idx]
        confidence = float(probs[pred_idx])
        
        classes_with_probs = [
            {"class": meta["classes"][i], "probability": float(probs[i])}
            for i in range(len(meta["classes"]))
        ]
        
        return jsonify({
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": classes_with_probs,
            "model_architecture": f"Capa 1: {meta['hidden1']} Neuronas, Capa 2: {meta['hidden2']} Neuronas"
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
