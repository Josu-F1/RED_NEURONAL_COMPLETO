import pandas as pd
import numpy as np
import pymongo
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import traceback

def run():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["ProyectoBueno"] 
    collection = db["fact_abastecimiento"]
    data = list(collection.find())
    
    if not data:
        print("No data found in fact_abastecimiento")
        return

    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    flat_data = [flatten_dict(record) for record in data]
    df = pd.DataFrame(flat_data)

    if 'cantidad_recibida' in df.columns and 'cantidad_solicitada' in df.columns:
        # Convert to numeric in case they are strings
        df['cantidad_recibida'] = pd.to_numeric(df['cantidad_recibida'], errors='coerce').fillna(0)
        df['cantidad_solicitada'] = pd.to_numeric(df['cantidad_solicitada'], errors='coerce').fillna(1)
        ratio = df['cantidad_recibida'] / df['cantidad_solicitada']
        df['y_target'] = pd.cut(ratio, bins=[-np.inf, 0.8, 1.0, np.inf], labels=[0, 1, 2]).astype(int)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df['y_target'] = (df[numeric_cols[0]] > df[numeric_cols[0]].median()).astype(int)
        else:
            df['y_target'] = np.random.randint(0, 2, len(df))

    cols_to_drop = []
    for col in df.columns:
        col_lower = col.lower()
        is_id = (col_lower == 'id' or col_lower == '_id' or col_lower == '_collection' or 
                 col_lower.startswith('id_') or col_lower.endswith('_id') or
                 '_id_' in col_lower or 'id_venta_dw' in col_lower)
        is_contact_or_date = ('telefono' in col_lower or 'correo' in col_lower or 'email' in col_lower or
                              'codigo_barras' in col_lower or 'codigo_producto' in col_lower or 
                              col_lower.endswith('_fecha') or col_lower.endswith('_hora') or 
                              col_lower == 'fecha' or col_lower == 'hora' or col_lower == 'tiempo_fecha')
        if (is_id or is_contact_or_date) and col != "y_target":
            cols_to_drop.append(col)

    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    
    y = df_clean['y_target']
    X_raw = df_clean.drop(columns=['y_target'])

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        X_raw[col].fillna(X_raw[col].median(), inplace=True)
    for col in categorical_cols:
        if not X_raw[col].mode().empty:
            X_raw[col].fillna(X_raw[col].mode().iloc[0], inplace=True)
        else:
            X_raw[col].fillna('UNKNOWN', inplace=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_raw[categorical_cols])
    feature_names = encoder.get_feature_names_out(categorical_cols)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=feature_names, index=X_raw.index)

    X_numeric = X_raw[numeric_cols]
    X_encoded = pd.concat([X_numeric, X_cat_df], axis=1)

    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_scaled_arr = scaler.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X_encoded.columns, index=X_encoded.index)

    corr_matrix = X_scaled.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.90)]
    X_uncorrelated = X_scaled.drop(columns=to_drop_corr)

    # Check if we have columns left
    if X_uncorrelated.shape[1] == 0:
        print("Error: 0 features left after correlation drop.")
        return

    pca = PCA(n_components=min(0.95, X_uncorrelated.shape[1]), random_state=42)
    try:
        X_pca = pca.fit_transform(X_uncorrelated)
    except Exception as e:
        print(f"PCA Error: {e}")
        return

    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_class_count = np.max(class_counts)

    X_bal_list = []
    y_bal_list = []

    for c in unique_classes:
        idx = np.where(y == c)[0]
        if len(idx) < max_class_count:
            np.random.seed(42)
            resampled_idx = np.random.choice(idx, size=max_class_count, replace=True)
            X_bal_list.append(X_pca[resampled_idx])
            y_bal_list.append(y.iloc[resampled_idx].values)
        else:
            X_bal_list.append(X_pca[idx])
            y_bal_list.append(y.iloc[idx].values)

    X_balanced = np.vstack(X_bal_list)
    y_balanced = np.concatenate(y_bal_list)

    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(y_balanced))
    X_bal = X_balanced[shuffle_idx]
    y_bal = y_balanced[shuffle_idx]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)
        nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), 
                                 activation='relu', 
                                 solver='adam', 
                                 max_iter=500, 
                                 random_state=42,
                                 early_stopping=True)
        nn_model.fit(X_train, y_train)
        print("Success! Training completed.")
    except Exception as e:
        print("Error in Step 8 (Training):")
        traceback.print_exc()

run()
