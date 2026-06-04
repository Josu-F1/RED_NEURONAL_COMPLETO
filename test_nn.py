import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

datasets = [
    'fact_abastecimiento_logistica',
    'fact_competencia',
    'fact_evaluacion_proveedores'
]

for ds in datasets:
    print(f"\\n==================================================")
    print(f"Red Neuronal para: {ds}")
    try:
        data = np.load(f"{ds}/outputs/01_processed_data.npz", allow_pickle=True)
        X = data['X_scaled']
        y = data['y']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001,
                           solver='adam', random_state=42, early_stopping=True, n_iter_no_change=10)
        nn.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, nn.predict(X_train))
        test_acc = accuracy_score(y_test, nn.predict(X_test))
        
        print(f"Training Accuracy: {train_acc*100:.2f}%")
        print(f"Testing Accuracy : {test_acc*100:.2f}%")
        print(f"Epochs: {nn.n_iter_}")
    except Exception as e:
        print(f"Error: {e}")
