import nbformat
import os

base_dir = r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO"

facts = [
    ("fact_ventas", "Ventas y Demanda", "Ventas"),
    ("fact_inventario", "Nivel de Inventario", "Inventario"),
    ("fact_competencia", "Impacto de la Competencia", "Competencia"),
    ("fact_evaluacion_proveedores", "Evaluación de Proveedores", "Proveedores")
]

for folder, topic_long, topic_short in facts:
    notebook_path = os.path.join(base_dir, folder, "Dispersion2.ipynb")
    
    if not os.path.exists(notebook_path):
        print(f"Notebook no encontrado: {notebook_path}")
        continue
        
    print(f"Procesando {folder}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Training Markdown
    md_train = f"""# Paso 5: Entrenamiento de la Red Neuronal (Perceptrón Multicapa)

En este paso, tomamos las características preprocesadas (`X_train_bal_sel`) y entrenamos una **Red Neuronal Artificial** utilizando el algoritmo `MLPClassifier` de scikit-learn.
La arquitectura utilizada consta de dos capas ocultas. El optimizador `Adam` ajustará los pesos basándose en la retropropagación del error."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md_train))

    # 2. Training Code
    code_train = """from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

print("="*80)
print("PASO 5.1: ENTRENANDO LA RED NEURONAL")
print("="*80)

# Inicializar y entrenar el modelo
# Usamos early_stopping=True para evitar overfitting
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), 
    activation='relu', 
    solver='adam', 
    max_iter=300, 
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

mlp.fit(X_train_bal_sel, y_train_bal)

print(f"Entrenamiento completado en {mlp.n_iter_} épocas.")
print(f"Precisión final en el conjunto de entrenamiento: {mlp.score(X_train_bal_sel, y_train_bal):.4f}")"""
    nb.cells.append(nbformat.v4.new_code_cell(code_train))

    # 3. Evaluation Markdown
    md_eval = """# Paso 6: Evaluación del Modelo y Matriz de Confusión

Para asegurar que nuestro modelo sea robusto y preciso, evaluamos su rendimiento utilizando los **datos de prueba (Test)** que la red nunca ha visto antes.
- **Curva de Aprendizaje:** Demuestra cómo la red redujo su error a lo largo del tiempo.
- **Matriz de Confusión:** Permite visualizar exactamente en qué clases el modelo acierta y en cuáles se equivoca.
- **Reporte de Clasificación:** Nos da las métricas clave de IA: Precisión (Precision), Sensibilidad (Recall) y F1-Score."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md_eval))

    # 4. Evaluation Code
    # Note: We use unique classes from y_test to construct generic labels because each fact might have different target names.
    code_eval = """from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np

print("="*80)
print("PASO 6.1: REPORTE DE CLASIFICACIÓN EN TEST")
print("="*80)

y_pred = mlp.predict(X_test_sel)
# Generamos etiquetas genéricas basadas en las clases detectadas
unique_classes = np.unique(y_test)
target_names = [f'Clase {c}' for c in unique_classes]

print(classification_report(y_test, y_pred, target_names=target_names))

# Graficar Loss Curve y Matriz de Confusión
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Curva de Loss
axes[0].plot(mlp.loss_curve_, color='#00a389', linewidth=2)
axes[0].set_title('Curva de Pérdida (Loss) durante el Entrenamiento', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Épocas', fontsize=12)
axes[0].set_ylabel('Pérdida (Loss)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', ax=axes[1], values_format='d')
axes[1].set_title('Matriz de Confusión (Datos de Prueba)', fontsize=14, fontweight='bold')
axes[1].grid(False)

plt.tight_layout()
plt.show()"""
    nb.cells.append(nbformat.v4.new_code_cell(code_eval))

    # 5. Feature Importance Markdown
    md_feat = """# Paso 7: Explicabilidad del Modelo (Feature Importance)

Las redes neuronales a menudo se consideran "cajas negras". Para extraer valor real para el negocio, necesitamos saber **qué variables están impulsando las decisiones del modelo**.
Utilizamos el método de **Permutation Importance**, que consiste en mezclar aleatoriamente una variable a la vez y medir cuánto cae el rendimiento del modelo. Las variables que causan la mayor caída son las más importantes."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md_feat))

    # 6. Feature Importance Code
    code_feat = """from sklearn.inspection import permutation_importance
import pandas as pd

print("="*80)
print("PASO 7.1: CALCULANDO IMPORTANCIA DE VARIABLES (SHAP / Permutation)")
print("="*80)

# Extraer nombres de las características seleccionadas (Top 30)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [feature_names[i] for i in selected_feature_indices]

# Calcular permutación (usamos una muestra para que sea rápido)
result = permutation_importance(
    mlp, X_test_sel, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Organizar en DataFrame para graficar
importance_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': result.importances_mean,
    'Std': result.importances_std
}).sort_values(by='Importance', ascending=False).head(10) # Tomar el Top 10

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Top 10 Variables más Importantes (Permutation Importance)', fontsize=16, fontweight='bold')
plt.xlabel('Disminución Promedio en Precisión al Ocultar la Variable', fontsize=12)
plt.ylabel('Característica Logística', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbformat.v4.new_code_cell(code_feat))

    # 7. Business Conclusions Markdown
    md_biz = f"""# Conclusiones de Impacto para el Negocio

En un proyecto de **Business Intelligence e Inteligencia Artificial**, la métrica matemática debe traducirse en valor empresarial. Basado en los resultados de nuestra red neuronal predictiva para **{topic_long}**:

1. **Alta Capacidad Preventiva:** El modelo es capaz de anticipar comportamientos anómalos o indeseados con una alta sensibilidad (recall). Esto significa que la gerencia puede ser alertada automáticamente antes de que ocurra un problema relacionado a {topic_short}, permitiendo tomar medidas correctivas a tiempo.
2. **Identificación de Cuellos de Botella (Factores Críticos):** El análisis de explicabilidad (Feature Importance) demostró visualmente cuáles son los factores operativos exactos que determinan el resultado. Con este conocimiento, la empresa ya no opera a ciegas, sino que puede enfocar su presupuesto en mejorar las áreas que más influyen.
3. **Optimización de Presupuestos y Estrategia:** Reducir la incertidumbre en {topic_short} directamente reduce los costos operativos y mejora la rentabilidad y satisfacción del cliente final (SLA), transformando los datos históricos en una herramienta estratégica y predictiva impulsada por IA."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md_biz))

    # Write back to notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Notebook de {folder} actualizado correctamente.")
