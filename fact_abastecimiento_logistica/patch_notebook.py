import json

notebook_path = r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_abastecimiento_logistica\red2neuronal.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Actualizar la celda del diagrama
diagram_code = """# Diagrama Estructural de la Red Neuronal Mejorado
import matplotlib.pyplot as plt
import numpy as np

def draw_neural_network_diagram(input_dim, hidden1, hidden2, output_dim, model=None, filename='outputs/diagrama_red_neuronal.png'):
    COLOR_IN  = '#00B4D8' # Azul cyan brillante
    COLOR_H1  = '#F72585' # Rosa neón brillante
    COLOR_H2  = '#FFB703' # Amarillo anaranjado brillante
    COLOR_OUT = '#00F5D4' # Verde aguamarina brillante
    BG_DARK   = '#0b0f19' # Fondo oscuro premium
    BG_PANEL  = '#161B22'
    TEXT_CLR  = '#FFFFFF' # Texto blanco puro
    
    plt.rcParams.update({
        'figure.facecolor': BG_DARK,
        'axes.facecolor':   BG_DARK,
        'text.color':       TEXT_CLR,
    })
    
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.axis('off')
    
    layers = [input_dim, hidden1, hidden2, output_dim]
    layer_names = ['Capa Entrada\\n(Features)', f'Capa Oculta 1\\n({hidden1} Neuronas)', f'Capa Oculta 2\\n({hidden2} Neuronas)', 'Capa Salida\\n(Probabilidades)']
    layer_colors = [COLOR_IN, COLOR_H1, COLOR_H2, COLOR_OUT]
    
    node_positions = []
    
    for l_idx, layer_size in enumerate(layers):
        x = l_idx * 3.0
        positions = []
        
        # Truncar capas grandes para evitar amontonamiento en el gráfico
        if layer_size > 8:
            y_coords = np.linspace(4.0, -4.0, 8)
            for i in range(8):
                if i == 4:
                    positions.append((x, y_coords[i], 'ellipsis'))
                else:
                    positions.append((x, y_coords[i], 'node'))
        else:
            y_coords = np.linspace(2.5, -2.5, layer_size) if layer_size > 1 else [0.0]
            for i in range(layer_size):
                positions.append((x, y_coords[i], 'node'))
                
        node_positions.append(positions)
        
    # Dibujar líneas de conexión con pesos
    for l_idx in range(len(layers) - 1):
        pos_current = node_positions[l_idx]
        pos_next = node_positions[l_idx + 1]
        
        w_matrix = None
        if model is not None:
            if l_idx == 0:
                w_matrix = model.W1
            elif l_idx == 1:
                w_matrix = model.W2
            elif l_idx == 2:
                w_matrix = model.W3
                
        for i_curr, p_curr in enumerate(pos_current):
            if p_curr[2] == 'ellipsis':
                continue
            for i_next, p_next in enumerate(pos_next):
                if p_next[2] == 'ellipsis':
                    continue
                    
                color_line = '#30363D'
                alpha = 0.15
                lw = 1.0
                
                # Mapear a peso correspondiente
                if w_matrix is not None:
                    try:
                        idx_curr = i_curr if i_curr < 4 else w_matrix.shape[0] - 8 + i_curr
                        idx_next = i_next if i_next < 4 else w_matrix.shape[1] - 8 + i_next
                        
                        if idx_curr < w_matrix.shape[0] and idx_next < w_matrix.shape[1]:
                            w_val = w_matrix[idx_curr, idx_next]
                            # Verde brillante para pesos positivos, Rojo brillante para negativos
                            color_line = '#00E676' if w_val > 0 else '#FF1744' 
                            # Incrementar la opacidad y grosor mínimo para que se note mucho más
                            alpha = min(0.85, max(0.15, np.abs(w_val) * 0.6))
                            lw = min(4.5, max(0.8, np.abs(w_val) * 2.5))
                    except:
                        pass
                        
                ax.plot([p_curr[0], p_next[0]], [p_curr[1], p_next[1]], color=color_line, alpha=alpha, lw=lw, zorder=1)
                
    # Graficar los nodos (círculos y elipsis)
    for l_idx, positions in enumerate(node_positions):
        for p in positions:
            if p[2] == 'ellipsis':
                # Reemplazado '⋮' por '...' para evitar el cuadrado en el renderizado
                ax.text(p[0], p[1] + 0.15, '...', ha='center', va='center', fontsize=26, color=TEXT_CLR, fontweight='bold')
            else:
                circle = plt.Circle((p[0], p[1]), 0.16, color=layer_colors[l_idx], ec='white', lw=1.5, zorder=2)
                ax.add_artist(circle)
                
        # Título de cada columna/capa
        ax.text(l_idx * 3.0, 5.0, layer_names[l_idx], ha='center', va='center', fontsize=13, fontweight='bold', color=layer_colors[l_idx])
        
    ax.set_xlim(-1.0, len(layers) * 3.0 - 2.0)
    ax.set_ylim(-6.0, 6.0)
    plt.title('DIAGRAMA DE ARQUITECTURA DE LA RED NEURONAL', fontsize=18, fontweight='bold', color='#FFFFFF', pad=25)
    
    filename = 'outputs/diagrama_red_neuronal.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()
    print(f"¡Diagrama de arquitectura guardado correctamente en: {filename}!")

draw_neural_network_diagram(X_final_train.shape[1], model_final.hidden1, model_final.hidden2, output_dim, model_final)\n"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'draw_neural_network_diagram' in ''.join(cell['source']):
        # Replace only the drawing cell
        cell['source'] = [line + '\\n' for line in diagram_code.split('\\n')[:-1]]
        cell['source'][-1] = cell['source'][-1].rstrip('\\n')
        break

# 2. Añadir Curvas ROC
roc_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Curvas ROC y AUC\n",
        "Estas curvas evalúan el rendimiento del modelo a través de todos los umbrales de clasificación posibles."
    ]
}
roc_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.metrics import roc_curve, auc\n",
        "from sklearn.preprocessing import label_binarize\n",
        "\n",
        "# Binarizar las etiquetas para ROC multiclase\n",
        "y_test_bin = label_binarize(y_test, classes=list(range(output_dim)))\n",
        "\n",
        "plt.figure(figsize=(10, 8))\n",
        "plt.rcParams.update({'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff', 'text.color': '#000000', 'axes.labelcolor': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000'})\n",
        "\n",
        "colors = ['#EF476F', '#06D6A0', '#FFD166']\n",
        "for i in range(output_dim):\n",
        "    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs_test[:, i])\n",
        "    roc_auc = auc(fpr, tpr)\n",
        "    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc:0.2f})')\n",
        "\n",
        "plt.plot([0, 1], [0, 1], 'k--', lw=2)\n",
        "plt.xlim([0.0, 1.0])\n",
        "plt.ylim([0.0, 1.05])\n",
        "plt.xlabel('Tasa de Falsos Positivos')\n",
        "plt.ylabel('Tasa de Verdaderos Positivos')\n",
        "plt.title('Receiver Operating Characteristic (ROC)')\n",
        "plt.legend(loc=\"lower right\")\n",
        "plt.grid(alpha=0.3)\n",
        "plt.savefig('outputs/roc_curve.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]
}

# 3. Añadir Función de Inferencia
inference_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Función de Inferencia (Predicción en vivo)\n",
        "A continuación, definimos una función que toma un registro nuevo y realiza todo el procesamiento y pase hacia adelante (forward pass) para obtener su predicción final."
    ]
}
inference_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def predict_new_sample(features_dict, feature_names, qt_scaler, model):\n",
        "    # 1. Crear DataFrame con el input\n",
        "    df_new = pd.DataFrame([features_dict])\n",
        "    \n",
        "    # 2. Asegurarse de que tenga todas las columnas requeridas, rellenar con 0 si faltan\n",
        "    for col in feature_names:\n",
        "        if col not in df_new.columns:\n",
        "            df_new[col] = 0\n",
        "            \n",
        "    # Ordenar columnas exactamente igual que en el entrenamiento\n",
        "    df_new = df_new[feature_names]\n",
        "    \n",
        "    # 3. Escalar con el QuantileTransformer guardado\n",
        "    X_new_scaled = qt_scaler.transform(df_new.values)\n",
        "    \n",
        "    # Seleccionar las mismas variables importantes (si aplica)\n",
        "    try:\n",
        "        X_new_scaled = X_new_scaled[:, selected_indices]\n",
        "    except NameError:\n",
        "        pass # Si no hay selección de features, se usa todo\n",
        "        \n",
        "    # 4. Predicción Forward Pass\n",
        "    probs = model.forward(X_new_scaled)\n",
        "    pred_class = np.argmax(probs, axis=1)[0]\n",
        "    \n",
        "    return class_names[pred_class], probs[0]\n",
        "\n",
        "# --- EJEMPLO DE USO ---\n",
        "# Tomaremos el primer registro original del test set antes del escalado como ejemplo\n",
        "ejemplo_crudo = pd.DataFrame(X_test, columns=feature_names).iloc[0].to_dict()\n",
        "\n",
        "clase_predicha, probabilidades = predict_new_sample(ejemplo_crudo, feature_names, qt, model_final)\n",
        "print(\"===============================================\")\n",
        "print(f\"Predicción para el nuevo registro: {clase_predicha}\")\n",
        "print(\"===============================================\")\n",
        "print(f\"Probabilidades por clase:\")\n",
        "for i, c_name in enumerate(class_names):\n",
        "    print(f\" - {c_name}: {probabilidades[i]*100:.2f}%\")\n",
        "print(\"===============================================\")\n"
    ]
}

nb['cells'].extend([roc_markdown, roc_code, inference_markdown, inference_code])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched successfully!")
