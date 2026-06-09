import json

notebooks = [
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_abastecimiento_logistica\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_competencia\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_evaluacion_proveedores\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_inventario\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_ventas\red2neuronal.ipynb"
]

diagram_code = r"""# Diagrama Estructural de la Red Neuronal Mejorado
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
    layer_names = ['Capa Entrada\n(Features)', f'Capa Oculta 1\n({hidden1} Neuronas)', f'Capa Oculta 2\n({hidden2} Neuronas)', 'Capa Salida\n(Probabilidades)']
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

draw_neural_network_diagram(X_final_train.shape[1], model_final.hidden1, model_final.hidden2, output_dim, model_final)
"""

for path in notebooks:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        # Find and replace the diagram drawing code
        for cell in nb['cells']:
            if cell['cell_type'] == 'code' and 'draw_neural_network_diagram' in ''.join(cell['source']):
                # Properly format into a list of strings ending with \n
                new_source = [line + '\n' for line in diagram_code.split('\n')]
                new_source[-1] = new_source[-1].rstrip('\n')
                cell['source'] = new_source
                break
                
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Fixed: {path}")
    except Exception as e:
        print(f"Error processing {path}: {e}")
