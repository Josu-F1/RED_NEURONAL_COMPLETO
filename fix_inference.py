import json

notebooks = [
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_abastecimiento_logistica\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_competencia\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_evaluacion_proveedores\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_inventario\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_ventas\red2neuronal.ipynb"
]

for path in notebooks:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        # Find the inference cell and replace selected_indices with selected_feature_indices
        for cell in nb['cells']:
            if cell['cell_type'] == 'code' and 'def predict_new_sample' in ''.join(cell['source']):
                new_source = []
                for line in cell['source']:
                    if 'X_new_scaled[:, selected_indices]' in line:
                        new_source.append(line.replace('selected_indices', 'selected_feature_indices'))
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                break
                
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Fixed: {path}")
    except Exception as e:
        print(f"Error processing {path}: {e}")
