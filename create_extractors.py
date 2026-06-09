import json
import os
import glob

notebooks = [
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_abastecimiento_logistica\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_competencia\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_evaluacion_proveedores\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_inventario\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_ventas\red2neuronal.ipynb"
]

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_lines = []
    # Collect code up to SelectKBest cell
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_lines.append(source)
            if 'SelectKBest' in source:
                break
    
    # We add code to export the extracted variables
    folder = os.path.dirname(nb_path)
    meta_path = os.path.join(folder, 'outputs', 'inference_model.json')
    
    export_code = f"""
import json
import numpy as np

# Load existing meta
meta_path = r"{meta_path}"
with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

meta['original_feature_names'] = feature_names
meta['selected_feature_indices'] = selected_feature_indices.tolist()

with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("Updated meta in " + meta_path)
"""
    code_lines.append(export_code)
    full_code = "\n".join(code_lines)
    
    temp_script = os.path.join(folder, "temp_extract.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(full_code)
    
    print(f"Created {temp_script}")
