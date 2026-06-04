import json

# 1. Abastecimiento
try:
    with open('fact_abastecimiento_logistica/01_analisis_correlacion_separabilidad.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "y = ((df['cantidad_recibida']" in line:
                    new_source.append("    ratio = df['cantidad_recibida'] / df['cantidad_solicitada']\n")
                    new_source.append("    y = pd.cut(ratio, bins=3, labels=[0, 1, 2]).astype(int)\n")
                elif "print(f\"✅ Target creado: Entrega Satisfactoria" in line:
                    new_source.append("    print(f\"✅ Target MULTICLASE creado automáticamente (3 clases)\")\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
    with open('fact_abastecimiento_logistica/01_analisis_correlacion_separabilidad.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
except Exception as e:
    print('Error abastecimiento:', e)

# 2. Evaluacion
try:
    with open('fact_evaluacion_proveedores/01_analisis_correlacion_separabilidad.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "y = (df[numeric_cols[0]] > df[numeric_cols[0]].median()).astype(int)" in line:
                    new_source.append("    y = pd.cut(df[numeric_cols[0]], bins=3, labels=[0, 1, 2]).astype(int)\n")
                elif "print(f\"✅ Target binario creado\")" in line:
                    new_source.append("    print(f\"✅ Target MULTICLASE creado automáticamente (3 clases)\")\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
    with open('fact_evaluacion_proveedores/01_analisis_correlacion_separabilidad.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
except Exception as e:
    print('Error evaluacion:', e)

print("Modificaciones realizadas!")
