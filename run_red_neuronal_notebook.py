import json
import os
import sys
import traceback
import matplotlib
# Configurar Agg backend para evitar bloqueos por ventanas GUI de matplotlib
matplotlib.use('Agg')

folder_path = r"c:\Users\ASUS\OneDrive\Escritorio\data_set_refinado\RED_NEURONAL"
notebook_file = "red_neuronal_ventas.ipynb"
notebook_path = os.path.join(folder_path, notebook_file)

print(f"Ejecutando celdas de: {notebook_path}...")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cambiar de directorio para que las rutas locales funcionen relativas a la carpeta del notebook
orig_cwd = os.getcwd()
os.chdir(folder_path)

globals_dict = {
    '__name__': '__main__',
    '__file__': notebook_file
}

cell_idx = 1
success = True

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        
        # Limpiar comandos mágicos o de consola si los hay
        clean_lines = []
        for line in source_lines:
            stripped = line.strip()
            if stripped.startswith('%') or stripped.startswith('!'):
                continue
            clean_lines.append(line)
            
        clean_code_str = "".join(clean_lines)
        if not clean_code_str.strip():
            continue
            
        try:
            exec(clean_code_str, globals_dict)
            cell["execution_count"] = cell_idx
            cell["outputs"] = []
            cell_idx += 1
        except Exception as e:
            print(f"[ERROR] Error en la celda de código {cell_idx}:")
            traceback.print_exc()
            success = False
            break

# Volver a directorio original
os.chdir(orig_cwd)

if success:
    print("[SUCCESS] Ejecución exitosa de todas las celdas de la Red Neuronal!")
    # Guardar el notebook ejecutado
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
else:
    print("[ERROR] Falló la ejecución del notebook de la Red Neuronal.")
