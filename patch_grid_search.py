import json
import glob

notebooks = [
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_abastecimiento_logistica\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_competencia\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_evaluacion_proveedores\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_inventario\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_ventas\red2neuronal.ipynb",
    r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\fact_abastecimiento_logistica\red_neuronal_abastecimiento.ipynb"
]

replacement_source = [
    "# Validación Cruzada de 5 Folds probando exactamente 5 hiperparámetros (25 iteraciones totales)\n",
    "# Configuraciones a probar: (Learning Rate, Lambda L2, Arquitectura)\n",
    "hyperparameter_grid = [\n",
    "    (0.1, 0.001, (64, 32)),   # Prueba 1\n",
    "    (0.05, 0.001, (32, 20)),  # Prueba 2\n",
    "    (0.01, 0.001, (16, 10)),  # Prueba 3\n",
    "    (0.1, 0.01, (32, 20)),    # Prueba 4\n",
    "    (0.05, 0.01, (16, 10))    # Prueba 5\n",
    "]\n",
    "output_dim = len(np.unique(y_train_bal))\n",
    "\n",
    "best_acc = 0.0\n",
    "best_hparams = None\n",
    "kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "\n",
    "print(\"=\"*75)\n",
    "print(\"  VALIDACIÓN CRUZADA EN 5 FOLDS (5 CONFIGURACIONES = 25 ENTRENAMIENTOS)\")\n",
    "print(\"=\"*75)\n",
    "\n",
    "for idx_conf, (lr, lam, arch) in enumerate(hyperparameter_grid):\n",
    "    h1, h2 = arch\n",
    "    fold_accs = []\n",
    "    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_bal_sel, y_train_bal)):\n",
    "        X_tr, X_te = X_train_bal_sel[train_idx], X_train_bal_sel[val_idx]\n",
    "        y_tr, y_te = y_train_bal[train_idx], y_train_bal[val_idx]\n",
    "        \n",
    "        if output_dim == 1:\n",
    "            Y_tr = y_tr.reshape(-1, 1).astype(float)\n",
    "            Y_te = y_te.reshape(-1, 1).astype(float)\n",
    "        else:\n",
    "            Y_tr = to_one_hot(y_tr, output_dim)\n",
    "            Y_te = to_one_hot(y_te, output_dim)\n",
    "        \n",
    "        model = NeuralNetworkMLP(X_tr.shape[1], h1, h2, output_dim, random_state=42 + fold)\n",
    "        \n",
    "        for epoch in range(150):\n",
    "            probs = model.forward(X_tr)\n",
    "            model.backward(X_tr, Y_tr, probs, lambda_l2=lam)\n",
    "            model.update_weights(lr=lr)\n",
    "            \n",
    "        probs_te = model.forward(X_te)\n",
    "        if output_dim == 1:\n",
    "            preds_te = (probs_te > 0.5).astype(int).flatten()\n",
    "        else:\n",
    "            preds_te = np.argmax(probs_te, axis=1)\n",
    "            \n",
    "        acc_te = np.mean(preds_te == y_te)\n",
    "        fold_accs.append(acc_te)\n",
    "        \n",
    "    mean_acc = np.mean(fold_accs)\n",
    "    print(f\"  Config {idx_conf+1}/5 | Arch: {str(arch):<8} | LR: {lr:<5} | L2: {lam:<5} | Acc CV Promedio: {mean_acc*100:6.2f}%\")\n",
    "    \n",
    "    if mean_acc > best_acc:\n",
    "        best_acc = mean_acc\n",
    "        best_hparams = (lr, lam, arch)\n",
    "\n",
    "print(\"=\"*75)\n",
    "print(f\"MEJOR CONFIGURACIÓN GANADORA: LR = {best_hparams[0]}, L2 = {best_hparams[1]}, ARCH = {best_hparams[2]}\")\n",
    "print(f\"Accuracy promedio en Validación Cruzada: {best_acc*100:.2f}%\")\n",
    "print(\"=\"*75)\n"
]

for path in notebooks:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        for cell in nb['cells']:
            src = ''.join(cell['source'])
            if cell['cell_type'] == 'code' and ('Validacin Cruzada de 5 Folds' in src or 'Validación Cruzada de 5 Folds' in src or 'kf = StratifiedKFold' in src):
                if 'architectures =' in src or 'hyperparameter_grid =' in src:
                    # Update variables
                    if 'X_bal' in src:
                        # For red_neuronal_abastecimiento.ipynb which uses X_bal instead of X_train_bal_sel
                        my_source = [s.replace('X_train_bal_sel', 'X_bal').replace('y_train_bal', 'y_bal') for s in replacement_source]
                        cell['source'] = my_source
                    else:
                        cell['source'] = replacement_source
                    break
                    
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Patched 5-config CV in {path}")
    except Exception as e:
        print(f"Error processing {path}: {e}")
