import re

html_file = r"c:\Users\Marlon\Documents\Sexto\BI\MiProyecto\RED_NEURONAL_COMPLETO\templates\index.html"

with open(html_file, 'r', encoding='utf-8') as f:
    content = f.read()

new_ui = """    <style>
        :root {
            --bg-page: #f4f7f6;
            --bg-panel: #ffffff;
            --bg-card: #ffffff;
            --border-clr: #e5e7eb;
            --text-main: #111827;
            --text-muted: #6b7280;
            --accent: #00a389; /* Green from the image */
            --accent-glow: rgba(0, 163, 137, 0.15);
            --accent-light: #e6f6f4;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --radius: 16px;
            --radius-sm: 8px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-page);
            color: var(--text-main);
            display: flex;
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: 280px;
            background-color: var(--bg-panel);
            border-right: 1px solid var(--border-clr);
            display: flex;
            flex-direction: column;
            padding: 2rem 1.5rem;
            flex-shrink: 0;
            z-index: 10;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 2.5rem;
            padding-left: 0.5rem;
        }

        .logo-icon {
            width: 32px;
            height: 32px;
            background-color: var(--accent);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-weight: 700;
            font-size: 1.2rem;
            box-shadow: 0 4px 10px var(--accent-glow);
        }

        .logo-text {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.5px;
        }

        .menu-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-bottom: 1rem;
            font-weight: 600;
            padding-left: 0.5rem;
        }

        .menu-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .menu-item {
            padding: 0.875rem 1rem;
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-weight: 500;
            font-size: 0.95rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            transition: var(--transition);
        }

        .menu-item:hover {
            color: var(--text-main);
            background-color: #f9fafb;
        }

        .menu-item.active {
            color: var(--accent);
            background-color: var(--accent-light);
            font-weight: 600;
        }

        .menu-item-icon {
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .menu-item-icon svg {
            width: 18px;
            height: 18px;
            fill: currentColor;
            opacity: 0.7;
        }
        .menu-item.active .menu-item-icon svg {
            opacity: 1;
        }

        /* Main Content */
        .main-content {
            flex-grow: 1;
            padding: 2.5rem 3rem;
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            overflow-y: auto;
            width: 100%;
        }

        header {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.5px;
        }

        .subtitle {
            font-size: 0.95rem;
            color: var(--text-muted);
        }

        /* Dashboard Grid Layout */
        .dash-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 2rem;
            align-items: start;
        }

        @media (max-width: 1024px) {
            .dash-grid { grid-template-columns: 1fr; }
        }

        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-clr);
            border-radius: var(--radius);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            box-shadow: var(--shadow-sm);
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--text-main);
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-clr);
        }

        .arch-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.35rem 0.75rem;
            background-color: var(--bg-page);
            color: var(--text-muted);
            border-radius: 20px;
            border: 1px solid var(--border-clr);
        }

        /* Dynamic Form Layout */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        @media (max-width: 640px) {
            .form-grid { grid-template-columns: 1fr; }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group.full-width { grid-column: span 2; }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
            text-transform: capitalize;
        }

        input[type="number"], select {
            width: 100%;
            background-color: #fff;
            border: 1px solid #d1d5db;
            border-radius: var(--radius-sm);
            padding: 0.75rem 1rem;
            font-family: inherit;
            font-size: 0.95rem;
            color: var(--text-main);
            transition: var(--transition);
            outline: none;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02) inset;
        }

        input[type="number"]:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        select {
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%236b7280'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 1rem center;
            background-size: 1.25rem;
            padding-right: 2.5rem;
            cursor: pointer;
        }

        .btn-predict {
            width: 100%;
            margin-top: 1rem;
            background-color: var(--accent);
            color: #fff;
            border: none;
            border-radius: var(--radius-sm);
            padding: 1rem;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 10px var(--accent-glow);
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }

        .btn-predict:hover {
            background-color: #008c75;
            transform: translateY(-1px);
            box-shadow: 0 6px 15px var(--accent-glow);
        }

        .btn-predict:active {
            transform: translateY(0);
        }

        /* Results Card */
        .result-empty {
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            text-align: center;
            padding: 3rem 2rem;
            color: var(--text-muted);
            background-color: var(--bg-page);
            border-radius: var(--radius-sm);
            border: 1px dashed #d1d5db;
        }

        .result-empty-icon {
            font-size: 2.5rem;
            opacity: 0.5;
            margin-bottom: 0.5rem;
        }

        .result-area {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .pred-card {
            background-color: var(--bg-page);
            border-radius: var(--radius-sm);
            padding: 1.5rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border-clr);
        }

        .pred-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent);
        }

        .pred-card.status-high::before { background: var(--success); }
        .pred-card.status-med::before { background: var(--warning); }
        .pred-card.status-low::before { background: var(--danger); }

        .pred-label {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }

        .pred-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .pred-confidence {
            font-size: 0.95rem;
            font-weight: 600;
        }
        .pred-card.status-high .pred-confidence { color: var(--success); }
        .pred-card.status-med .pred-confidence { color: var(--warning); }
        .pred-card.status-low .pred-confidence { color: var(--danger); }

        /* Probabilities List */
        .prob-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .prob-item {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .prob-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-main);
        }

        .prob-bar-container {
            width: 100%;
            height: 8px;
            background-color: var(--bg-page);
            border-radius: 4px;
            overflow: hidden;
        }

        .prob-bar {
            height: 100%;
            border-radius: 4px;
            width: 0%;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            background-color: var(--accent);
        }

        .prob-item:nth-child(1) .prob-bar { background-color: var(--danger); }
        .prob-item:nth-child(2) .prob-bar { background-color: var(--warning); }
        .prob-item:nth-child(3) .prob-bar { background-color: var(--success); }

        /* Loading Spinner */
        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-top: 3px solid #fff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <!-- Sidebar / Model Menu -->
    <div class="sidebar">
        <div class="logo-area">
            <div class="logo-icon">M</div>
            <div class="logo-text">Predictores</div>
        </div>
        <div class="menu-title">Modelos Disponibles</div>
        <ul class="menu-list">
            {% for model in available_models %}
            <li class="menu-item {% if loop.first %}active{% endif %}" data-model-id="{{ model.id }}" onclick="selectModel('{{ model.id }}')">
                <div class="menu-item-icon">
                    <svg viewBox="0 0 24 24"><path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/></svg>
                </div>
                <span>{{ model.name }}</span>
            </li>
            {% endfor %}
        </ul>
    </div>

    <!-- Main Content Area -->
    <div class="main-content">
        <header>
            <h1 id="model-title-main">Modelo</h1>
            <div class="subtitle" id="model-subtitle-main">Selecciona un modelo y rellena los campos para simular y predecir.</div>
        </header>

        <div class="dash-grid">
            <!-- Form Card -->
            <div class="card">
                <div class="card-title">
                    <span>Parámetros de Entrada</span>
                    <span class="arch-badge" id="arch-badge">Capa 1: -- Neuronas, Capa 2: -- Neuronas</span>
                </div>
                <form id="prediction-form" onsubmit="submitPrediction(event)">
                    <div class="form-grid" id="form-fields-container">
                        <!-- Dynamic inputs populated via JS -->
                    </div>
                    <button type="submit" class="btn-predict">
                        <span class="spinner" id="btn-spinner"></span>
                        <span id="btn-text">Realizar Predicción</span>
                    </button>
                </form>
            </div>

            <!-- Result Card -->
            <div class="card">
                <div class="card-title">Resultado de Predicción</div>
                <div id="result-container" style="height:100%">
                    <div class="result-empty">
                        <div class="result-empty-icon">📊</div>
                        <div>Completa el formulario de la izquierda y haz clic en "Realizar Predicción" para ver los resultados.</div>
                    </div>
                </div>
            </div>
        </div>
    </div>"""

# Replace block from <style> to <!-- JS Logic -->
pattern = re.compile(r'    <style>.*?</head>\s*<body>.*?</div>\s*</div>\s*(?=    <!-- JS Logic -->)', re.DOTALL)
new_content = pattern.sub(new_ui, content)

with open(html_file, 'w', encoding='utf-8') as f:
    f.write(new_content)
