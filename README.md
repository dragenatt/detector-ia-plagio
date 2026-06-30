# Veraz · Detector de IA y plagio

Estima la originalidad de un texto académico, su similitud con otras fuentes
(plagio) y la probabilidad de que sea generado por IA. Los resultados son
**estimaciones probabilísticas, no acusaciones**.

---

## 1. Estructura del proyecto

```
Veraz/
├── backend/                  API + motor de análisis (Python)
│   ├── app/
│   │   ├── main.py           API FastAPI (endpoints)
│   │   ├── config.py         Rutas y constantes
│   │   ├── schemas.py        Validación de peticiones (Pydantic)
│   │   ├── extract.py        PDF/DOCX/TXT → texto
│   │   ├── report.py         Reporte PDF (fpdf2)
│   │   ├── db.py             Historial + corpus (SQLite)
│   │   ├── analysis/         === MOTOR (Python puro, sin dependencias) ===
│   │   │   ├── text_utils.py     tokenización, oraciones, léxicos ES
│   │   │   ├── features.py       rasgos estilométricos (vector de features)
│   │   │   ├── ai_detection.py   9 señales explicables → probabilidad de IA
│   │   │   ├── plagiarism.py     TF-IDF + coseno + n-gramas vs. corpus
│   │   │   ├── highlighting.py   puntaje por oración → fragmentos coloreados
│   │   │   ├── originality.py    combina plagio + IA → índice de originalidad
│   │   │   ├── explain.py        traduce los números a lenguaje simple
│   │   │   ├── recommendations.py sugerencias de mejora
│   │   │   └── engine.py         orquesta todo el análisis
│   │   └── model/            === MODELO ENTRENABLE ===
│   │       ├── classifier.py     regresión logística en Python puro
│   │       └── trainer.py        lee el corpus, entrena y guarda el modelo
│   ├── training_data/        corpus etiquetado (humano/ia/mixto/...)
│   ├── models/               modelo entrenado (ai_model.json)
│   ├── tests/test_engine.py  pruebas del motor (sin dependencias)
│   ├── train.py              CLI de entrenamiento
│   ├── run.py                arranque del servidor
│   └── requirements.txt
└── frontend/                 Dashboard (React + Vite + TS + Tailwind)
    └── src/
        ├── App.tsx           orquesta la interfaz
        ├── lib/              cliente de API + tipos + tema claro/oscuro
        └── components/       Header, InputPanel, ScoreRing, HighlightedText…
```

**Decisión de arquitectura clave:** el motor de análisis (`app/analysis` y
`app/model`) es **Python puro, sin dependencias externas**. Esto lo hace
liviano, fácil de testear y deja puntos de extensión claros hacia modelos más
avanzados. FastAPI, la lectura de archivos y el PDF solo se usan en la capa web.

---

## 2 + 3. Cómo funciona cada módulo

### Detección de IA — sistema multi-señal explicable (`ai_detection.py`)
No es una regla única. Calcula **14 señales independientes**, cada una en `[0,1]`
con su explicación, y las combina con pesos:

| Señal | Qué mira |
|-------|----------|
| Frases genéricas / vacías | "cabe destacar", "hoy en día"… |
| Exceso de conectores | "además", "por lo tanto"… en exceso |
| Oraciones que abren con conector | Muchas frases empiezan con "Además,", "Por otro lado," |
| Uniformidad de oraciones | Longitudes demasiado parecidas entre sí |
| Ritmo plano (burstiness) | Falta de alternancia corto/largo |
| Repetición de ideas | n-gramas (trigramas/bigramas) repetidos |
| Aperturas repetitivas | Las oraciones empiezan siempre con las mismas palabras |
| Vocabulario poco variado | Diversidad léxica baja (MATTR, robusto a la longitud) |
| Exceso de cautela / generalidades | Atenuadores "suele", "en general", "podría"… |
| Párrafos de tamaño uniforme | Longitudes de párrafo llamativamente parejas |
| Estructura de listas | Mucho contenido en viñetas/enumeraciones |
| Redacción "demasiado limpia" | Sin marcas informales ni errores humanos |
| Tipografía "pulida" | Rayas (—) y comillas curvas que rara vez teclea alguien |
| Poca voz personal | Ausencia de opinión/experiencia en 1ª persona |

La probabilidad final mezcla estas heurísticas con el **modelo entrenado** (si
existe). Se recorta a `[3 %, 97 %]` para no afirmar nunca certezas absolutas.

> **Aprendizaje mejorado:** el clasificador usa ahora **ponderación de clases
> (`class_weight="balanced"`)** para que el desbalance del corpus (más textos
> de IA que humanos) no sesgue el resultado, y el entrenamiento reporta
> **precisión, recall y F1** además de la exactitud. El corpus humano se amplió
> con ejemplos variados para enseñarle mejor qué *no* es IA.
>
> **Probabilidad calibrada:** el % final se calibra con *Platt scaling* para que
> un "70 %" refleje la frecuencia real (se reporta el error de calibración,
> ECE/Brier). Además, un **peso adaptativo** evita falsos positivos: si la
> estilometría no ve marcas de IA, el modelo no condena por sí solo un texto
> académico humano formal.
>
> **Evaluación honesta:** `python evaluate.py` mide sobre un set apartado
> (`eval_data/`, nunca usado para entrenar) la matriz de confusión, precisión,
> recall, F1 y la calibración. Útil para ver si cada cambio mejora o empeora.

### Resaltado por oración (`highlighting.py`)
Cada oración recibe un puntaje de IA y de plagio. En la interfaz, las oraciones
con marcas de IA se resaltan en **ámbar con intensidad proporcional** al puntaje
(más marcado = más "parece IA"), y las que coinciden con una fuente, en azul.

### Detección de plagio (`plagiarism.py`)
- **Coincidencia textual (n-gramas de 5 palabras):** qué fracción del texto
  aparece en alguna fuente. Es la métrica principal de similitud y permite
  localizar las oraciones copiadas.
- **Similitud temática (TF-IDF + coseno):** parecido de vocabulario ponderado
  por rareza; detecta paráfrasis. Se reporta como señal secundaria.

### Originalidad (`originality.py`)
`originalidad = (1 − similitud) × (1 − 0.7 × probabilidad_IA) × 100`
(la copia penaliza directo; la IA penaliza de forma más suave por ser una
estimación menos certera).

### Modelo entrenable (`model/`)
Regresión logística **escrita a mano** (con estandarización z-score) que aprende
de tu corpus qué combinaciones de rasgos son típicas de IA vs. humano. Se guarda
como JSON. Si no hay modelo, el sistema funciona igual solo con heurísticas.

---

## 4. Cómo correrlo paso a paso

Necesitas **Python 3.8 o superior**. (Node solo hace falta para *desarrollar*
la interfaz; para usar la app no, porque ya viene compilada.)

> **Windows 8 / Windows 10 antiguo:** las versiones nuevas de Python exigen un
> Windows 10 reciente. Si el tuyo está desactualizado, instala **Python 3.8.10**
> (la app es compatible y ese instalador sí funciona en Windows 7/8/8.1):
> <https://www.python.org/downloads/release/python-3810/> → *Windows installer (64-bit)*.

> **Modo fácil (un solo clic):** el backend sirve también la interfaz ya
> compilada (`frontend/dist`), así que basta con ejecutar el backend y abrir
> `http://localhost:8000`. En la raíz del proyecto hay un lanzador para cada
> sistema que prepara todo la primera vez (entorno virtual, dependencias e
> interfaz), arranca el servidor y abre el navegador:
>
> | Sistema | Archivo | Cómo usarlo |
> |---------|---------|-------------|
> | **Windows** | `Veraz.bat` | Doble clic |
> | **macOS / Linux** | `Veraz.command` | Doble clic (en macOS puede pedir confirmación la primera vez: clic derecho → *Abrir*). Desde terminal: `./Veraz.command` |
>
> La primera ejecución tarda un poco (instala dependencias y compila la
> interfaz); las siguientes son casi inmediatas. Para compilar la interfaz
> necesitas Node 18+; sin él la API funciona igual pero la web no se sirve.

### Backend (API + interfaz)
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate           # PowerShell:  .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py                    # API + interfaz en http://127.0.0.1:8000
```
Documentación interactiva de la API en `http://127.0.0.1:8000/docs`.

> El **motor** se puede probar sin instalar nada:
> `python tests/test_engine.py`

### Frontend
- **Para usar la app:** ya viene compilado en `frontend/dist` y lo sirve el
  backend; no necesitas Node en tiempo de ejecución.
- **Para desarrollar la interfaz** (recarga en caliente):
```bash
cd frontend
npm install
npm run dev                      # http://127.0.0.1:5173 (proxy /api -> :8000)
```
Tras cambiar la UI, recompila con `npm run build` para que el backend sirva la
versión nueva.

---

## 5. Cómo cargar textos de entrenamiento
Coloca archivos `.txt` (uno por texto) dentro de `backend/training_data/<etiqueta>`:
`humano`, `ia`, `mixto`, `original`, `plagiado`, `referencias`.
Detalles y tabla de etiquetas: [`backend/training_data/README.md`](backend/training_data/README.md).
Ya se incluyen ejemplos para que funcione desde el primer momento.

## 6. Cómo mejorar el modelo con más datos
1. Agrega más ejemplos (idealmente 20+ por clase, equilibrados y variados).
2. Reentrena: `python train.py` o el botón **“Reentrenar”** de la app.
3. Revisa la precisión que imprime el entrenamiento (entrenamiento y validación).
4. Para un salto de calidad, pasa a embeddings o a un modelo externo (sección 8).

## 7. Cómo generar reportes
Tras analizar, pulsa **“Exportar reporte PDF”** (o `GET /api/report/{id}`). El
PDF incluye los tres puntajes, el resumen, las señales, las recomendaciones y el
aviso de fiabilidad.

---

## 8. Cómo hacerlo más preciso (hoja de ruta)
- **Embeddings semánticos** (sentence-transformers): reemplazar/triangular el
  TF-IDF para detectar paráfrasis y plagio de ideas. Punto de extensión:
  `plagiarism.py` (añadir un `_semantic_similarity`).
- **Perplejidad real** con un modelo de lenguaje (p. ej. vía API): señal fuerte
  de IA. Añádela como una señal más en `ai_detection.SIGNALS`.
- **Búsqueda web / APIs externas** (Bing, Google Programmable Search,
  Copyleaks, Turnitin): comparar contra la web, no solo el corpus local.
  Punto de extensión: `main._all_references()` → añadir un proveedor remoto.
- **Clasificadores más potentes** (scikit-learn, XGBoost) entrenados sobre las
  mismas features de `features.py`.
- **Calibración** de probabilidades con datos reales para que el "70 %" del
  detector signifique de verdad un 70 %.

## 9. Diseño visual
Estética **editorial-académica premium** (nada de "look genérico"):
- Tipografías con carácter: **Fraunces** (títulos), **Hanken Grotesk** (texto),
  **JetBrains Mono** (datos).
- Paleta cálida tipo papel + tinta, con **color semántico**:
  🟢 verde = original/humano · 🔵 azul = fuente/plagio · 🟠 ámbar = IA.
- Tarjetas suaves, bordes redondeados, sombras sutiles, textura de grano.
- **Modo claro y oscuro**, animaciones suaves (anillos que se dibujan,
  apariciones escalonadas) y diseño responsive.

## 10. Buenas prácticas para evitar falsos positivos
- **Lenguaje de probabilidad, no de acusación**: todo se muestra como "riesgo"
  o "probabilidad", con un aviso permanente.
- **Nivel de confianza por longitud**: en textos cortos (<80 palabras) la
  confianza baja y se advierte explícitamente.
- **Recorte de extremos**: nunca se reporta 0 % ni 100 %.
- **Multi-señal**: una sola señal no condena; el resultado es un consenso.
- **Señales débiles bien marcadas**: p. ej. "poca voz personal" tiene poco peso,
  porque muchos textos académicos legítimos evitan la 1ª persona.
- **Transparencia total**: se muestran TODAS las señales y el porqué de cada una.
- **El humano decide**: la herramienta sugiere revisar, no sanciona.

---

### Licencia y propósito
Proyecto educativo / de portafolio. Úsalo de forma ética: como apoyo a la
escritura y la honestidad académica, no como herramienta de castigo.
