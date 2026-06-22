# Corpus de entrenamiento

Esta carpeta es la **base de aprendizaje** de Veraz. Cada subcarpeta es una
etiqueta. Coloca dentro archivos `.txt` (o `.md`), **un texto por archivo**.

```
training_data/
├── humano/        Textos escritos por personas (redacciones, ensayos reales)
├── ia/            Textos generados por IA (ChatGPT, Gemini, etc.)
├── mixto/         Textos en parte humanos y en parte de IA
├── original/      Textos humanos originales y de calidad (cuentan como humano)
├── plagiado/      Textos copiados de una fuente (cuentan como humano + sirven de fuente)
└── referencias/   Documentos "fuente" para comparar plagio (NO entrenan el modelo)
```

## Cómo se usan las etiquetas

| Carpeta      | Entrena IA | Objetivo (humano=0 / IA=1) | Corpus de plagio |
|--------------|-----------|----------------------------|------------------|
| `humano`     | sí        | 0                          | no               |
| `original`   | sí        | 0                          | no               |
| `plagiado`   | sí        | 0 (es texto humano copiado)| **sí**           |
| `ia`         | sí        | 1                          | no               |
| `mixto`      | no (se reserva para validar) | —             | no               |
| `referencias`| no        | —                          | **sí**           |

> `plagiado` y `referencias` forman el corpus contra el que se mide la
> similitud. Para que la detección de copia tenga sentido, el archivo "fuente"
> debe estar en `referencias/` y la "copia" en `plagiado/`.

## Cómo cargar textos de entrenamiento

1. Copia archivos `.txt` dentro de la carpeta correspondiente.
2. Cuantos más y más variados, mejor (idealmente 20+ por clase).
3. Reentrena el modelo:

```bash
python train.py
```

   o desde la app pulsando **“Reentrenar modelo”** (llama a `POST /api/train`).

## Cómo mejorar el modelo con más datos

- **Equilibra las clases**: cantidades parecidas de `humano` e `ia`.
- **Varía el origen**: distintos autores, temas, niveles y modelos de IA.
- **Incluye “mixto”** para validar que el sistema no es binario en exceso.
- Revisa la precisión que imprime `train.py` (entrenamiento y validación).
- Para un salto de calidad, conecta features de embeddings o un modelo
  externo (ver la sección "Cómo hacerlo más preciso" del README principal).

> Recuerda: ningún detector es infalible. Más y mejores datos reducen los
> falsos positivos, pero el resultado siempre es una **estimación**.
