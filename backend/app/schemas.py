"""Modelos Pydantic para validar las peticiones de la API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    # 'model_weight' empieza por 'model_', que Pydantic v2 reserva; lo liberamos.
    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., min_length=1, description="Texto a analizar.")
    # Optional[str] (no 'str | None') para compatibilidad con Python 3.8/3.9.
    title: Optional[str] = Field(None, description="Título opcional para el historial.")
    use_model: bool = Field(True, description="Usar el modelo entrenado si existe.")
    model_weight: float = Field(0.5, ge=0.0, le=1.0,
                                description="Peso del modelo vs. heurísticas.")
    save: bool = Field(True, description="Guardar el análisis en el historial.")


class ReferenceIn(BaseModel):
    name: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class CorpusExampleIn(BaseModel):
    """Ejemplo etiquetado que el usuario añade para enseñarle al detector."""
    text: str = Field(..., min_length=1, description="Texto de ejemplo.")
    label: str = Field(..., description="Etiqueta: humano, ia o mixto.")
