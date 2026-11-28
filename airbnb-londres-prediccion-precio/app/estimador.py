"""Lógica del estimador de precios (sin dependencia de Streamlit, para poder probarla por separado)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent / 'modelo'

ETIQUETAS_ALOJAMIENTO = {
    'Entire home/apt': 'Casa o apartamento completo',
    'Private room': 'Habitación privada',
    'Shared room': 'Habitación compartida',
    'Hotel room': 'Habitación de hotel',
}


def cargar_modelo():
    """Carga el modelo entrenado y sus metadatos."""
    modelo = joblib.load(BASE / 'modelo_gb.joblib')
    with open(BASE / 'metadatos.json', encoding='utf-8') as f:
        meta = json.load(f)
    return modelo, meta


def construir_fila(meta, entradas):
    """Arma la fila de 67 variables que espera el modelo.

    `entradas` es un diccionario con las variables numéricas que se quieren fijar (con el nombre
    exacto de la columna) y tres claves categóricas: 'room_type', 'property_type' y 'barrio'.
    Lo que no se indica se rellena con la mediana del conjunto de entrenamiento.
    """
    fila = {c: 0.0 for c in meta['columnas']}
    fila.update(meta['defectos'])
    for clave, valor in entradas.items():
        if clave in fila:
            fila[clave] = float(valor)
    for prefijo, valor in [('room_type', entradas.get('room_type')),
                           ('property_type_agr', entradas.get('property_type')),
                           ('neighbourhood_cleansed', entradas.get('barrio'))]:
        columna = f'{prefijo}_{valor}'
        if columna in fila:            # la categoría de referencia no tiene columna: queda todo en 0
            fila[columna] = 1.0
    return pd.DataFrame([fila], columns=meta['columnas'])


def estimar(modelo, meta, entradas):
    """Devuelve el precio estimado por noche y su rango probable (80 % de los casos)."""
    ln_precio = float(modelo.predict(construir_fila(meta, entradas))[0])
    precio = float(np.exp(ln_precio))
    return precio, precio * meta['cociente_q10'], precio * meta['cociente_q90']
