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

# Ejemplos que se pueden cargar con un clic
EJEMPLOS = {
    'Habitación privada económica': dict(tipo='Private room', propiedad='Private room in home', barrio='Croydon',
                                         capacidad=2, habitaciones=1, banos=1.0, compartido=True, comodidades=12),
    'Apartamento para una pareja en el centro': dict(tipo='Entire home/apt', propiedad='Entire rental unit', barrio='Westminster',
                                                     capacidad=2, habitaciones=1, banos=1.0, compartido=False, comodidades=25),
    'Apartamento familiar': dict(tipo='Entire home/apt', propiedad='Entire rental unit', barrio='Camden',
                                 capacidad=4, habitaciones=2, banos=1.0, compartido=False, comodidades=30),
    'Casa grande de lujo': dict(tipo='Entire home/apt', propiedad='Entire home', barrio='Kensington and Chelsea',
                                capacidad=8, habitaciones=4, banos=3.0, compartido=False, comodidades=50),
    'Habitación de hotel': dict(tipo='Hotel room', propiedad='Otros', barrio='Westminster',
                                capacidad=2, habitaciones=1, banos=1.0, compartido=False, comodidades=20),
}

# Características cuyo efecto se muestra y el valor «típico» contra el que se comparan
CARACTERISTICAS = {
    'accommodates': 'Capacidad (huéspedes)',
    'bedrooms': 'Habitaciones',
    'banos': 'Baños',
    'bano_compartido': 'Baño compartido',
    'n_amenities': 'Número de comodidades',
    'dist_centro_km': 'Distancia al centro (km)',
    'host_is_superhost': 'Anfitrión superhost',
}

CURVAS = {
    'Capacidad (huéspedes)': ('accommodates', list(range(1, 13))),
    'Distancia al centro (km)': ('dist_centro_km', [round(x, 1) for x in np.arange(0.5, 30.5, 1.5)]),
    'Número de comodidades': ('n_amenities', list(range(0, 61, 4))),
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


def _predecir(modelo, meta, lista_entradas):
    """Predice el precio de varias combinaciones a la vez (una sola llamada al modelo)."""
    filas = pd.concat([construir_fila(meta, e) for e in lista_entradas], ignore_index=True)
    return np.exp(modelo.predict(filas))


def estimar(modelo, meta, entradas):
    """Devuelve el precio estimado por noche y su rango probable (80 % de los casos)."""
    precio = float(_predecir(modelo, meta, [entradas])[0])
    return precio, precio * meta['cociente_q10'], precio * meta['cociente_q90']


# ---------------------------------------------------------------------------
# Coherencia de los datos
# ---------------------------------------------------------------------------
def tipos_propiedad_para(meta, tipo):
    """Tipos de propiedad que tienen sentido para un tipo de alojamiento."""
    todos = meta['tipos_propiedad']
    if tipo == 'Entire home/apt':
        opciones = [t for t in todos if t.startswith('Entire')]
    elif tipo == 'Private room':
        opciones = [t for t in todos if t.startswith('Private room')]
    else:
        opciones = []
    if 'Otros' in todos:
        opciones.append('Otros')
    return opciones or list(todos)


def propiedad_por_defecto(meta, tipo):
    opciones = tipos_propiedad_para(meta, tipo)
    tipica = meta.get('propiedad_tipica', {}).get(tipo)
    return tipica if tipica in opciones else opciones[0]


def validar_entradas(e):
    """Avisos sobre combinaciones poco habituales (no impiden calcular el precio)."""
    avisos = []
    hab, cap, banos, tipo = e['bedrooms'], e['accommodates'], e['banos'], e['room_type']
    if hab > cap:
        avisos.append('Hay más habitaciones que huéspedes: revisa la capacidad.')
    if cap > 4 * hab:
        avisos.append('Son muchos huéspedes para tan pocas habitaciones.')
    if tipo in ('Private room', 'Shared room', 'Hotel room') and hab > 2:
        avisos.append('Una habitación privada, compartida o de hotel normalmente tiene 1 habitación.')
    if banos > hab + 2:
        avisos.append('Son muchos baños para tan pocas habitaciones.')
    return avisos


# ---------------------------------------------------------------------------
# Comparación con anuncios parecidos
# ---------------------------------------------------------------------------
def comparables(meta, barrio, tipo):
    """Precios de anuncios parecidos: mismo barrio y tipo; si hay pocos, solo el barrio o solo el tipo."""
    c = meta['comparables']
    for clave, nivel in [(f'{barrio}|{tipo}', 'del mismo barrio y tipo de alojamiento'),
                         (f'{barrio}|*', 'del mismo barrio'), (f'*|{tipo}', 'del mismo tipo de alojamiento')]:
        if clave in c:
            return dict(c[clave], nivel=nivel)
    return None


# ---------------------------------------------------------------------------
# Qué empuja el precio (efecto de cada característica frente a un alojamiento típico)
# ---------------------------------------------------------------------------
def valores_tipicos(meta, barrio):
    return {'accommodates': 2, 'bedrooms': 1, 'banos': 1.0, 'bano_compartido': 0,
            'n_amenities': meta['defectos']['n_amenities'], 'dist_centro_km': meta['barrios'][barrio]['dist_km'],
            'host_is_superhost': 0}


def contribuciones(modelo, meta, entradas):
    """Cuánto cambia el precio si cada característica tuviera su valor típico (el resto igual)."""
    tipicos = valores_tipicos(meta, entradas['barrio'])
    precio = float(_predecir(modelo, meta, [entradas])[0])
    filas, casos = [], []
    for clave in CARACTERISTICAS:
        valor = entradas.get(clave, meta['defectos'].get(clave, 0))
        if abs(float(valor) - float(tipicos[clave])) > 1e-9:
            filas.append((clave, valor))
            casos.append(dict(entradas, **{clave: tipicos[clave]}))
    if not casos:
        return pd.DataFrame(columns=['Característica', 'Tu valor', 'Valor típico', 'Efecto en el precio'])
    sin = _predecir(modelo, meta, casos)
    df = pd.DataFrame({'Característica': [CARACTERISTICAS[c] for c, _ in filas],
                       'Tu valor': [round(float(v), 1) for _, v in filas],
                       'Valor típico': [round(float(tipicos[c]), 1) for c, _ in filas],
                       'Efecto en el precio': precio - sin})
    return df.sort_values('Efecto en el precio', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Comparar barrios y ver cómo cambia el precio
# ---------------------------------------------------------------------------
def precios_por_barrio(modelo, meta, entradas):
    """Precio estimado del mismo alojamiento en cada barrio (con la distancia típica de cada barrio)."""
    casos, nombres = [], []
    for barrio, d in meta['barrios'].items():
        casos.append(dict(entradas, barrio=barrio, latitude=d['lat'], longitude=d['lon'], dist_centro_km=d['dist_km']))
        nombres.append(barrio)
    precios = _predecir(modelo, meta, casos)
    return pd.DataFrame({'Precio estimado': precios}, index=nombres).sort_values('Precio estimado', ascending=False)


def curva(modelo, meta, entradas, etiqueta):
    """Precio estimado al variar una característica (capacidad, distancia o comodidades)."""
    clave, valores = CURVAS[etiqueta]
    precios = _predecir(modelo, meta, [dict(entradas, **{clave: v}) for v in valores])
    return pd.DataFrame({'Precio estimado': precios}, index=valores)
