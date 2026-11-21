#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicción del precio de alojamientos de Airbnb en Londres (septiembre 2022)
Metodología CRISP-DM: Fase 2 (comprensión de los datos), Fase 3 (preparación),
Fase 4 (modelado) y Fase 5 (evaluación).

Este script reproduce, de principio a fin, el mismo análisis del cuaderno de Google Colab
(notebooks/Proyecto_CRISP-DM_Airbnb_Londres_Colab_Fase5_CSV.ipynb).

Uso:
    python src/modelado_airbnb_londres.py --datos data/clean_df.csv --salida results

Resultados:
    results/figuras/  gráficos en PNG
    results/tablas/   tablas de resultados en CSV
Tiempo aproximado: 10 a 15 minutos (la búsqueda de parámetros es la parte más lenta).
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Modelado del precio de alojamientos de Airbnb en Londres')
parser.add_argument('--datos', default='data/clean_df.csv', help='ruta del archivo clean_df.csv')
parser.add_argument('--salida', default='results', help='carpeta donde se guardan figuras y tablas')
parser.add_argument('--modelo-app', default='app/modelo', help='carpeta donde se guarda el modelo final para el estimador interactivo')
parser.add_argument('--rapido', action='store_true', help='prueba rápida con una muestra de 25.000 anuncios (los resultados cambian)')
args = parser.parse_args()

if not os.path.exists(args.datos):
    sys.exit(f'No se encontró el archivo de datos: {args.datos}\nDescárgalo de Kaggle (ver data/README.md).')

os.makedirs(os.path.join(args.salida, 'figuras'), exist_ok=True)
os.makedirs(os.path.join(args.salida, 'tablas'), exist_ok=True)

import matplotlib
matplotlib.use('Agg')            # permite ejecutar sin pantalla
import matplotlib.pyplot as plt

_celda = 'inicio'
_contador = [0]

def _guardar_figura(*a, **k):
    # Sustituye a plt.show(): guarda cada gráfico como PNG numerado
    _contador[0] += 1
    ruta = os.path.join(args.salida, 'figuras', f'{_contador[0]:02d}_{_celda}.png')
    plt.gcf().savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')

plt.show = _guardar_figura

# ==========================================================================
# 0. CONFIGURACIÓN INICIAL
# ==========================================================================
_celda = 'imports'
import pandas as pd                 # manejo de tablas de datos
import numpy as np                  # operaciones numéricas
import matplotlib.pyplot as plt     # gráficos
import seaborn as sns               # gráficos estadísticos
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# Paleta de colores del proyecto
AZUL, CORAL, VERDE = '#1B2A41', '#E4572E', '#2A9D8F'

# ==========================================================================
# 0. CONFIGURACIÓN INICIAL
# ==========================================================================
_celda = 'carga'
RUTA = args.datos

df = pd.read_csv(RUTA)

# En el CSV las fechas vienen como texto: se convierten a formato fecha
for col in ['host_since', 'first_review', 'last_review']:
    df[col] = pd.to_datetime(df[col], errors='coerce')

if args.rapido:
    df = df.sample(25000, random_state=42).reset_index(drop=True)   # modo de prueba rápida

print('Filas y columnas:', df.shape)

# ==========================================================================
# 2.1 Descripción general del conjunto de datos
# ==========================================================================
_celda = 'head'
print(df.head())

# ==========================================================================
# 2.1 Descripción general del conjunto de datos
# ==========================================================================
_celda = 'info'
df.info()

# ==========================================================================
# 2.2 Diccionario de las variables críticas
# ==========================================================================
_celda = 'diccionario'
diccionario = pd.DataFrame({
    'Variable': ['price', 'room_type', 'property_type', 'neighbourhood_cleansed', 'latitude / longitude',
                 'accommodates', 'bedrooms', 'beds', 'minimum_nights', 'number_of_reviews',
                 'review_scores_rating', 'host_is_superhost', 'amenities'],
    'Tipo': ['Numérica (objetivo, viene como texto)', 'Categórica', 'Categórica', 'Categórica', 'Geográfica',
             'Numérica', 'Numérica', 'Numérica', 'Numérica', 'Numérica',
             'Numérica', 'Categórica (t/f)', 'Texto'],
    'Descripción': ['Precio por noche', 'Tipo de alojamiento (casa completa, habitación privada, compartida, hotel)',
                    'Tipo de propiedad (100 categorías)', 'Barrio de Londres (33 categorías)', 'Ubicación del anuncio',
                    'Capacidad en huéspedes', 'Número de habitaciones', 'Número de camas',
                    'Noches mínimas por reserva', 'Total de reseñas', 'Puntuación global de reseñas',
                    'Anfitrión destacado', 'Lista de comodidades']
})
print(diccionario)

# ==========================================================================
# 2.2 Diccionario de las variables críticas
# ==========================================================================
_celda = 'tipos'
# Conteo de columnas por tipo de dato
numericas = df.select_dtypes(include=[np.number]).columns.tolist()
fechas = df.select_dtypes(include=['datetime']).columns.tolist()
texto = df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()
print('Columnas numéricas:', len(numericas))
print('Columnas de fecha:', len(fechas), fechas)
print('Columnas de texto/categóricas:', len(texto))

# ==========================================================================
# 2.3 Variable objetivo
# ==========================================================================
_celda = 'precio'
df['price_num'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

print('Precios válidos:', df['price_num'].notna().sum())
print('Registros sin precio:', df['price_num'].isna().sum())
print('Anuncios con precio igual a 0:', (df['price_num'] == 0).sum())
print(df.loc[df['price_num'] == 0, 'room_type'].value_counts())
print(df['price_num'].describe())

# ==========================================================================
# 2.4 Duplicados
# ==========================================================================
_celda = 'duplicados'
print('Filas completamente duplicadas:', df.duplicated().sum())
print('Identificadores (id) repetidos:', df['id'].duplicated().sum())

# ==========================================================================
# 2.5 Medidas de tendencia central y dispersión
# ==========================================================================
_celda = 'describe'
vars_num = ['price_num', 'accommodates', 'bedrooms', 'beds', 'minimum_nights', 'maximum_nights',
            'number_of_reviews', 'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_location',
            'review_scores_value', 'reviews_per_month', 'host_listings_count', 'calculated_host_listings_count']

resumen = df[vars_num].describe().T
resumen['mediana'] = df[vars_num].median()
resumen['CV (%)'] = resumen['std'] / resumen['mean'] * 100
resumen['asimetría'] = df[vars_num].skew()
print(resumen[['count', 'mean', 'mediana', 'std', 'CV (%)', 'asimetría', 'min', '25%', '75%', 'max']].round(2))

# ==========================================================================
# 2.6 Datos faltantes
# ==========================================================================
_celda = 'nulos'
original = df.drop(columns='price_num')      # solo las 52 columnas originales
nulos = original.isnull().sum()
tabla_nulos = pd.DataFrame({'Nulos': nulos, '% Nulos': nulos / len(df) * 100})
tabla_nulos = tabla_nulos[tabla_nulos['Nulos'] > 0].sort_values('Nulos', ascending=False)
print('Columnas con al menos un nulo:', len(tabla_nulos), 'de', original.shape[1])
print(tabla_nulos.head(20))

# ==========================================================================
# 2.6 Datos faltantes
# ==========================================================================
_celda = 'nulos_graf'
top = tabla_nulos.head(15).sort_values('% Nulos')
plt.figure(figsize=(10, 6))
plt.barh(top.index, top['% Nulos'], color=CORAL)
for i, v in enumerate(top['% Nulos']):
    plt.text(v + 0.5, i, f'{v:.1f} %', va='center', fontsize=9)
plt.title('Las 15 columnas con más datos faltantes (%)')
plt.xlabel('% de registros sin dato')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

# ==========================================================================
# 2.7 Distribución de la variable objetivo (precio por noche)
# ==========================================================================
_celda = 'dist_precio'
precio = df['price_num'].dropna()
precio_pos = precio[precio > 0]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Histograma hasta 500 para ver la forma principal
ax[0].hist(precio[precio <= 500], bins=50, color=CORAL, edgecolor='white')
ax[0].axvline(precio.median(), color=AZUL, linestyle='--', label=f'Mediana: {precio.median():,.0f}')
ax[0].axvline(precio.mean(), color=VERDE, linestyle='--', label=f'Media: {precio.mean():,.0f}')
ax[0].set_title('Precio por noche (anuncios hasta 500)')
ax[0].set_xlabel('Precio por noche')
ax[0].set_ylabel('Número de anuncios')
ax[0].legend()

# Histograma del logaritmo del precio
ax[1].hist(np.log(precio_pos), bins=50, color=VERDE, edgecolor='white')
ax[1].set_title('Logaritmo del precio (todos los anuncios con precio > 0)')
ax[1].set_xlabel('ln(precio)')
ax[1].set_ylabel('Número de anuncios')

plt.tight_layout()
plt.show()

print('Asimetría del precio:', round(precio.skew(), 2))
print('Asimetría del logaritmo del precio:', round(np.log(precio_pos).skew(), 2))

# ==========================================================================
# 2.8 Análisis univariado de las variables categóricas
# ==========================================================================
_celda = 'categoricas_graf'
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Tipo de alojamiento
orden = df['room_type'].value_counts()
ax[0].bar(orden.index, orden.values, color=AZUL)
for i, v in enumerate(orden.values):
    ax[0].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)
ax[0].set_title('Anuncios por tipo de alojamiento')
ax[0].tick_params(axis='x', rotation=20)

# 10 barrios con más anuncios
barrios = df['neighbourhood_cleansed'].value_counts().head(10).sort_values()
ax[1].barh(barrios.index, barrios.values, color=CORAL)
ax[1].set_title('10 barrios con más anuncios')

# 10 tipos de propiedad más frecuentes
props = df['property_type'].value_counts().head(10).sort_values()
ax[2].barh(props.index, props.values, color=VERDE)
ax[2].set_title('10 tipos de propiedad más frecuentes')

plt.tight_layout()
plt.show()

# ==========================================================================
# 2.9 Análisis bivariado: relación de cada variable con el precio
# ==========================================================================
_celda = 'precio_tipo'
datos = df[df['price_num'] > 0]
orden_tipos = ['Hotel room', 'Entire home/apt', 'Private room', 'Shared room']

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(data=datos, x='room_type', y='price_num', order=orden_tipos, color=CORAL, ax=ax[0], fliersize=1)
ax[0].set_yscale('log')
ax[0].set_title('Precio por tipo de alojamiento (escala logarítmica)')
ax[0].set_xlabel('Tipo de alojamiento')
ax[0].set_ylabel('Precio por noche')
ax[0].tick_params(axis='x', rotation=15)

sns.boxplot(data=datos, x='host_is_superhost', y='price_num', order=['t', 'f'], color=VERDE, ax=ax[1], fliersize=1)
ax[1].set_yscale('log')
ax[1].set_title('Precio según si el anfitrión es superhost (t = sí)')
ax[1].set_xlabel('Superhost')
ax[1].set_ylabel('Precio por noche')

plt.tight_layout()
plt.show()

print(df.groupby('room_type')['price_num'].agg(['count', 'median', 'mean']).round(1))

# ==========================================================================
# 2.9 Análisis bivariado: relación de cada variable con el precio
# ==========================================================================
_celda = 'precio_barrio'
# Precio mediano por barrio (solo barrios con al menos 100 anuncios)
por_barrio = df.groupby('neighbourhood_cleansed')['price_num'].agg(['median', 'count'])
por_barrio = por_barrio[por_barrio['count'] >= 100].sort_values('median')

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
por_barrio['median'].head(10).plot(kind='barh', color=VERDE, ax=ax[0])
ax[0].set_title('10 barrios con menor precio mediano')
ax[0].set_xlabel('Precio mediano por noche')
por_barrio['median'].tail(10).plot(kind='barh', color=CORAL, ax=ax[1])
ax[1].set_title('10 barrios con mayor precio mediano')
ax[1].set_xlabel('Precio mediano por noche')
plt.tight_layout()
plt.show()

print(por_barrio.sort_values('median', ascending=False).head(5))

# ==========================================================================
# 2.9 Análisis bivariado: relación de cada variable con el precio
# ==========================================================================
_celda = 'precio_capacidad'
datos = df[(df['price_num'] > 0) & (df['accommodates'] <= 10)]

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(data=datos, x='accommodates', y='price_num', color=AZUL, ax=ax[0], fliersize=1)
ax[0].set_yscale('log')
ax[0].set_title('Precio según capacidad (huéspedes)')
ax[0].set_xlabel('Capacidad (huéspedes)')
ax[0].set_ylabel('Precio por noche')

dh = df[(df['price_num'] > 0) & (df['bedrooms'] <= 6)]
sns.boxplot(data=dh, x='bedrooms', y='price_num', color=VERDE, ax=ax[1], fliersize=1)
ax[1].set_yscale('log')
ax[1].set_title('Precio según número de habitaciones')
ax[1].set_xlabel('Habitaciones')
ax[1].set_ylabel('Precio por noche')

plt.tight_layout()
plt.show()

# ==========================================================================
# 2.10 Matriz de correlación
# ==========================================================================
_celda = 'correlacion'
corr = df[vars_num].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', center=0, linewidths=.5, annot_kws={'size': 8})
plt.title('Matriz de correlación (Pearson)')
plt.tight_layout()
plt.show()

# ==========================================================================
# 2.10 Matriz de correlación
# ==========================================================================
_celda = 'correlacion_precio'
# Correlación de cada variable con el precio. Spearman es más robusto cuando hay atípicos y asimetría
df['ln_precio'] = np.log(df['price_num'].where(df['price_num'] > 0))
cp = pd.DataFrame({
    'Pearson': df[vars_num].corr()['price_num'],
    'Spearman': df[vars_num].corr(method='spearman')['price_num'],
    'Pearson con ln(precio)': df[vars_num].corrwith(df['ln_precio'])
}).drop('price_num').sort_values('Spearman')

cp.plot(kind='barh', figsize=(10, 8), color=[AZUL, CORAL, VERDE])
plt.title('Correlación de cada variable con el precio')
plt.xlabel('Coeficiente de correlación')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

print(cp.round(2).sort_values('Spearman', ascending=False))

# ==========================================================================
# 2.11 Valores atípicos (outliers)
# ==========================================================================
_celda = 'atipicos'
resumen_atipicos = []
for col in vars_num:
    serie = df[col].dropna()
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_atip = ((serie < lim_inf) | (serie > lim_sup)).sum()
    resumen_atipicos.append({
        'Variable': col, 'Q1': round(q1, 2), 'Q3': round(q3, 2), 'IQR': round(iqr, 2),
        'Límite inferior': round(lim_inf, 2), 'Límite superior': round(lim_sup, 2),
        'Total atípicos': n_atip, '% atípicos': round(n_atip / len(serie) * 100, 2)
    })

df_atipicos = pd.DataFrame(resumen_atipicos).sort_values('Total atípicos', ascending=False)
print(df_atipicos)

# ==========================================================================
# 2.11 Valores atípicos (outliers)
# ==========================================================================
_celda = 'atipicos_graf'
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

q1, q3 = precio.quantile(0.25), precio.quantile(0.75)
lim_sup = q3 + 1.5 * (q3 - q1)

sns.boxplot(y=precio, ax=ax[0], color=CORAL, width=0.4, fliersize=2)
ax[0].axhline(lim_sup, color='red', linestyle='--', label=f'Límite superior (IQR): {lim_sup:,.0f}')
ax[0].set_title('Precio por noche: valores atípicos')
ax[0].set_ylabel('Precio por noche')
ax[0].legend()

sns.boxplot(y=precio_pos, ax=ax[1], color=VERDE, width=0.4, fliersize=2)
ax[1].set_yscale('log')
ax[1].set_title('Precio por noche (escala logarítmica)')
ax[1].set_ylabel('Precio por noche')

plt.suptitle('Detección visual de outliers en el precio', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print('Anuncios con precio mayor a 1.000:', (precio > 1000).sum())
print('Anuncios con precio mayor a 5.000:', (precio > 5000).sum())

# Otras inconsistencias detectadas en variables numéricas
print('\nmaximum_nights mayor a 1.125 noches (valor imposible):', (df['maximum_nights'] > 1125).sum())
print('Valor máximo de maximum_nights:', f"{df['maximum_nights'].max():,.0f}")
print('Anuncios con capacidad (accommodates) igual a 0:', (df['accommodates'] == 0).sum())
print('Anuncios con minimum_nights mayor a 365:', (df['minimum_nights'] > 365).sum())

# ==========================================================================
# 2.12 Distribución geográfica del precio
# ==========================================================================
_celda = 'mapa'
datos_mapa = df[df['price_num'] > 0]
muestra = datos_mapa.sample(min(30000, len(datos_mapa)), random_state=42)

plt.figure(figsize=(10, 7))
sc = plt.scatter(muestra['longitude'], muestra['latitude'], c=np.log(muestra['price_num']),
                 cmap='plasma', s=3, alpha=0.6, vmin=3, vmax=7)
plt.colorbar(sc, label='ln(precio por noche)')
plt.title('Ubicación de los anuncios y nivel de precio (muestra de 30.000)')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.tight_layout()
plt.show()

# ==========================================================================
# 3.1 Exclusión de registros no válidos
# ==========================================================================
_celda = 'f3_registros'
dfp = df.copy()   # copia de trabajo
registro = [('Datos originales', 0, len(dfp))]

def excluir(d, conservar, etiqueta):
    # Conserva las filas que cumplen la condición y deja constancia de cuántas se excluyen
    antes = len(d)
    d = d[conservar(d)]
    registro.append((etiqueta, antes - len(d), len(d)))
    return d

dfp = excluir(dfp, lambda d: d['price_num'].notna(), 'Anuncios sin precio')
dfp = excluir(dfp, lambda d: d['price_num'] > 0, 'Anuncios con precio igual a 0')

print('Anuncios con capacidad igual a 0 que quedan:', (dfp['accommodates'] == 0).sum())

print(pd.DataFrame(registro, columns=['Paso', 'Registros excluidos', 'Registros restantes']))

# ==========================================================================
# 3.2 Corrección de inconsistencias
# ==========================================================================
_celda = 'f3_inconsist'
print('Anuncios con maximum_nights > 1.125:', (dfp['maximum_nights'] > 1125).sum())
print('Anuncios con minimum_nights > 365:', (dfp['minimum_nights'] > 365).sum())

dfp['maximum_nights'] = dfp['maximum_nights'].clip(upper=1125)
dfp['minimum_nights'] = dfp['minimum_nights'].clip(upper=365)

print('\nMáximo de maximum_nights después:', dfp['maximum_nights'].max())
print('Máximo de minimum_nights después:', dfp['minimum_nights'].max())

# ==========================================================================
# 3.3 Tratamiento de los precios extremos
# ==========================================================================
_celda = 'f3_atipicos'
dfp['ln_precio'] = np.log(dfp['price_num'])
antes = dfp.copy()   # copia para comparar antes y después

q1, q3 = dfp['ln_precio'].quantile([0.25, 0.75])
iqr = q3 - q1
lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f'Límites sobre ln(precio): {lim_inf:.2f} a {lim_sup:.2f}')
print(f'Equivalen a un precio por noche entre {np.exp(lim_inf):,.1f} y {np.exp(lim_sup):,.1f}')
print('Anuncios por debajo del límite inferior:', (dfp['ln_precio'] < lim_inf).sum())
print('Anuncios por encima del límite superior:', (dfp['ln_precio'] > lim_sup).sum())

dfp = dfp[dfp['ln_precio'].between(lim_inf, lim_sup)]
registro.append(('Precios atípicos (IQR sobre ln precio)', len(antes) - len(dfp), len(dfp)))

print('\nAsimetría del precio antes:', round(antes['price_num'].skew(), 2), '| después:', round(dfp['price_num'].skew(), 2))
print('Precio máximo antes:', f"{antes['price_num'].max():,.0f}", '| después:', f"{dfp['price_num'].max():,.0f}")
print('Registros restantes:', len(dfp))

# ==========================================================================
# 3.3 Tratamiento de los precios extremos
# ==========================================================================
_celda = 'f3_atip_graf'
fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

ax[0].hist(antes['price_num'].clip(upper=1500), bins=60, color=CORAL, edgecolor='white')
ax[0].set_title('Antes (los precios sobre 1.500 se agrupan en el último valor)')
ax[0].set_xlabel('Precio por noche')
ax[0].set_ylabel('Número de anuncios')

ax[1].hist(dfp['price_num'], bins=60, color=VERDE, edgecolor='white')
ax[1].set_title('Después de excluir los precios atípicos')
ax[1].set_xlabel('Precio por noche')

plt.tight_layout()
plt.show()

# ==========================================================================
# 3.4 Tratamiento de los datos faltantes
# ==========================================================================
_celda = 'f3_nulos_antes'
cols_rev = ['bedrooms', 'beds', 'bathrooms_text', 'review_scores_rating', 'review_scores_location',
            'review_scores_value', 'reviews_per_month', 'host_is_superhost', 'host_identity_verified',
            'instant_bookable', 'host_since', 'host_listings_count', 'calculated_host_listings_count']
nulos_antes = dfp[cols_rev].isnull().sum()
print(nulos_antes[nulos_antes > 0].to_frame('Nulos antes'))

# ==========================================================================
# 3.4 Tratamiento de los datos faltantes
# ==========================================================================
_celda = 'f3_imputar'
# a) Baños: se extrae el número del texto (por ejemplo '1.5 shared baths' -> 1.5)
txt = dfp['bathrooms_text'].str.lower()
dfp['banos'] = txt.str.extract(r'(\d+\.?\d*)')[0].astype(float)
dfp.loc[txt.str.contains('half', na=False) & dfp['banos'].isna(), 'banos'] = 0.5
dfp['bano_compartido'] = txt.str.contains('shared', na=False).astype(int)

# b) Habitaciones, camas y baños: mediana del grupo (tipo de alojamiento y capacidad);
#    si el grupo no tiene ningún dato, se usa la mediana general
for col in ['bedrooms', 'beds', 'banos']:
    mediana_grupo = dfp.groupby(['room_type', 'accommodates'])[col].transform('median')
    dfp[col] = dfp[col].fillna(mediana_grupo).fillna(dfp[col].median())

# b2) Valores extremos de habitaciones, camas y baños (errores de captura): se limitan al percentil 99,9
for col in ['bedrooms', 'beds', 'banos']:
    limite = dfp[col].quantile(0.999)
    print(f'{col}: máximo antes {dfp[col].max():.0f} | límite (percentil 99,9) {limite:.1f}')
    dfp[col] = dfp[col].clip(upper=limite)

# c) Reseñas: el indicador conserva la información de que el anuncio no tiene reseñas
dfp['sin_resenas'] = dfp['review_scores_rating'].isna().astype(int)
for col in ['review_scores_rating', 'review_scores_location', 'review_scores_value']:
    dfp[col] = dfp[col].fillna(dfp[col].median())
dfp['reviews_per_month'] = dfp['reviews_per_month'].fillna(0)

# d) Conteos de anuncios del anfitrión con algún dato faltante: mediana
for col in ['host_listings_count', 'host_total_listings_count', 'calculated_host_listings_count']:
    dfp[col] = dfp[col].fillna(dfp[col].median())

# e) Variables t/f del anfitrión: se rellenan con la moda
for col in ['host_is_superhost', 'host_identity_verified', 'instant_bookable']:
    dfp[col] = dfp[col].fillna(dfp[col].mode()[0])

cols_check = ['bedrooms', 'beds', 'banos', 'host_listings_count', 'host_total_listings_count', 'calculated_host_listings_count', 'review_scores_rating', 'review_scores_location',
              'review_scores_value', 'reviews_per_month', 'host_is_superhost',
              'host_identity_verified', 'instant_bookable']
print('Nulos restantes en las variables tratadas:', int(dfp[cols_check].isnull().sum().sum()))
print('Anuncios sin reseñas:', dfp['sin_resenas'].sum(), f"({dfp['sin_resenas'].mean() * 100:.1f} %)")
print(dfp[['bedrooms', 'beds', 'banos']].describe().T.round(2))

# ==========================================================================
# 3.5 Creación de variables (ingeniería de características)
# ==========================================================================
_celda = 'f3_features'
import json

# a) Antigüedad del anfitrión (años) respecto a la fecha de captura de los datos
FECHA_REF = df['last_review'].max()
dfp['antiguedad_anfitrion'] = (FECHA_REF - dfp['host_since']).dt.days / 365.25
dfp['antiguedad_anfitrion'] = dfp['antiguedad_anfitrion'].fillna(dfp['antiguedad_anfitrion'].median())

# b) Número de comodidades
dfp['n_amenities'] = dfp['amenities'].fillna('[]').apply(lambda x: len(json.loads(x)))

# c) Distancia al centro de Londres (Charing Cross: 51.5074, -0.1278) con la fórmula de Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371   # radio de la Tierra en km
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

dfp['dist_centro_km'] = haversine(dfp['latitude'], dfp['longitude'], 51.5074, -0.1278)

# d) Variables t/f a 1/0
for col in ['host_is_superhost', 'host_identity_verified', 'instant_bookable']:
    dfp[col] = (dfp[col] == 't').astype(int)

print(dfp[['antiguedad_anfitrion', 'n_amenities', 'dist_centro_km', 'banos']].describe().T.round(2))

# ==========================================================================
# 3.5 Creación de variables (ingeniería de características)
# ==========================================================================
_celda = 'f3_feat_graf'
num_feats = ['accommodates', 'bedrooms', 'beds', 'banos', 'bano_compartido', 'minimum_nights', 'maximum_nights',
             'number_of_reviews', 'number_of_reviews_ltm', 'reviews_per_month', 'review_scores_rating',
             'review_scores_location', 'review_scores_value', 'sin_resenas', 'host_listings_count',
             'host_total_listings_count', 'calculated_host_listings_count', 'host_is_superhost',
             'host_identity_verified', 'instant_bookable', 'antiguedad_anfitrion', 'n_amenities',
             'dist_centro_km', 'latitude', 'longitude']
nuevas = ['banos', 'bano_compartido', 'sin_resenas', 'antiguedad_anfitrion', 'n_amenities', 'dist_centro_km']

corr_ln = dfp[num_feats + ['ln_precio']].corr(method='spearman')['ln_precio'].drop('ln_precio').sort_values()

fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1.1, 1]})
colores = [CORAL if v in nuevas else AZUL for v in corr_ln.index]
ax[0].barh(corr_ln.index, corr_ln.values, color=colores)
ax[0].axvline(0, color='black', linewidth=0.8)
ax[0].set_title('Correlación de Spearman con ln(precio)\n(en coral, las variables creadas)')
ax[0].set_xlabel('Coeficiente de correlación')

hb = ax[1].hexbin(dfp['dist_centro_km'], dfp['ln_precio'], gridsize=45, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax[1], label='Número de anuncios')
ax[1].set_title('Distancia al centro y precio')
ax[1].set_xlabel('Distancia al centro de Londres (km)')
ax[1].set_ylabel('ln(precio)')

plt.tight_layout()
plt.show()

print(corr_ln.sort_values(ascending=False).head(8).round(2))

# ==========================================================================
# 3.6 Selección de variables y multicolinealidad
# ==========================================================================
_celda = 'f3_multicol'
corr_abs = dfp[num_feats].corr().abs()
pares = []
for i, a in enumerate(num_feats):
    for b in num_feats[i + 1:]:
        if corr_abs.loc[a, b] > 0.80:
            pares.append((a, b, round(corr_abs.loc[a, b], 2)))

print(pd.DataFrame(pares, columns=['Variable A', 'Variable B', '|Correlación|']).sort_values('|Correlación|', ascending=False))

# ==========================================================================
# 3.6 Selección de variables y multicolinealidad
# ==========================================================================
_celda = 'f3_descartar'
descartar = ['beds', 'host_total_listings_count']
num_cols = [c for c in num_feats if c not in descartar]
print('Variables numéricas conservadas:', len(num_cols))
print('Variables numéricas descartadas:', descartar)

# ==========================================================================
# 3.7 Codificación de las variables categóricas
# ==========================================================================
_celda = 'f3_codif'
# Agrupar tipos de propiedad poco frecuentes
frec = dfp['property_type'].value_counts(normalize=True)
principales = frec[frec >= 0.01].index
dfp['property_type_agr'] = dfp['property_type'].where(dfp['property_type'].isin(principales), 'Otros')
print('Tipos de propiedad:', dfp['property_type'].nunique(), '->', dfp['property_type_agr'].nunique())

cat_cols = ['room_type', 'property_type_agr', 'neighbourhood_cleansed']
dummies = pd.get_dummies(dfp[cat_cols], drop_first=True, dtype=int)
X = pd.concat([dfp[num_cols], dummies], axis=1)
y = dfp['ln_precio']

print('\nVariables numéricas:', len(num_cols), '| Variables indicadoras:', dummies.shape[1])
print('Variables explicativas en total:', X.shape[1], '| Registros:', X.shape[0])
print('Nulos en X:', X.isnull().sum().sum(), '| Nulos en y:', y.isnull().sum())

# ==========================================================================
# 3.8 Partición de los datos y escalado
# ==========================================================================
_celda = 'f3_split'
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

continuas = [c for c in num_cols if X[c].nunique() > 2]    # excluye las variables binarias
escalador = StandardScaler().fit(X_train[continuas])        # se ajusta solo con entrenamiento

X_train_esc, X_test_esc = X_train.copy(), X_test.copy()
X_train_esc[continuas] = escalador.transform(X_train[continuas])
X_test_esc[continuas] = escalador.transform(X_test[continuas])

print('Entrenamiento:', X_train.shape, '| Prueba:', X_test.shape)
print('Variables escaladas:', len(continuas))
print('\nComprobación en entrenamiento (media y desviación de las variables escaladas):')
print(X_train_esc[continuas].agg(['mean', 'std']).T.round(2).head(8))

# ==========================================================================
# 3.9 Conjunto final y resumen de la preparación
# ==========================================================================
_celda = 'f3_resumen'
resumen_f3 = pd.DataFrame({
    'Indicador': ['Registros', 'Columnas', 'Valores nulos', 'Asimetría del precio', 'Precio máximo', 'Precio mediano'],
    'Antes (datos originales)': [len(df), original.shape[1], int(original.isnull().sum().sum()),
                                 round(df['price_num'].skew(), 2), f"{df['price_num'].max():,.0f}", f"{df['price_num'].median():,.0f}"],
    'Después (conjunto final)': [len(X), X.shape[1], int(X.isnull().sum().sum()),
                                 round(dfp['price_num'].skew(), 2), f"{dfp['price_num'].max():,.0f}", f"{dfp['price_num'].median():,.0f}"]
})
print(resumen_f3)

# ==========================================================================
# 3.9 Conjunto final y resumen de la preparación
# ==========================================================================
_celda = 'f3_guardar'
# Se guarda el conjunto preparado para usarlo en la Fase 4 (modelado)
final = X.copy()
final['ln_precio'] = y
final['price_num'] = dfp['price_num']
final.to_csv(os.path.join(args.salida, 'airbnb_londres_preparado.csv'), index=False)
print('Archivo guardado en', args.salida, final.shape)

# Descarga opcional al computador (solo funciona en Google Colab)
try:
    from google.colab import files
    files.download(os.path.join(args.salida, 'airbnb_londres_preparado.csv'))
except ImportError:
    pass

# ==========================================================================
# 4.1 Librerías y función de evaluación
# ==========================================================================
_celda = 'f4_funcion'
import time
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

GRIS = '#8D99AE'
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')   # tres decimales para ver las diferencias entre modelos

def evaluar(nombre, y_real_ln, y_pred_ln):
    # Métricas sobre ln(precio) y sobre el precio original (se revierte con la función exponencial)
    p_real, p_pred = np.exp(y_real_ln), np.exp(y_pred_ln)
    return {'Modelo': nombre,
            'R² (ln)': r2_score(y_real_ln, y_pred_ln),
            'MAE (ln)': mean_absolute_error(y_real_ln, y_pred_ln),
            'RMSE (ln)': np.sqrt(mean_squared_error(y_real_ln, y_pred_ln)),
            'R² (precio)': r2_score(p_real, p_pred),
            'MAE (precio)': mean_absolute_error(p_real, p_pred),
            'RMSE (precio)': np.sqrt(mean_squared_error(p_real, p_pred))}
print('Función de evaluación lista')

# ==========================================================================
# 4.2 Conjunto de validación
# ==========================================================================
_celda = 'f4_particion'
idx_tr, idx_val = train_test_split(X_train.index, test_size=0.20, random_state=42)

Xt, Xv = X_train.loc[idx_tr], X_train.loc[idx_val]                 # variables sin escalar (modelos de árboles)
Xt_esc, Xv_esc = X_train_esc.loc[idx_tr], X_train_esc.loc[idx_val]  # variables escaladas (regresión lineal)
yt, yv = y_train.loc[idx_tr], y_train.loc[idx_val]

print('Entrenamiento:', Xt.shape[0], '| Validación:', Xv.shape[0], '| Prueba (sin tocar):', X_test.shape[0])

# ==========================================================================
# 4.3 Modelos con los parámetros por defecto («antes»)
# ==========================================================================
_celda = 'f4_defecto'
modelos_defecto = {
    'Modelo base (mediana)': (DummyRegressor(strategy='median'), 'plano'),
    'Regresión lineal (Ridge)': (Ridge(alpha=1.0), 'esc'),
    'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), 'plano'),
    'Gradient Boosting': (HistGradientBoostingRegressor(random_state=42), 'plano')
}

res_defecto = []
for nombre, (modelo, tipo) in modelos_defecto.items():
    t0 = time.time()
    A, V = (Xt_esc, Xv_esc) if tipo == 'esc' else (Xt, Xv)
    modelo.fit(A, yt)
    fila = evaluar(nombre, yv, modelo.predict(V))
    fila['Tiempo (s)'] = round(time.time() - t0, 1)
    res_defecto.append(fila)

tabla_defecto = pd.DataFrame(res_defecto).set_index('Modelo')
print(tabla_defecto.round(3))

# ==========================================================================
# 4.3 Modelos con los parámetros por defecto («antes»)
# ==========================================================================
_celda = 'f4_graf_defecto'
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
colores = [GRIS, AZUL, CORAL, VERDE]

for eje, col, titulo in [(ax[0], 'R² (ln)', 'R² en validación (más alto es mejor)'),
                         (ax[1], 'RMSE (ln)', 'RMSE en validación (más bajo es mejor)')]:
    barras = eje.bar(tabla_defecto.index, tabla_defecto[col], color=colores)
    for b, v in zip(barras, tabla_defecto[col]):
        eje.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    eje.set_title(titulo)
    eje.tick_params(axis='x', rotation=15)

plt.suptitle('Modelos con parámetros por defecto', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==========================================================================
# 4.4 Ajuste de parámetros (hiperparámetros)
# ==========================================================================
_celda = 'f4_ajuste_ridge'
t0 = time.time()
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000]).fit(Xt_esc, yt)
print('Regresión lineal (Ridge): mejor alpha =', ridge_cv.alpha_, '| tiempo:', round(time.time() - t0, 1), 's')

# ==========================================================================
# 4.4 Ajuste de parámetros (hiperparámetros)
# ==========================================================================
_celda = 'f4_ajuste_gb'
t0 = time.time()
espacio_gb = {'learning_rate': [0.03, 0.05, 0.1, 0.2],
              'max_iter': [200, 400, 600],
              'max_leaf_nodes': [15, 31, 63],
              'min_samples_leaf': [10, 20, 50],
              'l2_regularization': [0, 0.1, 1.0]}
busqueda_gb = RandomizedSearchCV(HistGradientBoostingRegressor(random_state=42), espacio_gb, n_iter=12, cv=3,
                                 scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
busqueda_gb.fit(Xt, yt)
mejores_gb = busqueda_gb.best_params_
print('Gradient Boosting - mejores parámetros:', mejores_gb)
print('RMSE medio en validación cruzada:', round(-busqueda_gb.best_score_, 4), '| tiempo:', round(time.time() - t0), 's')

# ==========================================================================
# 4.4 Ajuste de parámetros (hiperparámetros)
# ==========================================================================
_celda = 'f4_ajuste_rf'
t0 = time.time()
muestra = Xt.sample(min(15000, len(Xt)), random_state=42)
espacio_rf = {'max_depth': [15, 25, None],
              'min_samples_leaf': [1, 2, 5],
              'max_features': [0.3, 0.5, 1.0]}
busqueda_rf = RandomizedSearchCV(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), espacio_rf,
                                 n_iter=6, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
busqueda_rf.fit(muestra, yt.loc[muestra.index])
mejores_rf = busqueda_rf.best_params_
print('Random Forest - mejores parámetros:', mejores_rf)
print('RMSE medio en validación cruzada:', round(-busqueda_rf.best_score_, 4), '| tiempo:', round(time.time() - t0), 's')

# ==========================================================================
# 4.5 Comparación antes y después de ajustar los parámetros
# ==========================================================================
_celda = 'f4_parametros'
filas = [('Regresión lineal (Ridge)', 'alpha', '1.0', str(ridge_cv.alpha_))]
for nombre, clase, mejores in [('Random Forest', RandomForestRegressor, mejores_rf),
                               ('Gradient Boosting', HistGradientBoostingRegressor, mejores_gb)]:
    por_defecto = clase().get_params()
    for parametro, valor in mejores.items():
        filas.append((nombre, parametro, str(por_defecto[parametro]), str(valor)))

print(pd.DataFrame(filas, columns=['Modelo', 'Parámetro', 'Antes (por defecto)', 'Después (ajustado)']))

# ==========================================================================
# 4.5 Comparación antes y después de ajustar los parámetros
# ==========================================================================
_celda = 'f4_comparacion'
modelos_aj = {
    'Regresión lineal (Ridge)': (Ridge(alpha=ridge_cv.alpha_), 'esc'),
    'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, **mejores_rf), 'plano'),
    'Gradient Boosting': (HistGradientBoostingRegressor(random_state=42, **mejores_gb), 'plano')
}

res_aj = []
for nombre, (modelo, tipo) in modelos_aj.items():
    A, V = (Xt_esc, Xv_esc) if tipo == 'esc' else (Xt, Xv)
    modelo.fit(A, yt)
    res_aj.append(evaluar(nombre, yv, modelo.predict(V)))
tabla_aj = pd.DataFrame(res_aj).set_index('Modelo')

antes = tabla_defecto.drop('Modelo base (mediana)')
comparacion = pd.DataFrame({
    'R² antes': antes['R² (ln)'], 'R² después': tabla_aj['R² (ln)'],
    'RMSE antes': antes['RMSE (ln)'], 'RMSE después': tabla_aj['RMSE (ln)'],
    'MAE precio antes': antes['MAE (precio)'], 'MAE precio después': tabla_aj['MAE (precio)']})
comparacion['Mejora del RMSE (%)'] = (comparacion['RMSE antes'] - comparacion['RMSE después']) / comparacion['RMSE antes'] * 100
print(comparacion.round(3))

# ==========================================================================
# 4.5 Comparación antes y después de ajustar los parámetros
# ==========================================================================
_celda = 'f4_comp_graf'
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(comparacion))
ancho = 0.38

for eje, (c_antes, c_desp), titulo in [(ax[0], ('R² antes', 'R² después'), 'R² en validación'),
                                       (ax[1], ('RMSE antes', 'RMSE después'), 'RMSE en validación')]:
    b1 = eje.bar(x - ancho / 2, comparacion[c_antes], ancho, label='Antes (por defecto)', color=GRIS)
    b2 = eje.bar(x + ancho / 2, comparacion[c_desp], ancho, label='Después (ajustado)', color=VERDE)
    for b in list(b1) + list(b2):
        eje.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    eje.set_xticks(x)
    eje.set_xticklabels(comparacion.index, rotation=10)
    eje.set_title(titulo)
    eje.legend()

plt.suptitle('Efecto del ajuste de parámetros', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==========================================================================
# 4.6 Selección del modelo
# ==========================================================================
_celda = 'f4_seleccion'
ranking = tabla_aj[['R² (ln)', 'RMSE (ln)', 'MAE (precio)']].sort_values('RMSE (ln)')
print(ranking.round(4))

# Si la diferencia de RMSE con el mejor es menor que 0,002 se considera un empate técnico y se prefiere el modelo más rápido
mejor_rmse = ranking['RMSE (ln)'].min()
empatados = ranking[ranking['RMSE (ln)'] <= mejor_rmse + 0.002].index
mejor_nombre = tabla_defecto.loc[empatados, 'Tiempo (s)'].idxmin()
print('\nModelos dentro de 0,002 del mejor RMSE:', list(empatados))
print('Modelo seleccionado en validación:', mejor_nombre)

# ==========================================================================
# 5.1 Resultados en el conjunto de prueba
# ==========================================================================
_celda = 'f5_final'
modelos_test = {
    'Modelo base (mediana)': (DummyRegressor(strategy='median'), 'plano'),
    'Regresión lineal (Ridge)': (Ridge(alpha=ridge_cv.alpha_), 'esc'),
    'Random Forest (por defecto)': (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), 'plano'),
    'Random Forest (ajustado)': (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, **mejores_rf), 'plano'),
    'Gradient Boosting (por defecto)': (HistGradientBoostingRegressor(random_state=42), 'plano'),
    'Gradient Boosting (ajustado)': (HistGradientBoostingRegressor(random_state=42, **mejores_gb), 'plano')
}

pred_test, res_test, entrenados = {}, [], {}
for nombre, (modelo, tipo) in modelos_test.items():
    A, T = (X_train_esc, X_test_esc) if tipo == 'esc' else (X_train, X_test)
    modelo.fit(A, y_train)
    pred_test[nombre] = modelo.predict(T)
    entrenados[nombre] = (modelo, tipo)
    res_test.append(evaluar(nombre, y_test, pred_test[nombre]))

tabla_test = pd.DataFrame(res_test).set_index('Modelo')
print(tabla_test.round(3))

# ==========================================================================
# 5.1 Resultados en el conjunto de prueba
# ==========================================================================
_celda = 'f5_graf_modelos'
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
colores = [GRIS, AZUL, '#F4A261', CORAL, '#8FD0C7', VERDE]

for eje, col, titulo in [(ax[0], 'R² (ln)', 'R² en prueba (más alto es mejor)'),
                         (ax[1], 'MAE (precio)', 'Error absoluto medio en precio (más bajo es mejor)')]:
    barras = eje.barh(tabla_test.index[::-1], tabla_test[col][::-1], color=colores[::-1])
    for b, v in zip(barras, tabla_test[col][::-1]):
        eje.text(v, b.get_y() + b.get_height() / 2, f' {v:.2f}', va='center', fontsize=10)
    eje.set_title(titulo)

plt.suptitle('Comparación de modelos en el conjunto de prueba', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ==========================================================================
# 5.2 Comparación antes y después del ajuste en el conjunto de prueba
# ==========================================================================
_celda = 'f5_antes_despues'
pares = [('Random Forest', 'Random Forest (por defecto)', 'Random Forest (ajustado)'),
         ('Gradient Boosting', 'Gradient Boosting (por defecto)', 'Gradient Boosting (ajustado)')]
filas = []
for nombre, a, d in pares:
    filas.append({'Modelo': nombre,
                  'R² antes': tabla_test.loc[a, 'R² (ln)'], 'R² después': tabla_test.loc[d, 'R² (ln)'],
                  'RMSE antes': tabla_test.loc[a, 'RMSE (ln)'], 'RMSE después': tabla_test.loc[d, 'RMSE (ln)'],
                  'MAE precio antes': tabla_test.loc[a, 'MAE (precio)'], 'MAE precio después': tabla_test.loc[d, 'MAE (precio)']})
antes_despues = pd.DataFrame(filas).set_index('Modelo')
antes_despues['Mejora del RMSE (%)'] = (antes_despues['RMSE antes'] - antes_despues['RMSE después']) / antes_despues['RMSE antes'] * 100
antes_despues['Mejora del MAE en precio (%)'] = (antes_despues['MAE precio antes'] - antes_despues['MAE precio después']) / antes_despues['MAE precio antes'] * 100
print(antes_despues.round(3))

# ==========================================================================
# 5.3 Cumplimiento de los criterios de éxito
# ==========================================================================
_celda = 'f5_criterios'
clave = mejor_nombre if mejor_nombre == 'Regresión lineal (Ridge)' else mejor_nombre + ' (ajustado)'
base = tabla_test.loc['Modelo base (mediana)']
final = tabla_test.loc[clave]

mejora_mae_ln = (base['MAE (ln)'] - final['MAE (ln)']) / base['MAE (ln)'] * 100
mejora_mae_precio = (base['MAE (precio)'] - final['MAE (precio)']) / base['MAE (precio)'] * 100

criterios = pd.DataFrame({
    'Criterio': ['R² en prueba (ln precio)', 'Reducción del MAE frente al modelo base (ln precio)', 'Reducción del MAE frente al modelo base (precio)'],
    'Meta': ['≥ 0,60', '≥ 25 %', '≥ 25 %'],
    'Resultado': [f"{final['R² (ln)']:.3f}", f'{mejora_mae_ln:.1f} %', f'{mejora_mae_precio:.1f} %'],
    'Cumple': ['Sí' if final['R² (ln)'] >= 0.60 else 'No', 'Sí' if mejora_mae_ln >= 25 else 'No', 'Sí' if mejora_mae_precio >= 25 else 'No']})
print('Modelo evaluado:', clave)
print(f"MAE del modelo base en precio: {base['MAE (precio)']:.1f} | MAE del modelo final: {final['MAE (precio)']:.1f}")
print(criterios)

# ==========================================================================
# 5.4 Predicciones frente a valores reales
# ==========================================================================
_celda = 'f5_graf_pred'
pred_final = pred_test[clave]
residuos = y_test - pred_final

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

hb = ax[0].hexbin(y_test, pred_final, gridsize=45, cmap='YlOrRd', mincnt=1)
lim = [y_test.min(), y_test.max()]
ax[0].plot(lim, lim, color=AZUL, linestyle='--', label='Predicción perfecta')
plt.colorbar(hb, ax=ax[0], label='Número de anuncios')
ax[0].set_title(f'Real frente a predicho: {clave}')
ax[0].set_xlabel('ln(precio) real')
ax[0].set_ylabel('ln(precio) predicho')
ax[0].legend()

ax[1].hist(residuos, bins=60, color=VERDE, edgecolor='white')
ax[1].axvline(0, color=AZUL, linestyle='--')
ax[1].set_title('Distribución de los residuos (real − predicho)')
ax[1].set_xlabel('Error en ln(precio)')
ax[1].set_ylabel('Número de anuncios')

plt.tight_layout()
plt.show()

print('Media de los residuos:', round(residuos.mean(), 4))
print('Desviación de los residuos:', round(residuos.std(), 4))
print('Anuncios con error menor al 30 % del precio:', f'{(np.abs(np.exp(pred_final) / np.exp(y_test) - 1) < 0.30).mean() * 100:.1f} %')

# ==========================================================================
# 5.5 Dónde se equivoca más el modelo
# ==========================================================================
_celda = 'f5_error_grupos'
ev = dfp.loc[X_test.index, ['room_type', 'price_num']].copy()
ev['pred'] = np.exp(pred_final)
ev['error_abs'] = (ev['pred'] - ev['price_num']).abs()
ev['error_pct'] = ev['error_abs'] / ev['price_num'] * 100
ev['rango'] = pd.cut(ev['price_num'], [0, 50, 100, 150, 250, 500, 1100],
                     labels=['hasta 50', '50-100', '100-150', '150-250', '250-500', '500-1.100'])

por_tipo = ev.groupby('room_type').agg(anuncios=('error_abs', 'size'), MAE=('error_abs', 'mean'),
                                       mediana_error_pct=('error_pct', 'median')).round(1)
por_rango = ev.groupby('rango', observed=True).agg(anuncios=('error_abs', 'size'), MAE=('error_abs', 'mean'),
                                                   mediana_error_pct=('error_pct', 'median')).round(1)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(por_tipo.index, por_tipo['MAE'], color=CORAL)
ax[0].set_title('Error absoluto medio por tipo de alojamiento')
ax[0].set_ylabel('MAE (precio)')
ax[0].tick_params(axis='x', rotation=15)
ax[1].bar(por_rango.index.astype(str), por_rango['MAE'], color=VERDE)
ax[1].set_title('Error absoluto medio por rango de precio real')
ax[1].set_ylabel('MAE (precio)')
plt.tight_layout()
plt.show()

print(por_tipo)
print()
print(por_rango)

# ==========================================================================
# 5.6 Variables más importantes
# ==========================================================================
_celda = 'f5_importancia'
from sklearn.inspection import permutation_importance

modelo_final, tipo_final = entrenados[clave]
T = X_test_esc if tipo_final == 'esc' else X_test
muestra_t = T.sample(4000, random_state=42)

imp = permutation_importance(modelo_final, muestra_t, y_test.loc[muestra_t.index], n_repeats=5,
                             random_state=42, scoring='neg_root_mean_squared_error')
importancia = pd.Series(imp.importances_mean, index=T.columns).sort_values(ascending=False)

top = importancia.head(15).sort_values()
plt.figure(figsize=(10, 7))
plt.barh(top.index, top.values, color=CORAL)
plt.title('15 variables más importantes (aumento del RMSE al mezclarlas)')
plt.xlabel('Aumento del RMSE en ln(precio)')
plt.tight_layout()
plt.show()

print(importancia.head(10).round(3))

# ==========================================================================
# 5.7 Resumen de la evaluación
# ==========================================================================
_celda = 'f5_resumen'
resumen_final = tabla_test[['R² (ln)', 'MAE (ln)', 'RMSE (ln)', 'MAE (precio)', 'RMSE (precio)']].round(3)
resumen_final['Mejora del MAE vs base (%)'] = ((base['MAE (precio)'] - tabla_test['MAE (precio)']) / base['MAE (precio)'] * 100).round(1)
print(resumen_final)

# ============================================================================
# Guardar las tablas de resultados
# ============================================================================
tablas = {
    'resumen_preparacion': 'resumen_f3', 'modelos_validacion_por_defecto': 'tabla_defecto',
    'modelos_validacion_ajustados': 'tabla_aj', 'comparacion_antes_despues_validacion': 'comparacion',
    'resultados_prueba': 'tabla_test', 'comparacion_antes_despues_prueba': 'antes_despues',
    'criterios_de_exito': 'criterios', 'error_por_tipo': 'por_tipo', 'error_por_rango_de_precio': 'por_rango',
    'importancia_variables': 'importancia', 'parametros_antes_despues': None}
for archivo, variable in tablas.items():
    if variable and variable in globals():
        objeto = globals()[variable]
        objeto.to_csv(os.path.join(args.salida, 'tablas', archivo + '.csv'), encoding='utf-8-sig')
print('\nListo. Figuras y tablas guardadas en:', args.salida)

# ============================================================================
# Exportar el modelo final y sus metadatos para el estimador interactivo (app/)
# ============================================================================
import json
import joblib
import sklearn

modelo_app = entrenados['Gradient Boosting (ajustado)'][0]          # entrenado con el conjunto de entrenamiento
pred_app = pred_test['Gradient Boosting (ajustado)']
cociente = np.exp(y_test.values - pred_app)                          # precio real / precio predicho
met = tabla_test.loc['Gradient Boosting (ajustado)']
met_base = tabla_test.loc['Modelo base (mediana)']

meta_app = {
    'columnas': list(X.columns),
    'defectos': {c: float(X[c].median()) for c in num_cols},
    'tipos_alojamiento': sorted(dfp['room_type'].unique().tolist()),
    'tipos_propiedad': sorted(dfp['property_type_agr'].unique().tolist()),
    'barrios': {b: {'lat': float(g['latitude'].median()), 'lon': float(g['longitude'].median()),
                    'dist_km': float(g['dist_centro_km'].median()), 'precio_mediano': float(g['price_num'].median()),
                    'anuncios': int(len(g))} for b, g in dfp.groupby('neighbourhood_cleansed')},
    'precio_mediano_tipo': {t: float(g['price_num'].median()) for t, g in dfp.groupby('room_type')},
    'rango_precio_entrenamiento': [float(np.exp(lim_inf)), float(np.exp(lim_sup))],
    'cociente_q10': float(np.quantile(cociente, 0.10)),
    'cociente_q90': float(np.quantile(cociente, 0.90)),
    'metricas': {'r2_ln': float(met['R² (ln)']), 'mae_precio': float(met['MAE (precio)']),
                 'rmse_precio': float(met['RMSE (precio)']), 'mae_base': float(met_base['MAE (precio)']),
                 'anuncios_entrenamiento': int(len(X_train)), 'anuncios_prueba': int(len(X_test))},
    'version_sklearn': sklearn.__version__}

os.makedirs(args.modelo_app, exist_ok=True)
joblib.dump(modelo_app, os.path.join(args.modelo_app, 'modelo_gb.joblib'), compress=3)
with open(os.path.join(args.modelo_app, 'metadatos.json'), 'w', encoding='utf-8') as f:
    json.dump(meta_app, f, ensure_ascii=False, indent=1)
print('Modelo y metadatos del estimador guardados en', args.modelo_app)
