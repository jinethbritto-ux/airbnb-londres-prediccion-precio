"""Estimador interactivo del precio por noche de alojamientos de Airbnb en Londres (septiembre 2022)."""
import pandas as pd
import streamlit as st

from estimador import (CURVAS, EJEMPLOS, ETIQUETAS_ALOJAMIENTO, cargar_modelo, comparables, contribuciones, curva,
                       estimar, precios_por_barrio, propiedad_por_defecto, tipos_propiedad_para, validar_entradas)

st.set_page_config(page_title='Estimador de precios Airbnb Londres', page_icon='🏠', layout='wide',
                   initial_sidebar_state='expanded')


@st.cache_resource
def recursos():
    return cargar_modelo()


try:
    modelo, meta = recursos()
except Exception as error:   # por ejemplo, si la versión de scikit-learn no coincide
    st.error('No se pudo cargar el modelo. Revisa que la versión de scikit-learn sea la indicada en '
             'requirements.txt, o vuelve a generar el modelo con el script del repositorio.')
    st.exception(error)
    st.stop()

m = meta['metricas']
d = meta['defectos']
barrios = meta['barrios']
lista_barrios = sorted(barrios)
estado = st.session_state

# --------------------------------------------------------------------------
# Valores iniciales (se guardan en la sesión para poder cargar ejemplos)
# --------------------------------------------------------------------------
INICIAL = {
    'tipo': 'Entire home/apt', 'barrio': 'Westminster', 'capacidad': 2, 'habitaciones': 1, 'banos': 1.0,
    'compartido': False, 'comodidades': int(d['n_amenities']), 'superhost': False, 'verificado': True,
    'instantanea': False, 'antiguedad': round(d['antiguedad_anfitrion'] * 2) / 2,
    'anuncios': int(d['calculated_host_listings_count']), 'tiene_resenas': True,
    'total_resenas': max(1, int(d['number_of_reviews'])), 'resenas_12m': int(d['number_of_reviews_ltm']),
    'puntuacion': float(round(d['review_scores_rating'], 2)), 'noches_min': int(d['minimum_nights']),
    'noches_max': int(d['maximum_nights']),
}
for clave, valor in INICIAL.items():
    estado.setdefault(clave, valor)
estado.setdefault('propiedad', propiedad_por_defecto(meta, estado['tipo']))


def cargar_ejemplo():
    nombre = estado['ejemplo']
    if nombre not in EJEMPLOS:
        return
    v = EJEMPLOS[nombre]
    for clave in ('tipo', 'propiedad', 'barrio', 'capacidad', 'habitaciones', 'banos', 'compartido', 'comodidades'):
        estado[clave] = v[clave]
    estado[f"dist_{v['barrio']}"] = round(barrios[v['barrio']]['dist_km'], 1)


# --------------------------------------------------------------------------
# Datos de entrada (barra lateral)
# --------------------------------------------------------------------------
with st.sidebar:
    st.header('Tu alojamiento')
    st.selectbox('Cargar un ejemplo', ['Elegir…'] + list(EJEMPLOS), key='ejemplo', on_change=cargar_ejemplo)

    tipo = st.selectbox('Tipo de alojamiento', meta['tipos_alojamiento'],
                        format_func=lambda t: ETIQUETAS_ALOJAMIENTO.get(t, t), key='tipo')
    opciones_propiedad = tipos_propiedad_para(meta, tipo)
    if estado['propiedad'] not in opciones_propiedad:
        estado['propiedad'] = propiedad_por_defecto(meta, tipo)
    propiedad = st.selectbox('Tipo de propiedad', opciones_propiedad, key='propiedad',
                             help='Solo se muestran los tipos que tienen sentido para el alojamiento elegido. '
                                  '"Otros" agrupa los tipos poco frecuentes.')
    barrio = st.selectbox('Barrio', lista_barrios, key='barrio')

    capacidad = st.number_input('Capacidad (huéspedes)', min_value=1, max_value=16, step=1, key='capacidad')
    habitaciones = st.number_input('Habitaciones', min_value=1, max_value=6, step=1, key='habitaciones')
    banos = st.number_input('Baños', min_value=0.5, max_value=5.0, step=0.5, key='banos')
    compartido = st.checkbox('El baño es compartido', key='compartido')
    comodidades = st.slider('Número de comodidades (wifi, cocina, etc.)', 0, 90, key='comodidades')

    estado.setdefault(f'dist_{barrio}', round(barrios[barrio]['dist_km'], 1))
    distancia = st.slider('Distancia al centro de Londres (km)', 0.0, 30.0, step=0.1, key=f'dist_{barrio}',
                          help=f'Mediana en {barrio}: {barrios[barrio]["dist_km"]:.1f} km. Cambia al elegir otro barrio.')

    with st.expander('Anfitrión'):
        superhost = st.checkbox('Es superhost', key='superhost')
        verificado = st.checkbox('Identidad verificada', key='verificado')
        instantanea = st.checkbox('Reserva instantánea', key='instantanea')
        antiguedad = st.slider('Años como anfitrión', 0.0, 14.0, step=0.5, key='antiguedad')
        anuncios = st.number_input('Número de anuncios del anfitrión', min_value=1, max_value=500, step=1, key='anuncios')

    with st.expander('Reseñas'):
        tiene = st.checkbox('El anuncio ya tiene reseñas', key='tiene_resenas')
        if tiene:
            total = st.number_input('Total de reseñas', min_value=1, max_value=2000, step=1, key='total_resenas')
            ultimo_anio = st.number_input('Reseñas en los últimos 12 meses', min_value=0, max_value=500, step=1,
                                          key='resenas_12m')
            puntuacion = st.slider('Puntuación global', 1.0, 5.0, step=0.05, key='puntuacion')

    with st.expander('Reglas de la reserva'):
        noches_min = st.number_input('Noches mínimas', min_value=1, max_value=365, step=1, key='noches_min')
        noches_max = st.number_input('Noches máximas', min_value=1, max_value=1125, step=1, key='noches_max')

entradas = {
    'room_type': tipo, 'property_type': propiedad, 'barrio': barrio,
    'accommodates': capacidad, 'bedrooms': habitaciones, 'banos': banos, 'bano_compartido': int(compartido),
    'n_amenities': comodidades, 'dist_centro_km': distancia,
    'latitude': barrios[barrio]['lat'], 'longitude': barrios[barrio]['lon'],
    'host_is_superhost': int(superhost), 'host_identity_verified': int(verificado), 'instant_bookable': int(instantanea),
    'antiguedad_anfitrion': antiguedad, 'host_listings_count': anuncios, 'calculated_host_listings_count': anuncios,
    'minimum_nights': noches_min, 'maximum_nights': noches_max, 'sin_resenas': int(not tiene),
}
if tiene:
    entradas.update({'number_of_reviews': total, 'number_of_reviews_ltm': ultimo_anio,
                     'reviews_per_month': ultimo_anio / 12, 'review_scores_rating': puntuacion})
else:
    entradas.update({'number_of_reviews': 0, 'number_of_reviews_ltm': 0, 'reviews_per_month': 0})

precio, bajo, alto = estimar(modelo, meta, entradas)

# --------------------------------------------------------------------------
# Resultados
# --------------------------------------------------------------------------
st.title('Estimador del precio por noche de un alojamiento en Londres')
st.write('Describe un alojamiento de Airbnb en la barra lateral y el modelo estima su precio por noche. '
         'Es una **referencia** para quien va a publicar un alojamiento, no un precio exacto.')

for aviso in validar_entradas(entradas):
    st.warning(aviso)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Estimación', 'Comparar barrios', 'Qué influye', 'Cómo cambia el precio',
                                        'Acerca del modelo'])

with tab1:
    comp = comparables(meta, barrio, tipo)
    c1, c2, c3 = st.columns(3)
    c1.metric('Precio estimado por noche', f'{precio:,.0f}',
              delta=(f'{precio - comp["p50"]:+,.0f} frente a la mediana de anuncios parecidos' if comp else None),
              delta_color='off')
    c2.metric('Rango probable (80 % de los casos)', f'{bajo:,.0f} a {alto:,.0f}',
              help='En 8 de cada 10 anuncios de prueba, el precio real estuvo dentro de este rango.')
    c3.metric('Anuncios parecidos', f'{comp["n"]:,}' if comp else '—',
              help='Anuncios reales del mismo barrio y tipo de alojamiento (o solo del barrio o del tipo si hay pocos).')
    st.caption('Moneda de la fuente (Inside Airbnb).')

    if comp:
        st.markdown(f'**Tu estimación frente a anuncios reales {comp["nivel"]}**')
        grafico = pd.DataFrame({'Precio por noche': [comp['p10'], comp['p50'], comp['p90'], precio]},
                               index=['1. Parecidos: 10 % más baratos', '2. Parecidos: mediana',
                                      '3. Parecidos: 10 % más caros', '4. Tu estimación'])
        st.bar_chart(grafico)

    minimo, maximo = meta['rango_precio_entrenamiento']
    if precio > 600:
        st.warning('Es un precio alto: el modelo se equivoca más en los alojamientos caros y tiende a subestimarlos.')
    if precio < minimo or precio > maximo:
        st.warning(f'El modelo se entrenó con precios entre {minimo:,.0f} y {maximo:,.0f} por noche; '
                   'esta estimación está fuera de ese rango.')

    resumen = pd.DataFrame({
        'Característica': ['Tipo de alojamiento', 'Tipo de propiedad', 'Barrio', 'Capacidad', 'Habitaciones', 'Baños',
                           'Baño compartido', 'Comodidades', 'Distancia al centro (km)', 'Precio estimado por noche',
                           'Rango probable (desde)', 'Rango probable (hasta)'],
        'Valor': [ETIQUETAS_ALOJAMIENTO.get(tipo, tipo), propiedad, barrio, capacidad, habitaciones, banos,
                  'Sí' if compartido else 'No', comodidades, distancia, round(precio), round(bajo), round(alto)]})
    st.download_button('Descargar esta estimación (CSV)', resumen.to_csv(index=False).encode('utf-8-sig'),
                       file_name='estimacion_precio.csv', mime='text/csv')

with tab2:
    st.markdown('### El mismo alojamiento en cada barrio')
    st.write('Precio estimado de este alojamiento si estuviera en cada barrio de Londres, con la distancia '
             'al centro típica de cada barrio.')
    tabla = precios_por_barrio(modelo, meta, entradas)
    b1, b2, b3 = st.columns(3)
    b1.metric('Barrio más caro', f'{tabla.index[0]}: {tabla.iloc[0, 0]:,.0f}')
    b2.metric('Barrio más barato', f'{tabla.index[-1]}: {tabla.iloc[-1, 0]:,.0f}')
    b3.metric(f'En {barrio}', f'{tabla.loc[barrio, "Precio estimado"]:,.0f}')
    grafico_barrios = pd.DataFrame({
        'Barrio elegido': tabla['Precio estimado'].where(tabla.index == barrio, 0),
        'Otros barrios': tabla['Precio estimado'].where(tabla.index != barrio, 0)})
    st.bar_chart(grafico_barrios)
    st.dataframe(tabla.round(0), use_container_width=True)

with tab3:
    st.markdown('### Qué empuja el precio hacia arriba o hacia abajo')
    st.write('Efecto de cada característica frente a un alojamiento típico (2 huéspedes, 1 habitación, 1 baño no '
             'compartido, comodidades y distancia típicas del barrio), manteniendo todo lo demás igual.')
    efectos = contribuciones(modelo, meta, entradas)
    if efectos.empty:
        st.info('Tu alojamiento coincide con un alojamiento típico en estas características.')
    else:
        st.bar_chart(efectos.set_index('Característica')['Efecto en el precio'])
        mostrar = efectos.copy()
        mostrar['Efecto en el precio'] = mostrar['Efecto en el precio'].map(lambda v: f'{v:+,.0f}')
        st.dataframe(mostrar, use_container_width=True, hide_index=True)
    st.caption('Son efectos aproximados que describen lo que aprendió el modelo, no relaciones de causa y efecto. '
               'Varias características están relacionadas entre sí (por ejemplo, capacidad, habitaciones y baños).')

with tab4:
    st.markdown('### Cómo cambia el precio al variar una característica')
    etiqueta = st.selectbox('¿Qué quieres variar?', list(CURVAS))
    st.line_chart(curva(modelo, meta, entradas, etiqueta))
    st.caption('El resto de las características se mantienen como las elegiste en la barra lateral.')

with tab5:
    st.markdown('### Cómo funciona')
    st.markdown(
        '1. **Tú describes** el alojamiento (unos 15 datos).\n'
        '2. **La app completa** las 67 variables que necesita el modelo: lo que no pides se rellena con valores '
        'típicos del conjunto de entrenamiento y las categorías se convierten en columnas de 0 y 1.\n'
        '3. **El modelo predice** el logaritmo del precio y la app lo devuelve a precio.\n'
        '4. **El rango** sale de los errores del modelo en anuncios que no vio durante el entrenamiento: '
        f'el precio real estuvo entre {meta["cociente_q10"]:.2f} y {meta["cociente_q90"]:.2f} veces la '
        'predicción en 8 de cada 10 casos.')
    st.markdown('### El modelo')
    st.write(f'Gradient Boosting entrenado con {m["anuncios_entrenamiento"]:,} anuncios de Londres (septiembre de 2022). '
             f'En {m["anuncios_prueba"]:,} anuncios que no vio durante el entrenamiento obtuvo un R² de '
             f'{m["r2_ln"]:.2f} (sobre el logaritmo del precio) y un error absoluto medio de {m["mae_precio"]:.0f} por '
             f'noche, frente a {m["mae_base"]:.0f} de un modelo que siempre predice la mediana.')
    st.markdown('### Limitaciones')
    st.markdown(
        '- El precio de los datos es el **publicado**, de un solo mes, no el realmente pagado.\n'
        '- No se conocen el estado del alojamiento, las fotos ni la temporada, que también influyen en el precio.\n'
        '- Se excluyeron los anuncios con precio superior a 1.065 por noche: no sirve para alojamientos de lujo extremo.\n'
        '- El error crece con el precio y el modelo tiende a subestimar los alojamientos más caros.\n'
        '- Las habitaciones compartidas y de hotel tienen pocos anuncios: sus estimaciones son menos confiables.\n'
        '- Es un proyecto académico (Diplomado en Analítica de Datos, Uniminuto): úsalo como referencia, no como '
        'precio definitivo.')
    st.caption(f'Datos: Inside Airbnb, Londres, septiembre de 2022 · scikit-learn {meta["version_sklearn"]}')

st.caption('Jineth Valentina Díaz Romero · Diplomado en Analítica de Datos · Uniminuto')
