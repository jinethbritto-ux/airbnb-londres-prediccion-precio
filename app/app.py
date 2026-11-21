"""Estimador interactivo del precio por noche de alojamientos de Airbnb en Londres (septiembre 2022)."""
import streamlit as st

from estimador import ETIQUETAS_ALOJAMIENTO, cargar_modelo, estimar

st.set_page_config(page_title='Estimador de precios Airbnb Londres', page_icon='🏠', layout='wide')


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
barrios = meta['barrios']

st.title('Estimador del precio por noche de un alojamiento en Londres')
st.write('Indica las características de un alojamiento de Airbnb y el modelo estima su precio por noche. '
         'Es una **referencia** para quien va a publicar un alojamiento, no un precio exacto.')

izq, der = st.columns([1.15, 1])

with izq:
    st.subheader('Alojamiento')
    c1, c2 = st.columns(2)
    tipo = c1.selectbox('Tipo de alojamiento', meta['tipos_alojamiento'],
                        format_func=lambda t: ETIQUETAS_ALOJAMIENTO.get(t, t))
    propiedad = c2.selectbox('Tipo de propiedad', meta['tipos_propiedad'],
                             help='Categorías de Airbnb; "Otros" agrupa los tipos poco frecuentes.')
    barrio = st.selectbox('Barrio', sorted(barrios), index=sorted(barrios).index('Westminster'))

    c3, c4, c5 = st.columns(3)
    capacidad = c3.number_input('Capacidad (huéspedes)', 1, 16, 2)
    habitaciones = c4.number_input('Habitaciones', 1, 6, 1)
    banos = c5.number_input('Baños', 0.5, 5.0, 1.0, step=0.5)
    compartido = st.checkbox('El baño es compartido')
    comodidades = st.slider('Número de comodidades (wifi, cocina, etc.)', 0, 90, int(meta['defectos']['n_amenities']))
    distancia = st.slider(f'Distancia al centro de Londres (km). Mediana en {barrio}: {barrios[barrio]["dist_km"]:.1f} km',
                          0.0, 30.0, round(barrios[barrio]['dist_km'], 1), step=0.1, key=f'dist_{barrio}')

    with st.expander('Anfitrión'):
        a1, a2 = st.columns(2)
        superhost = a1.checkbox('Es superhost')
        verificado = a1.checkbox('Identidad verificada', value=True)
        instantanea = a2.checkbox('Reserva instantánea')
        antiguedad = a2.slider('Años como anfitrión', 0.0, 14.0, float(round(meta['defectos']['antiguedad_anfitrion'], 1)), step=0.5)
        anuncios = st.number_input('Número de anuncios del anfitrión', 1, 500, int(meta['defectos']['calculated_host_listings_count']))

    with st.expander('Reseñas'):
        tiene = st.checkbox('El anuncio ya tiene reseñas', value=True)
        if tiene:
            r1, r2, r3 = st.columns(3)
            total = r1.number_input('Total de reseñas', 1, 2000, max(1, int(meta['defectos']['number_of_reviews'])))
            ultimo_anio = r2.number_input('Reseñas en los últimos 12 meses', 0, 500, int(meta['defectos']['number_of_reviews_ltm']))
            puntuacion = r3.slider('Puntuación global', 1.0, 5.0, float(round(meta['defectos']['review_scores_rating'], 2)), step=0.05)

    with st.expander('Reglas de la reserva'):
        n1, n2 = st.columns(2)
        noches_min = n1.number_input('Noches mínimas', 1, 365, int(meta['defectos']['minimum_nights']))
        noches_max = n2.number_input('Noches máximas', 1, 1125, int(meta['defectos']['maximum_nights']))

entradas = {
    'room_type': tipo, 'property_type': propiedad, 'barrio': barrio,
    'accommodates': capacidad, 'bedrooms': habitaciones, 'banos': banos, 'bano_compartido': int(compartido),
    'n_amenities': comodidades, 'dist_centro_km': distancia,
    'latitude': barrios[barrio]['lat'], 'longitude': barrios[barrio]['lon'],
    'host_is_superhost': int(superhost), 'host_identity_verified': int(verificado), 'instant_bookable': int(instantanea),
    'antiguedad_anfitrion': antiguedad, 'host_listings_count': anuncios, 'calculated_host_listings_count': anuncios,
    'minimum_nights': noches_min, 'maximum_nights': noches_max,
    'sin_resenas': int(not tiene),
}
if tiene:
    entradas.update({'number_of_reviews': total, 'number_of_reviews_ltm': ultimo_anio,
                     'reviews_per_month': ultimo_anio / 12, 'review_scores_rating': puntuacion})
else:
    entradas.update({'number_of_reviews': 0, 'number_of_reviews_ltm': 0, 'reviews_per_month': 0})

precio, bajo, alto = estimar(modelo, meta, entradas)

with der:
    st.subheader('Precio estimado')
    st.metric('Precio por noche', f'{precio:,.0f}')
    st.write(f'**Rango probable (80 % de los casos): {bajo:,.0f} a {alto:,.0f}**')
    st.caption('Moneda de la fuente (Inside Airbnb).')

    k1, k2 = st.columns(2)
    k1.metric(f'Mediana en {barrio}', f'{barrios[barrio]["precio_mediano"]:,.0f}')
    k2.metric(f'Mediana de {ETIQUETAS_ALOJAMIENTO.get(tipo, tipo).lower()}', f'{meta["precio_mediano_tipo"][tipo]:,.0f}')

    minimo, maximo = meta['rango_precio_entrenamiento']
    if precio > 600:
        st.warning('Es un precio alto: el modelo se equivoca más en los alojamientos caros y tiende a subestimarlos.')
    if precio < minimo or precio > maximo:
        st.warning(f'El modelo se entrenó con precios entre {minimo:,.0f} y {maximo:,.0f} por noche; esta estimación está fuera de ese rango.')

    st.divider()
    st.markdown('**Sobre el modelo**')
    st.write(f'Gradient Boosting entrenado con {m["anuncios_entrenamiento"]:,} anuncios. En {m["anuncios_prueba"]:,} anuncios '
             f'que no vio durante el entrenamiento obtuvo un R² de {m["r2_ln"]:.2f} (sobre el logaritmo del precio) '
             f'y un error absoluto medio de {m["mae_precio"]:.0f} por noche, frente a {m["mae_base"]:.0f} de un modelo que '
             f'siempre predice la mediana.')

with st.expander('Limitaciones'):
    st.markdown(
        '- El precio de los datos es el **publicado**, de un solo mes (septiembre de 2022), no el realmente pagado.\n'
        '- No se conocen el estado del alojamiento, las fotos ni la temporada, que también influyen en el precio.\n'
        '- Se excluyeron los anuncios con precio superior a 1.065 por noche: no uses la herramienta para alojamientos de lujo extremo.\n'
        '- El error crece con el precio y el modelo tiende a subestimar los alojamientos más caros.\n'
        '- Es un proyecto académico (Diplomado en Analítica de Datos, Uniminuto): úsalo como referencia, no como precio definitivo.')

st.caption('Jineth Valentina Díaz Romero · Diplomado en Analítica de Datos · Uniminuto · Datos: Inside Airbnb, Londres, septiembre de 2022.')
