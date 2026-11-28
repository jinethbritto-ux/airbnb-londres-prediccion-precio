# Estimador interactivo de precios

Aplicación de [Streamlit](https://streamlit.io) que estima el precio por noche de un alojamiento de Airbnb en Londres con el modelo final del proyecto (Gradient Boosting ajustado).

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `app.py` | La aplicación (pantalla y formulario) |
| `estimador.py` | La lógica de la predicción, sin Streamlit |
| `modelo/modelo_gb.joblib` | El modelo entrenado (2 MB) |
| `modelo/metadatos.json` | Valores por defecto, categorías, métricas y rango probable del precio |
| `requirements.txt` | Librerías con las versiones exactas |

## Ejecutarla en tu computador

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

## Publicarla con un enlace público

Ver la sección «Estimador interactivo» del [README principal](../README.md).

## Si el modelo no carga

El modelo guardado depende de la versión de scikit-learn con la que se entrenó (la de `requirements.txt`). Si la aplicación muestra un error al cargarlo, genera de nuevo el modelo con tus versiones:

```bash
python src/modelado_airbnb_londres.py --datos data/clean_df.csv --salida results
```

El script guarda el modelo y sus metadatos en `app/modelo/`. Después ajusta la versión de scikit-learn en `app/requirements.txt` a la que muestre `metadatos.json` en el campo `version_sklearn`.
