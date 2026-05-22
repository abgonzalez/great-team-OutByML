# SalimosHoy? - Demo GitHub

SalimosHoy? es una app web en Streamlit que ayuda a decidir si conviene hacer una actividad en una ciudad y hora concreta usando clima actualizado, Machine Learning y reglas finales de seguridad climatica.

Esta version demo/GitHub esta preparada para usar un unico modelo ligero optimizado para repositorios normales de GitHub.

---

## Modelo oficial de la demo

La app carga este modelo:

```text
models/light/salimoshoy_rf_100_depth16_compressed.pkl
```

Metricas del modelo ligero:

| Metrica | Valor |
|---|---:|
| Peso | 28.49 MB |
| Accuracy test | 0.994267 |
| Recall malo | 0.992699 |
| Recall regular | 0.992456 |

El clima viene de Open-Meteo. El modelo no predice el clima: clasifica las condiciones climaticas disponibles para la actividad elegida y devuelve una recomendacion entre **malo**, **regular**, **bueno** y **excelente**.

El dataset completo y los modelos pesados forman parte del respaldo tecnico del proyecto, pero no son necesarios para ejecutar esta demo ligera.

---

## Como funciona

Flujo principal:

```text
Usuario selecciona ciudad y actividad
↓
El catalogo interno de 200 ciudades aporta coordenadas y metadata
↓
Open-Meteo Forecast API obtiene clima actualizado por hora
↓
La app prepara las 13 features del modelo
↓
El modelo clasifica las condiciones para la actividad
↓
Reglas finales de seguridad ajustan casos extremos
↓
La app muestra recommendation_final, mapa, graficos y Top 5
```

Las reglas finales evitan recomendaciones poco realistas ante lluvia fuerte, viento fuerte o temperaturas extremas.

---

## Features del modelo

Features numericas:

- `temperature_2m`
- `apparent_temperature`
- `relative_humidity_2m`
- `precipitation_probability`
- `wind_speed_10m`
- `wind_gusts_10m`
- `cloud_cover`
- `uv_index`
- `hour`

Features categoricas:

- `activity`
- `continent`
- `climate_group`
- `season_block`

Target de entrenamiento:

- `recommendation`

Columnas que no se usan como features:

- `activity_score`
- `comfort_distance`
- `recommendation`
- `time`
- `city`
- `country`
- `latitude`
- `longitude`
- `timezone`

---

## Actividades disponibles

- Pasear o hacer senderismo
- Jardineria y agricultura
- Deportes al aire libre
- Picnic o actividades en parque
- Ir al cine
- Ir a la playa

---

## APIs utilizadas

Open-Meteo Forecast API:

```text
https://api.open-meteo.com/v1/forecast
```

No se requiere API key.

La demo usa un catalogo interno de 200 ciudades, por lo que no necesita geocoding para las ciudades del selector.

---

## Instalacion

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activacion:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## Ejecutar la app

Desde la raiz del proyecto:

```bash
python -m streamlit run app.py
```

La app se abre en:

```text
http://localhost:8501
```

---

## Estructura minima de la demo

```text
SalimosHoy_demo/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── assets/
├── models/
│   └── light/
│       └── salimoshoy_rf_100_depth16_compressed.pkl
└── src/
    ├── config.py
    ├── dataset_builder.py
    ├── features.py
    ├── geocoding.py
    ├── mapping.py
    ├── recommendations.py
    ├── visualization.py
    └── weather_api.py
```

Los scripts de entrenamiento, notebooks, datasets completos, reportes tecnicos y modelos pesados no son obligatorios para ejecutar esta demo.

---

## Despliegue

### Streamlit Cloud

1. Subir la version demo a GitHub.
2. Crear una app en Streamlit Cloud.
3. Usar `app.py` como archivo principal.
4. Verificar que `requirements.txt` este incluido.
5. Verificar que el modelo ligero oficial este en `models/light/`.

### Render

| Configuracion | Valor |
|---|---|
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true` |

---

## Limitaciones

- SalimosHoy? no reemplaza una app meteorologica profesional.
- El clima actualizado depende de Open-Meteo.
- El modelo clasifica condiciones climaticas para actividades; no predice el clima.
- Si el modelo no esta disponible, la app usa reglas de respaldo para no romper la experiencia.
- La calidad de la recomendacion depende de los datos disponibles para la ciudad y hora seleccionadas.

---

## Nota final
---

## Limitaciones

- SalimosHoy? no reemplaza una app meteorologica profesional.
- El clima actualizado depende de Open-Meteo.
- El modelo clasifica condiciones climaticas para actividades; no predice el clima.
- Si el modelo no esta disponible, la app usa reglas de respaldo para no romper la experiencia.
- La calidad de la recomendacion depende de los datos disponibles para la ciudad y hora seleccionadas.

---

## Nota final

SalimosHoy? convierte datos climaticos reales en una decision practica:

```text
Conviene salir ahora?
Cual es la mejor hora?
Que tan favorable es esta actividad segun el clima?
```

SalimosHoy? convierte datos climaticos reales en una decision practica:

```text
Conviene salir ahora?
Cual es la mejor hora?
Que tan favorable es esta actividad segun el clima?
```
