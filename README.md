# OutByML - Recomendador de Actividades según el Clima

Aplicación web que recomienda la mejor hora del día para realizar una actividad al aire libre, basándose en datos meteorológicos en tiempo real de la API de Open-Meteo.

## Descripción

OutByML permite al usuario:

1. Buscar una ciudad por nombre (usando la API de geocodificación de Open-Meteo).
2. Seleccionar una actividad: Pasear, Turismo, Deporte, Bici o Lavar ropa.
3. Obtener la mejor franja horaria del día según las condiciones climáticas (temperatura, humedad, viento, lluvia, índice UV).
4. Visualizar gráficos interactivos del clima por hora y un mapa de la ciudad seleccionada.

## Actividades iniciales

- Pasear
- Turismo
- Deporte
- Bici
- Lavar ropa

## Tecnologias usadas

- Python
- Requests
- Pandas
- Streamlit
- Plotly
- Pydeck
- Open-Meteo Geocoding API
- Open-Meteo Forecast API

## Requisitos previos

- **Python 3.12** o superior
- **pip** (gestor de paquetes de Python)
- Conexión a internet (para consultar las APIs de clima y geocodificación)

## Instalación

### 1. Clonar o descargar el proyecto

```bash
git clone <url-del-repositorio>
cd project
```

### 2. Crear un entorno virtual

```bash

```

### 3. Activar el entorno virtual

**macOS / Linux:**

```bash
source env/bin/activate
```

**Windows:**

```bash
env\Scripts\activate
```

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

## Ejecución en local

Con el entorno virtual activado, ejecuta la aplicación web:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

### Flujo de uso

1. En el panel lateral, escribe el nombre de una ciudad.
2. Selecciona una actividad del desplegable.
3. Presiona el botón **Buscar**.
4. Selecciona la ciudad correcta de los resultados.
5. La aplicación mostrará:
   - La mejor hora recomendada con su puntuación.
   - El top 5 de mejores horas en una tabla.
   - Gráficos interactivos de temperatura, lluvia, viento y score por hora.
   - Un mapa interactivo con la ubicación de la ciudad.

## Despliegue en Streamlit Cloud

Para desplegar la aplicación de forma gratuita en [Streamlit Cloud](https://streamlit.io/cloud):

1. Sube el proyecto a un repositorio de GitHub (asegúrate de **no** incluir la carpeta `env/`).
2. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Haz clic en **"New app"** y selecciona:
   - **Repositorio:** tu repositorio de GitHub.
   - **Rama:** `main` (o la rama principal).
   - **Archivo principal:** `app.py`
4. Haz clic en **Deploy** y espera a que la aplicación se construya.

Esta aplication esta publicada en: `https://outbyml.streamlit.app/`

## Estructura del proyecto

```
project/
├── README.md                # Este archivo
├── requirements.txt         # Dependencias del proyecto
├── app.py                   # Aplicación web (Streamlit)
├── test_outbyml_api.py      # Script CLI original (referencia)
├── .gitignore               # Archivos a excluir del repositorio
└── env/                     # Entorno virtual (no incluir en control de versiones)
```

## APIs utilizadas

- [Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api) — Búsqueda de ciudades por nombre.
- [Open-Meteo Weather Forecast API](https://open-meteo.com/en/docs) — Pronóstico meteorológico por hora.

## Actividades disponibles

| Actividad   | Franja horaria | Temp. ideal | Humedad ideal | Viento ideal |
|-------------|----------------|-------------|---------------|--------------|
| Pasear      | 07:00 - 22:00  | 22°C        | 50%           | 5 km/h       |
| Turismo     | 08:00 - 20:00  | 21°C        | 50%           | 6 km/h       |
| Deporte     | 06:00 - 21:00  | 18°C        | 45%           | 5 km/h       |
| Bici        | 07:00 - 21:00  | 20°C        | 50%           | 4 km/h       |
| Lavar ropa  | 09:00 - 18:00  | 24°C        | 35%           | 10 km/h      |

## Notas

- No se requiere ninguna API key; las APIs de Open-Meteo son gratuitas y abiertas.
- La aplicación web utiliza Streamlit y se puede desplegar gratuitamente en Streamlit Cloud.
- Los gráficos y el mapa son interactivos directamente en el navegador.
