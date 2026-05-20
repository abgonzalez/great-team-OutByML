"""Constructor educativo de dataset historico para SalimosHoy?."""

import pandas as pd
import requests

from src.features import (
    apply_weather_safety_rules,
    calculate_activity_score,
    calculate_comfort_distance,
    classify_score,
    get_activity_settings,
    get_valid_hours,
)


HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables historicas que intentamos pedir directamente a Open-Meteo.
HOURLY_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "uv_index",
]

# Compatibilidad: algunas variables no existen igual en el historico.
ARCHIVE_FALLBACK_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "shortwave_radiation",
]

# Lista oficial de actividades; evita reintroducir actividades antiguas.
ACTIVITIES = [
    "Pasear o hacer senderismo",
    "Jardineria y agricultura",
    "Deportes al aire libre",
    "Picnic o actividades en parque",
    "Ir al cine",
    "Ir a la playa",
]

SAMPLE_CITIES = [
    {
        "city": "Madrid",
        "country": "España",
        "continent": "Europa",
        "latitude": 40.4165,
        "longitude": -3.70256,
        "timezone": "Europe/Madrid",
    },
    {
        "city": "Bilbao",
        "country": "España",
        "continent": "Europa",
        "latitude": 43.26271,
        "longitude": -2.92528,
        "timezone": "Europe/Madrid",
    },
    {
        "city": "London",
        "country": "Reino Unido",
        "continent": "Europa",
        "latitude": 51.50853,
        "longitude": -0.12574,
        "timezone": "Europe/London",
    },
    {
        "city": "New York",
        "country": "Estados Unidos",
        "continent": "America",
        "latitude": 40.71427,
        "longitude": -74.00597,
        "timezone": "America/New_York",
    },
    {
        "city": "Buenos Aires",
        "country": "Argentina",
        "continent": "America",
        "latitude": -34.61315,
        "longitude": -58.37723,
        "timezone": "America/Argentina/Buenos_Aires",
    },
    {
        "city": "Caracas",
        "country": "Venezuela",
        "continent": "America",
        "latitude": 10.48801,
        "longitude": -66.87919,
        "timezone": "America/Caracas",
    },
    {
        "city": "Tokyo",
        "country": "Japon",
        "continent": "Asia",
        "latitude": 35.6895,
        "longitude": 139.69171,
        "timezone": "Asia/Tokyo",
    },
    {
        "city": "Dubai",
        "country": "Emiratos Arabes Unidos",
        "continent": "Asia",
        "latitude": 25.07725,
        "longitude": 55.30927,
        "timezone": "Asia/Dubai",
    },
    {
        "city": "Cairo",
        "country": "Egipto",
        "continent": "Africa",
        "latitude": 30.06263,
        "longitude": 31.24967,
        "timezone": "Africa/Cairo",
    },
    {
        "city": "Sydney",
        "country": "Australia",
        "continent": "Oceania",
        "latitude": -33.86785,
        "longitude": 151.20732,
        "timezone": "Australia/Sydney",
    },
]

SMART_CITY_GROUPS = {
    "templado_urbano": [
        ("Madrid", "España", "Europa", 40.4165, -3.70256, "Europe/Madrid"),
        ("Barcelona", "España", "Europa", 41.38879, 2.15899, "Europe/Madrid"),
        ("Paris", "Francia", "Europa", 48.85341, 2.3488, "Europe/Paris"),
        ("London", "Reino Unido", "Europa", 51.50853, -0.12574, "Europe/London"),
        ("Rome", "Italia", "Europa", 41.89193, 12.51133, "Europe/Rome"),
        ("Lisbon", "Portugal", "Europa", 38.71667, -9.13333, "Europe/Lisbon"),
        ("Berlin", "Alemania", "Europa", 52.52437, 13.41053, "Europe/Berlin"),
        ("Amsterdam", "Paises Bajos", "Europa", 52.37403, 4.88969, "Europe/Amsterdam"),
        ("Vienna", "Austria", "Europa", 48.20849, 16.37208, "Europe/Vienna"),
        ("Prague", "Republica Checa", "Europa", 50.08804, 14.42076, "Europe/Prague"),
    ],
    "calor_seco": [
        ("Dubai", "Emiratos Arabes Unidos", "Asia", 25.07725, 55.30927, "Asia/Dubai"),
        ("Riyadh", "Arabia Saudi", "Asia", 24.68773, 46.72185, "Asia/Riyadh"),
        ("Doha", "Qatar", "Asia", 25.28545, 51.53096, "Asia/Qatar"),
        ("Cairo", "Egipto", "Africa", 30.06263, 31.24967, "Africa/Cairo"),
        ("Marrakech", "Marruecos", "Africa", 31.63416, -7.99994, "Africa/Casablanca"),
        ("Phoenix", "Estados Unidos", "America", 33.44838, -112.07404, "America/Phoenix"),
        ("Las Vegas", "Estados Unidos", "America", 36.17497, -115.13722, "America/Los_Angeles"),
        ("Seville", "España", "Europa", 37.38283, -5.97317, "Europe/Madrid"),
        ("Baghdad", "Irak", "Asia", 33.34058, 44.40088, "Asia/Baghdad"),
        ("Kuwait City", "Kuwait", "Asia", 29.36972, 47.97833, "Asia/Kuwait"),
    ],
    "frio_fuerte": [
        ("Reykjavik", "Islandia", "Europa", 64.13548, -21.89541, "Atlantic/Reykjavik"),
        ("Oslo", "Noruega", "Europa", 59.91273, 10.74609, "Europe/Oslo"),
        ("Helsinki", "Finlandia", "Europa", 60.16952, 24.93545, "Europe/Helsinki"),
        ("Stockholm", "Suecia", "Europa", 59.33258, 18.0649, "Europe/Stockholm"),
        ("Moscow", "Rusia", "Europa", 55.75222, 37.61556, "Europe/Moscow"),
        ("Montreal", "Canada", "America", 45.50884, -73.58781, "America/Toronto"),
        ("Toronto", "Canada", "America", 43.70011, -79.4163, "America/Toronto"),
        ("Ulaanbaatar", "Mongolia", "Asia", 47.90771, 106.88324, "Asia/Ulaanbaatar"),
        ("Anchorage", "Estados Unidos", "America", 61.21806, -149.90028, "America/Anchorage"),
        ("Nuuk", "Groenlandia", "America", 64.18347, -51.72157, "America/Godthab"),
    ],
    "lluvia_humedad": [
        ("Bilbao", "España", "Europa", 43.26271, -2.92528, "Europe/Madrid"),
        ("Dublin", "Irlanda", "Europa", 53.33306, -6.24889, "Europe/Dublin"),
        ("Glasgow", "Reino Unido", "Europa", 55.86515, -4.25763, "Europe/London"),
        ("Vancouver", "Canada", "America", 49.24966, -123.11934, "America/Vancouver"),
        ("Bogota", "Colombia", "America", 4.60971, -74.08175, "America/Bogota"),
        ("Quito", "Ecuador", "America", -0.22985, -78.52495, "America/Guayaquil"),
        ("Singapore", "Singapur", "Asia", 1.28967, 103.85007, "Asia/Singapore"),
        ("Kuala Lumpur", "Malasia", "Asia", 3.1412, 101.68653, "Asia/Kuala_Lumpur"),
        ("Bangkok", "Tailandia", "Asia", 13.75398, 100.50144, "Asia/Bangkok"),
        ("Mumbai", "India", "Asia", 19.07283, 72.88261, "Asia/Kolkata"),
    ],
    "tropical_humedo": [
        ("Caracas", "Venezuela", "America", 10.48801, -66.87919, "America/Caracas"),
        ("Panama City", "Panama", "America", 8.9936, -79.51973, "America/Panama"),
        ("Havana", "Cuba", "America", 23.13302, -82.38304, "America/Havana"),
        ("Santo Domingo", "Republica Dominicana", "America", 18.47186, -69.89232, "America/Santo_Domingo"),
        ("San Juan", "Puerto Rico", "America", 18.46633, -66.10572, "America/Puerto_Rico"),
        ("Manaus", "Brasil", "America", -3.10194, -60.025, "America/Manaus"),
        ("Lagos", "Nigeria", "Africa", 6.45407, 3.39467, "Africa/Lagos"),
        ("Nairobi", "Kenia", "Africa", -1.28333, 36.81667, "Africa/Nairobi"),
        ("Jakarta", "Indonesia", "Asia", -6.21462, 106.84513, "Asia/Jakarta"),
        ("Manila", "Filipinas", "Asia", 14.6042, 120.9822, "Asia/Manila"),
    ],
    "hemisferio_sur": [
        ("Buenos Aires", "Argentina", "America", -34.61315, -58.37723, "America/Argentina/Buenos_Aires"),
        ("Santiago", "Chile", "America", -33.45694, -70.64827, "America/Santiago"),
        ("Lima", "Peru", "America", -12.04318, -77.02824, "America/Lima"),
        ("Sao Paulo", "Brasil", "America", -23.5475, -46.63611, "America/Sao_Paulo"),
        ("Cape Town", "Sudafrica", "Africa", -33.92584, 18.42322, "Africa/Johannesburg"),
        ("Johannesburg", "Sudafrica", "Africa", -26.20227, 28.04363, "Africa/Johannesburg"),
        ("Sydney", "Australia", "Oceania", -33.86785, 151.20732, "Australia/Sydney"),
        ("Melbourne", "Australia", "Oceania", -37.814, 144.96332, "Australia/Melbourne"),
        ("Auckland", "Nueva Zelanda", "Oceania", -36.86667, 174.76667, "Pacific/Auckland"),
        ("Perth", "Australia", "Oceania", -31.95224, 115.8614, "Australia/Perth"),
    ],
}

SMART_CLIMATE_GROUP_ORDER = [
    "templado_urbano",
    "calor_seco",
    "frio_fuerte",
    "lluvia_humedad",
    "tropical_humedo",
    "hemisferio_sur",
]

# Catalogo final de 200 ciudades, balanceado por grupos climaticos.
SMART_CITY_GROUPS_200 = {
    "templado_urbano": [
        ("Madrid", "Espana", "Europa", 40.4165, -3.7026, "Europe/Madrid"),
        ("Barcelona", "Espana", "Europa", 41.3874, 2.1686, "Europe/Madrid"),
        ("Paris", "Francia", "Europa", 48.8566, 2.3522, "Europe/Paris"),
        ("London", "Reino Unido", "Europa", 51.5072, -0.1276, "Europe/London"),
        ("Rome", "Italia", "Europa", 41.9028, 12.4964, "Europe/Rome"),
        ("Lisbon", "Portugal", "Europa", 38.7223, -9.1393, "Europe/Lisbon"),
        ("Berlin", "Alemania", "Europa", 52.5200, 13.4050, "Europe/Berlin"),
        ("Amsterdam", "Paises Bajos", "Europa", 52.3676, 4.9041, "Europe/Amsterdam"),
        ("Vienna", "Austria", "Europa", 48.2082, 16.3738, "Europe/Vienna"),
        ("Prague", "Republica Checa", "Europa", 50.0755, 14.4378, "Europe/Prague"),
        ("Brussels", "Belgica", "Europa", 50.8503, 4.3517, "Europe/Brussels"),
        ("Zurich", "Suiza", "Europa", 47.3769, 8.5417, "Europe/Zurich"),
        ("Munich", "Alemania", "Europa", 48.1351, 11.5820, "Europe/Berlin"),
        ("Milan", "Italia", "Europa", 45.4642, 9.1900, "Europe/Rome"),
        ("Copenhagen", "Dinamarca", "Europa", 55.6761, 12.5683, "Europe/Copenhagen"),
        ("Dublin", "Irlanda", "Europa", 53.3498, -6.2603, "Europe/Dublin"),
        ("Warsaw", "Polonia", "Europa", 52.2297, 21.0122, "Europe/Warsaw"),
        ("Budapest", "Hungria", "Europa", 47.4979, 19.0402, "Europe/Budapest"),
        ("Lyon", "Francia", "Europa", 45.7640, 4.8357, "Europe/Paris"),
        ("Porto", "Portugal", "Europa", 41.1579, -8.6291, "Europe/Lisbon"),
        ("Seattle", "Estados Unidos", "America", 47.6062, -122.3321, "America/Los_Angeles"),
        ("San Francisco", "Estados Unidos", "America", 37.7749, -122.4194, "America/Los_Angeles"),
        ("New York", "Estados Unidos", "America", 40.7128, -74.0060, "America/New_York"),
        ("Boston", "Estados Unidos", "America", 42.3601, -71.0589, "America/New_York"),
        ("Chicago", "Estados Unidos", "America", 41.8781, -87.6298, "America/Chicago"),
        ("Washington", "Estados Unidos", "America", 38.9072, -77.0369, "America/New_York"),
        ("Philadelphia", "Estados Unidos", "America", 39.9526, -75.1652, "America/New_York"),
        ("Toronto", "Canada", "America", 43.6532, -79.3832, "America/Toronto"),
        ("Montreal", "Canada", "America", 45.5017, -73.5673, "America/Toronto"),
        ("Tokyo", "Japon", "Asia", 35.6762, 139.6503, "Asia/Tokyo"),
        ("Seoul", "Corea del Sur", "Asia", 37.5665, 126.9780, "Asia/Seoul"),
        ("Beijing", "China", "Asia", 39.9042, 116.4074, "Asia/Shanghai"),
        ("Istanbul", "Turquia", "Europa", 41.0082, 28.9784, "Europe/Istanbul"),
        ("Athens", "Grecia", "Europa", 37.9838, 23.7275, "Europe/Athens"),
    ],
    "calor_seco": [
        ("Dubai", "Emiratos Arabes Unidos", "Asia", 25.2048, 55.2708, "Asia/Dubai"),
        ("Abu Dhabi", "Emiratos Arabes Unidos", "Asia", 24.4539, 54.3773, "Asia/Dubai"),
        ("Riyadh", "Arabia Saudi", "Asia", 24.7136, 46.6753, "Asia/Riyadh"),
        ("Jeddah", "Arabia Saudi", "Asia", 21.4858, 39.1925, "Asia/Riyadh"),
        ("Doha", "Qatar", "Asia", 25.2854, 51.5310, "Asia/Qatar"),
        ("Kuwait City", "Kuwait", "Asia", 29.3759, 47.9774, "Asia/Kuwait"),
        ("Muscat", "Oman", "Asia", 23.5880, 58.3829, "Asia/Muscat"),
        ("Manama", "Barein", "Asia", 26.2235, 50.5876, "Asia/Bahrain"),
        ("Cairo", "Egipto", "Africa", 30.0444, 31.2357, "Africa/Cairo"),
        ("Luxor", "Egipto", "Africa", 25.6872, 32.6396, "Africa/Cairo"),
        ("Marrakech", "Marruecos", "Africa", 31.6295, -7.9811, "Africa/Casablanca"),
        ("Casablanca", "Marruecos", "Africa", 33.5731, -7.5898, "Africa/Casablanca"),
        ("Tunis", "Tunez", "Africa", 36.8065, 10.1815, "Africa/Tunis"),
        ("Algiers", "Argelia", "Africa", 36.7538, 3.0588, "Africa/Algiers"),
        ("Tripoli", "Libia", "Africa", 32.8872, 13.1913, "Africa/Tripoli"),
        ("Khartoum", "Sudan", "Africa", 15.5007, 32.5599, "Africa/Khartoum"),
        ("Phoenix", "Estados Unidos", "America", 33.4484, -112.0740, "America/Phoenix"),
        ("Las Vegas", "Estados Unidos", "America", 36.1699, -115.1398, "America/Los_Angeles"),
        ("Tucson", "Estados Unidos", "America", 32.2226, -110.9747, "America/Phoenix"),
        ("El Paso", "Estados Unidos", "America", 31.7619, -106.4850, "America/Denver"),
        ("Palm Springs", "Estados Unidos", "America", 33.8303, -116.5453, "America/Los_Angeles"),
        ("Mexicali", "Mexico", "America", 32.6245, -115.4523, "America/Tijuana"),
        ("Hermosillo", "Mexico", "America", 29.0729, -110.9559, "America/Hermosillo"),
        ("Lima", "Peru", "America", -12.0464, -77.0428, "America/Lima"),
        ("Arequipa", "Peru", "America", -16.4090, -71.5375, "America/Lima"),
        ("Antofagasta", "Chile", "America", -23.6509, -70.3975, "America/Santiago"),
        ("Mendoza", "Argentina", "America", -32.8895, -68.8458, "America/Argentina/Mendoza"),
        ("Alice Springs", "Australia", "Oceania", -23.6980, 133.8807, "Australia/Darwin"),
        ("Perth", "Australia", "Oceania", -31.9523, 115.8613, "Australia/Perth"),
        ("Nouakchott", "Mauritania", "Africa", 18.0735, -15.9582, "Africa/Nouakchott"),
        ("Niamey", "Niger", "Africa", 13.5116, 2.1254, "Africa/Niamey"),
        ("Seville", "Espana", "Europa", 37.3891, -5.9845, "Europe/Madrid"),
        ("Baghdad", "Irak", "Asia", 33.3152, 44.3661, "Asia/Baghdad"),
    ],
    "frio_fuerte": [
        ("Reykjavik", "Islandia", "Europa", 64.1466, -21.9426, "Atlantic/Reykjavik"),
        ("Oslo", "Noruega", "Europa", 59.9139, 10.7522, "Europe/Oslo"),
        ("Helsinki", "Finlandia", "Europa", 60.1699, 24.9384, "Europe/Helsinki"),
        ("Stockholm", "Suecia", "Europa", 59.3293, 18.0686, "Europe/Stockholm"),
        ("Tromso", "Noruega", "Europa", 69.6492, 18.9553, "Europe/Oslo"),
        ("Oulu", "Finlandia", "Europa", 65.0121, 25.4651, "Europe/Helsinki"),
        ("Rovaniemi", "Finlandia", "Europa", 66.5039, 25.7294, "Europe/Helsinki"),
        ("Kiruna", "Suecia", "Europa", 67.8558, 20.2253, "Europe/Stockholm"),
        ("Moscow", "Rusia", "Europa", 55.7558, 37.6173, "Europe/Moscow"),
        ("Saint Petersburg", "Rusia", "Europa", 59.9311, 30.3609, "Europe/Moscow"),
        ("Novosibirsk", "Rusia", "Asia", 55.0084, 82.9357, "Asia/Novosibirsk"),
        ("Yakutsk", "Rusia", "Asia", 62.0355, 129.6755, "Asia/Yakutsk"),
        ("Irkutsk", "Rusia", "Asia", 52.2864, 104.2807, "Asia/Irkutsk"),
        ("Ulaanbaatar", "Mongolia", "Asia", 47.8864, 106.9057, "Asia/Ulaanbaatar"),
        ("Astana", "Kazajistan", "Asia", 51.1694, 71.4491, "Asia/Almaty"),
        ("Harbin", "China", "Asia", 45.8038, 126.5349, "Asia/Shanghai"),
        ("Sapporo", "Japon", "Asia", 43.0618, 141.3545, "Asia/Tokyo"),
        ("Anchorage", "Estados Unidos", "America", 61.2181, -149.9003, "America/Anchorage"),
        ("Fairbanks", "Estados Unidos", "America", 64.8378, -147.7164, "America/Anchorage"),
        ("Juneau", "Estados Unidos", "America", 58.3019, -134.4197, "America/Juneau"),
        ("Winnipeg", "Canada", "America", 49.8951, -97.1384, "America/Winnipeg"),
        ("Edmonton", "Canada", "America", 53.5461, -113.4938, "America/Edmonton"),
        ("Calgary", "Canada", "America", 51.0447, -114.0719, "America/Edmonton"),
        ("Quebec City", "Canada", "America", 46.8139, -71.2080, "America/Toronto"),
        ("Ottawa", "Canada", "America", 45.4215, -75.6972, "America/Toronto"),
        ("Halifax", "Canada", "America", 44.6488, -63.5752, "America/Halifax"),
        ("Nuuk", "Groenlandia", "America", 64.1814, -51.6941, "America/Nuuk"),
        ("Torshavn", "Islas Feroe", "Europa", 62.0079, -6.7900, "Atlantic/Faroe"),
        ("Stavanger", "Noruega", "Europa", 58.9690, 5.7331, "Europe/Oslo"),
        ("Tallinn", "Estonia", "Europa", 59.4370, 24.7536, "Europe/Tallinn"),
        ("Riga", "Letonia", "Europa", 56.9496, 24.1052, "Europe/Riga"),
        ("Vilnius", "Lituania", "Europa", 54.6872, 25.2797, "Europe/Vilnius"),
        ("Minneapolis", "Estados Unidos", "America", 44.9778, -93.2650, "America/Chicago"),
    ],
    "lluvia_humedad": [
        ("Bilbao", "Espana", "Europa", 43.2630, -2.9350, "Europe/Madrid"),
        ("San Sebastian", "Espana", "Europa", 43.3183, -1.9812, "Europe/Madrid"),
        ("Glasgow", "Reino Unido", "Europa", 55.8642, -4.2518, "Europe/London"),
        ("Manchester", "Reino Unido", "Europa", 53.4808, -2.2426, "Europe/London"),
        ("Cardiff", "Reino Unido", "Europa", 51.4816, -3.1791, "Europe/London"),
        ("Belfast", "Reino Unido", "Europa", 54.5973, -5.9301, "Europe/London"),
        ("Galway", "Irlanda", "Europa", 53.2707, -9.0568, "Europe/Dublin"),
        ("Brest", "Francia", "Europa", 48.3904, -4.4861, "Europe/Paris"),
        ("Hamburg", "Alemania", "Europa", 53.5511, 9.9937, "Europe/Berlin"),
        ("Gothenburg", "Suecia", "Europa", 57.7089, 11.9746, "Europe/Stockholm"),
        ("Bergen", "Noruega", "Europa", 60.3913, 5.3221, "Europe/Oslo"),
        ("Portland", "Estados Unidos", "America", 45.5152, -122.6784, "America/Los_Angeles"),
        ("Eugene", "Estados Unidos", "America", 44.0521, -123.0868, "America/Los_Angeles"),
        ("Vancouver", "Canada", "America", 49.2827, -123.1207, "America/Vancouver"),
        ("Victoria", "Canada", "America", 48.4284, -123.3656, "America/Vancouver"),
        ("Bogota", "Colombia", "America", 4.7110, -74.0721, "America/Bogota"),
        ("Medellin", "Colombia", "America", 6.2442, -75.5812, "America/Bogota"),
        ("Quito", "Ecuador", "America", -0.1807, -78.4678, "America/Guayaquil"),
        ("Cuenca", "Ecuador", "America", -2.9006, -79.0045, "America/Guayaquil"),
        ("San Jose", "Costa Rica", "America", 9.9281, -84.0907, "America/Costa_Rica"),
        ("Singapore", "Singapur", "Asia", 1.3521, 103.8198, "Asia/Singapore"),
        ("Kuala Lumpur", "Malasia", "Asia", 3.1390, 101.6869, "Asia/Kuala_Lumpur"),
        ("Penang", "Malasia", "Asia", 5.4141, 100.3288, "Asia/Kuala_Lumpur"),
        ("Bangkok", "Tailandia", "Asia", 13.7563, 100.5018, "Asia/Bangkok"),
        ("Chiang Mai", "Tailandia", "Asia", 18.7883, 98.9853, "Asia/Bangkok"),
        ("Mumbai", "India", "Asia", 19.0760, 72.8777, "Asia/Kolkata"),
        ("Kochi", "India", "Asia", 9.9312, 76.2673, "Asia/Kolkata"),
        ("Colombo", "Sri Lanka", "Asia", 6.9271, 79.8612, "Asia/Colombo"),
        ("Dhaka", "Bangladesh", "Asia", 23.8103, 90.4125, "Asia/Dhaka"),
        ("Hanoi", "Vietnam", "Asia", 21.0278, 105.8342, "Asia/Bangkok"),
        ("Ho Chi Minh City", "Vietnam", "Asia", 10.8231, 106.6297, "Asia/Ho_Chi_Minh"),
        ("Taipei", "Taiwan", "Asia", 25.0330, 121.5654, "Asia/Taipei"),
        ("Hong Kong", "China", "Asia", 22.3193, 114.1694, "Asia/Hong_Kong"),
    ],
    "tropical_humedo": [
        ("Caracas", "Venezuela", "America", 10.4806, -66.9036, "America/Caracas"),
        ("Panama City", "Panama", "America", 8.9824, -79.5199, "America/Panama"),
        ("Havana", "Cuba", "America", 23.1136, -82.3666, "America/Havana"),
        ("Santo Domingo", "Republica Dominicana", "America", 18.4861, -69.9312, "America/Santo_Domingo"),
        ("San Juan", "Puerto Rico", "America", 18.4655, -66.1057, "America/Puerto_Rico"),
        ("Kingston", "Jamaica", "America", 18.0179, -76.8099, "America/Jamaica"),
        ("Belize City", "Belice", "America", 17.5046, -88.1962, "America/Belize"),
        ("Georgetown", "Guyana", "America", 6.8013, -58.1551, "America/Guyana"),
        ("Paramaribo", "Surinam", "America", 5.8520, -55.2038, "America/Paramaribo"),
        ("Manaus", "Brasil", "America", -3.1190, -60.0217, "America/Manaus"),
        ("Belem", "Brasil", "America", -1.4558, -48.4902, "America/Belem"),
        ("Recife", "Brasil", "America", -8.0476, -34.8770, "America/Recife"),
        ("Salvador", "Brasil", "America", -12.9777, -38.5016, "America/Bahia"),
        ("Guayaquil", "Ecuador", "America", -2.1894, -79.8891, "America/Guayaquil"),
        ("Lagos", "Nigeria", "Africa", 6.5244, 3.3792, "Africa/Lagos"),
        ("Accra", "Ghana", "Africa", 5.6037, -0.1870, "Africa/Accra"),
        ("Abidjan", "Costa de Marfil", "Africa", 5.3600, -4.0083, "Africa/Abidjan"),
        ("Douala", "Camerun", "Africa", 4.0511, 9.7679, "Africa/Douala"),
        ("Libreville", "Gabon", "Africa", 0.4162, 9.4673, "Africa/Libreville"),
        ("Kinshasa", "Republica Democratica del Congo", "Africa", -4.4419, 15.2663, "Africa/Kinshasa"),
        ("Mombasa", "Kenia", "Africa", -4.0435, 39.6682, "Africa/Nairobi"),
        ("Dar es Salaam", "Tanzania", "Africa", -6.7924, 39.2083, "Africa/Dar_es_Salaam"),
        ("Antananarivo", "Madagascar", "Africa", -18.8792, 47.5079, "Indian/Antananarivo"),
        ("Jakarta", "Indonesia", "Asia", -6.2088, 106.8456, "Asia/Jakarta"),
        ("Surabaya", "Indonesia", "Asia", -7.2575, 112.7521, "Asia/Jakarta"),
        ("Denpasar", "Indonesia", "Asia", -8.6705, 115.2126, "Asia/Makassar"),
        ("Manila", "Filipinas", "Asia", 14.5995, 120.9842, "Asia/Manila"),
        ("Cebu", "Filipinas", "Asia", 10.3157, 123.8854, "Asia/Manila"),
        ("Davao", "Filipinas", "Asia", 7.1907, 125.4553, "Asia/Manila"),
        ("Phnom Penh", "Camboya", "Asia", 11.5564, 104.9282, "Asia/Phnom_Penh"),
        ("Vientiane", "Laos", "Asia", 17.9757, 102.6331, "Asia/Vientiane"),
        ("Brunei", "Brunei", "Asia", 4.9031, 114.9398, "Asia/Brunei"),
        ("Suva", "Fiyi", "Oceania", -18.1248, 178.4501, "Pacific/Fiji"),
    ],
    "hemisferio_sur": [
        ("Buenos Aires", "Argentina", "America", -34.6037, -58.3816, "America/Argentina/Buenos_Aires"),
        ("Cordoba", "Argentina", "America", -31.4201, -64.1888, "America/Argentina/Cordoba"),
        ("Rosario", "Argentina", "America", -32.9442, -60.6505, "America/Argentina/Cordoba"),
        ("Santiago", "Chile", "America", -33.4489, -70.6693, "America/Santiago"),
        ("Valparaiso", "Chile", "America", -33.0472, -71.6127, "America/Santiago"),
        ("Montevideo", "Uruguay", "America", -34.9011, -56.1645, "America/Montevideo"),
        ("Asuncion", "Paraguay", "America", -25.2637, -57.5759, "America/Asuncion"),
        ("La Paz", "Bolivia", "America", -16.4897, -68.1193, "America/La_Paz"),
        ("Santa Cruz", "Bolivia", "America", -17.7833, -63.1821, "America/La_Paz"),
        ("Sao Paulo", "Brasil", "America", -23.5558, -46.6396, "America/Sao_Paulo"),
        ("Rio de Janeiro", "Brasil", "America", -22.9068, -43.1729, "America/Sao_Paulo"),
        ("Curitiba", "Brasil", "America", -25.4284, -49.2733, "America/Sao_Paulo"),
        ("Porto Alegre", "Brasil", "America", -30.0346, -51.2177, "America/Sao_Paulo"),
        ("Florianopolis", "Brasil", "America", -27.5949, -48.5482, "America/Sao_Paulo"),
        ("Cape Town", "Sudafrica", "Africa", -33.9249, 18.4241, "Africa/Johannesburg"),
        ("Johannesburg", "Sudafrica", "Africa", -26.2041, 28.0473, "Africa/Johannesburg"),
        ("Durban", "Sudafrica", "Africa", -29.8587, 31.0218, "Africa/Johannesburg"),
        ("Pretoria", "Sudafrica", "Africa", -25.7479, 28.2293, "Africa/Johannesburg"),
        ("Maputo", "Mozambique", "Africa", -25.9692, 32.5732, "Africa/Maputo"),
        ("Harare", "Zimbabue", "Africa", -17.8252, 31.0335, "Africa/Harare"),
        ("Lusaka", "Zambia", "Africa", -15.3875, 28.3228, "Africa/Lusaka"),
        ("Gaborone", "Botsuana", "Africa", -24.6282, 25.9231, "Africa/Gaborone"),
        ("Windhoek", "Namibia", "Africa", -22.5609, 17.0658, "Africa/Windhoek"),
        ("Sydney", "Australia", "Oceania", -33.8688, 151.2093, "Australia/Sydney"),
        ("Melbourne", "Australia", "Oceania", -37.8136, 144.9631, "Australia/Melbourne"),
        ("Brisbane", "Australia", "Oceania", -27.4698, 153.0251, "Australia/Brisbane"),
        ("Adelaide", "Australia", "Oceania", -34.9285, 138.6007, "Australia/Adelaide"),
        ("Hobart", "Australia", "Oceania", -42.8821, 147.3272, "Australia/Hobart"),
        ("Canberra", "Australia", "Oceania", -35.2809, 149.1300, "Australia/Sydney"),
        ("Darwin", "Australia", "Oceania", -12.4634, 130.8456, "Australia/Darwin"),
        ("Auckland", "Nueva Zelanda", "Oceania", -36.8509, 174.7645, "Pacific/Auckland"),
        ("Wellington", "Nueva Zelanda", "Oceania", -41.2865, 174.7762, "Pacific/Auckland"),
        ("Christchurch", "Nueva Zelanda", "Oceania", -43.5321, 172.6362, "Pacific/Auckland"),
        ("Dunedin", "Nueva Zelanda", "Oceania", -45.8788, 170.5028, "Pacific/Auckland"),
    ],
}

STRATEGIC_MONTHS = [
    ("enero", 1, 31),
    ("marzo", 3, 31),
    ("mayo", 5, 31),
    ("julio", 7, 31),
    ("septiembre", 9, 30),
    ("noviembre", 11, 30),
]


def get_sample_cities():
    """Devuelve una lista inicial de ciudades de prueba."""
    return SAMPLE_CITIES.copy()


def get_smart_cities():
    """Devuelve ciudades variadas por grupo climatico para construir datasets."""
    cities = []
    max_group_size = max(len(group) for group in SMART_CITY_GROUPS.values())

    for index in range(max_group_size):
        for climate_group in SMART_CLIMATE_GROUP_ORDER:
            group_cities = SMART_CITY_GROUPS[climate_group]
            if index < len(group_cities):
                city, country, continent, latitude, longitude, timezone = group_cities[index]
                cities.append(
                    {
                        "city": city,
                        "country": country,
                        "continent": continent,
                        "climate_group": climate_group,
                        "latitude": latitude,
                        "longitude": longitude,
                        "timezone": timezone,
                    }
                )

    return cities


def get_sample_smart_cities(n=24):
    """Devuelve una muestra pequena balanceada por grupo climatico."""
    all_cities = get_smart_cities()
    cities_by_group = {
        climate_group: [
            city for city in all_cities if city["climate_group"] == climate_group
        ]
        for climate_group in SMART_CLIMATE_GROUP_ORDER
    }
    base_amount = n // len(SMART_CLIMATE_GROUP_ORDER)
    extra_amount = n % len(SMART_CLIMATE_GROUP_ORDER)

    selected_cities = []
    for index, climate_group in enumerate(SMART_CLIMATE_GROUP_ORDER):
        group_limit = base_amount
        if index < extra_amount:
            group_limit += 1
        selected_cities.extend(cities_by_group[climate_group][:group_limit])

    return selected_cities[:n]


def _build_city_records(city_groups):
    """Convierte grupos de tuplas en diccionarios de ciudad."""
    cities = []
    for climate_group in SMART_CLIMATE_GROUP_ORDER:
        for city, country, continent, latitude, longitude, timezone in city_groups[
            climate_group
        ]:
            cities.append(
                {
                    "city": city,
                    "country": country,
                    "continent": continent,
                    "climate_group": climate_group,
                    "latitude": latitude,
                    "longitude": longitude,
                    "timezone": timezone,
                }
            )
    return cities


def get_smart_cities_200():
    """Devuelve 200 ciudades balanceadas por grupo climatico para dataset v2."""
    # La app demo usa este mismo catalogo para coordenadas y metadata.
    return _build_city_records(SMART_CITY_GROUPS_200)


def get_sample_smart_cities_v2(n=30):
    """Devuelve una muestra v2 balanceada por grupo climatico."""
    all_cities = get_smart_cities_200()
    cities_by_group = {
        climate_group: [
            city for city in all_cities if city["climate_group"] == climate_group
        ]
        for climate_group in SMART_CLIMATE_GROUP_ORDER
    }
    base_amount = n // len(SMART_CLIMATE_GROUP_ORDER)
    extra_amount = n % len(SMART_CLIMATE_GROUP_ORDER)

    selected_cities = []
    for index, climate_group in enumerate(SMART_CLIMATE_GROUP_ORDER):
        # Reparte la muestra entre grupos para no sesgar el dataset.
        group_limit = base_amount
        if index < extra_amount:
            group_limit += 1
        selected_cities.extend(cities_by_group[climate_group][:group_limit])

    return selected_cities[:n]


def get_strategic_month_ranges(year=2025):
    """Devuelve los rangos mensuales estrategicos para el dataset v2."""
    # Se entreno con meses alternos para cubrir estaciones sin inflar el dataset.
    return [
        {
            "season_block": season_block,
            "start_date": f"{year}-{month:02d}-01",
            "end_date": f"{year}-{month:02d}-{last_day:02d}",
        }
        for season_block, month, last_day in STRATEGIC_MONTHS
    ]


def get_historical_weather(latitude, longitude, timezone, start_date, end_date):
    """Obtiene clima historico horario desde Open-Meteo."""
    # Primera llamada: intenta usar las mismas variables que consume la app.
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone,
    }

    try:
        response = requests.get(HISTORICAL_WEATHER_URL, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        print(
            "No se pudo descargar el historico con todas las variables. "
            "Se intentara una consulta compatible con Historical Weather API."
        )
    except ValueError:
        print(
            "No se pudo leer el historico con todas las variables. "
            "Se intentara una consulta compatible con Historical Weather API."
        )

    fallback_params = params.copy()
    fallback_params["hourly"] = ",".join(ARCHIVE_FALLBACK_VARIABLES)

    # Segunda llamada: usa variables disponibles y las aproxima despues.
    try:
        response = requests.get(HISTORICAL_WEATHER_URL, params=fallback_params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        print("Error al consultar el clima historico. Revisa la conexion o intenta mas tarde.")
        return None
    except ValueError:
        print("Error al leer la respuesta del clima historico.")
        return None


def build_weather_dataframe(weather_data, city_info):
    """Convierte la respuesta historica en un DataFrame horario con metadatos."""
    if weather_data is None or "hourly" not in weather_data:
        return pd.DataFrame()

    try:
        df = pd.DataFrame(weather_data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.hour

        # Aproxima columnas faltantes para mantener el mismo esquema final.
        if "precipitation_probability" not in df.columns and "precipitation" in df.columns:
            df["precipitation_probability"] = df["precipitation"].apply(
                _precipitation_to_probability
            )
        if "uv_index" not in df.columns and "shortwave_radiation" in df.columns:
            df["uv_index"] = (df["shortwave_radiation"] / 100).clip(lower=0, upper=10)

        df["city"] = city_info["city"]
        df["country"] = city_info["country"]
        df["continent"] = city_info["continent"]
        df["climate_group"] = city_info.get("climate_group", "sin_grupo")
        df["latitude"] = city_info["latitude"]
        df["longitude"] = city_info["longitude"]
        df["timezone"] = city_info["timezone"]

        return clean_weather_dataframe(df)
    except Exception:
        print("No se pudieron procesar los datos historicos.")
        return pd.DataFrame()


def clean_weather_dataframe(df):
    """Limpia valores faltantes del clima horario historico."""
    df_clean = df.copy()
    numeric_columns = [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "precipitation_probability",
        "wind_speed_10m",
        "wind_gusts_10m",
        "cloud_cover",
        "uv_index",
    ]

    for column in numeric_columns:
        # El modelo necesita columnas numericas completas y consistentes.
        if column not in df_clean.columns:
            df_clean[column] = pd.NA
        df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")

    df_clean["temperature_2m"] = (
        df_clean["temperature_2m"].ffill().bfill().fillna(20)
    )
    df_clean["apparent_temperature"] = df_clean["apparent_temperature"].fillna(
        df_clean["temperature_2m"]
    )
    df_clean["relative_humidity_2m"] = df_clean["relative_humidity_2m"].fillna(
        df_clean["relative_humidity_2m"].median()
    ).fillna(50)
    df_clean["precipitation_probability"] = df_clean[
        "precipitation_probability"
    ].fillna(0)
    df_clean["wind_speed_10m"] = df_clean["wind_speed_10m"].fillna(0)
    df_clean["wind_gusts_10m"] = df_clean["wind_gusts_10m"].fillna(
        df_clean["wind_speed_10m"]
    )
    df_clean["cloud_cover"] = df_clean["cloud_cover"].fillna(
        df_clean["cloud_cover"].median()
    ).fillna(50)
    df_clean["uv_index"] = df_clean["uv_index"].fillna(0)

    return df_clean


def expand_by_activity(df_weather):
    """Crea una fila por actividad y conserva solo horas validas."""
    if df_weather.empty:
        return pd.DataFrame()

    activity_frames = []

    for activity in ACTIVITIES:
        # Cada actividad tiene rango horario e ideales climaticos propios.
        settings = get_activity_settings(activity)
        df_activity = get_valid_hours(
            df_weather,
            settings["start_hour"],
            settings["end_hour"],
        )

        if df_activity.empty:
            continue

        df_activity = df_activity.copy()
        df_activity["activity"] = activity
        df_activity["activity_score"] = df_activity.apply(
            lambda row: calculate_activity_score(row, settings),
            axis=1,
        )
        df_activity["comfort_distance"] = df_activity.apply(
            lambda row: calculate_comfort_distance(row, settings),
            axis=1,
        )
        df_activity["recommendation"] = df_activity["activity_score"].apply(classify_score)
        # Las etiquetas del dataset ya incorporan reglas de seguridad climatica.
        df_activity["recommendation"] = df_activity.apply(
            lambda row: apply_weather_safety_rules(
                row,
                activity,
                row["recommendation"],
            ),
            axis=1,
        )
        activity_frames.append(df_activity)

    if not activity_frames:
        return pd.DataFrame()

    return pd.concat(activity_frames, ignore_index=True)


def build_dataset(cities, start_date, end_date):
    """Construye un dataset historico pequeno para varias ciudades."""
    dataset_frames = []

    for city_info in cities:
        print(f"Procesando {city_info['city']}, {city_info['country']}...")
        weather_data = get_historical_weather(
            city_info["latitude"],
            city_info["longitude"],
            city_info["timezone"],
            start_date,
            end_date,
        )
        df_weather = build_weather_dataframe(weather_data, city_info)
        df_activity = expand_by_activity(df_weather)

        if not df_activity.empty:
            dataset_frames.append(df_activity)

    if not dataset_frames:
        return pd.DataFrame()

    return pd.concat(dataset_frames, ignore_index=True)


def _precipitation_to_probability(precipitation):
    """Crea una probabilidad simple a partir de milimetros historicos."""
    if pd.isna(precipitation) or precipitation <= 0:
        return 0
    if precipitation <= 0.5:
        return 35
    if precipitation <= 2:
        return 70
    return 90
