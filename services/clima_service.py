import os
import json
from urllib.request import urlopen, Request
from urllib.error import URLError

CAUSAS_AMBIENTALES = {
    'hongo': [
        {'causa': 'Alta humedad relativa (>80%)', 'detalle': 'Condiciones favorables para desarrollo de hongos'},
        {'causa': 'Temperaturas moderadas (15-25°C)', 'detalle': 'Rango óptimo para esporulación fúngica'},
        {'causa': 'Poca ventilación', 'detalle': 'El aire estancado facilita la propagación'},
        {'causa': 'Riego excesivo o lluvias frecuentes', 'detalle': 'El agua libre en hojas permite la germinación de esporas'},
    ],
    'bacteria': [
        {'causa': 'Heridas en tejidos vegetales', 'detalle': 'Puertas de entrada para bacterias'},
        {'causa': 'Alta humedad y temperatura', 'detalle': 'Condiciones ideales para multiplicación bacteriana'},
        {'causa': 'Riego por aspersión', 'detalle': 'Facilita la dispersión de bacterias'},
    ],
    'virus': [
        {'causa': 'Presencia de insectos vectores', 'detalle': 'Pulgones, moscas blancas y trips transmiten virus'},
        {'causa': 'Material vegetal infectado', 'detalle': 'Semillas o esquejes contaminados'},
        {'causa': 'Herramientas contaminadas', 'detalle': 'Transmisión mecánica durante podas'},
    ],
    'plaga': [
        {'causa': 'Monocultivo extensivo', 'detalle': 'Falta de biodiversidad favorece plagas'},
        {'causa': 'Falta de enemigos naturales', 'detalle': 'Desequilibrio ecológico'},
        {'causa': 'Condiciones de estrés hídrico', 'detalle': 'Plantas débiles más susceptibles'},
    ],
    'sano': [
        {'causa': 'Buenas prácticas agrícolas', 'detalle': 'Manejo integrado del cultivo'},
        {'causa': 'Condiciones climáticas favorables', 'detalle': 'Temperatura y humedad equilibradas'},
        {'causa': 'Suelo con nutrientes balanceados', 'detalle': 'Fertilización adecuada'},
    ],
}


def analizar_causas(clase, es_sana):
    if not clase or clase in ('Resultado_poco_confiable', 'No_reconocida'):
        return [{'causa': 'No se pudo determinar', 'detalle': 'Análisis no concluyente'}]
    if es_sana:
        return CAUSAS_AMBIENTALES['sano']
    c = clase.lower()
    if any(x in c for x in ['scab', 'rust', 'blight', 'mildew', 'mold', 'rot', 'spot', 'cercospora', 'septoria', 'anthracnose', 'fung']):
        return CAUSAS_AMBIENTALES['hongo']
    if any(x in c for x in ['bacterial', 'xanthomonas']):
        return CAUSAS_AMBIENTALES['bacteria']
    if any(x in c for x in ['mosaic', 'virus', 'curl', 'yellow']):
        return CAUSAS_AMBIENTALES['virus']
    if any(x in c for x in ['mite', 'spider', 'insect', 'plaga']):
        return CAUSAS_AMBIENTALES['plaga']
    return CAUSAS_AMBIENTALES['hongo']


def get_weather_data(lat=-12.065, lon=-75.212):
    api_key = os.environ.get('OPENWEATHER_API_KEY', '')
    if api_key:
        try:
            url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=es'
            req = Request(url, headers={'User-Agent': 'CelesteAI/1.0'})
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return _parse_weather(data)
        except Exception:
            pass

    return {
        'temp': 18, 'feels_like': 17, 'humidity': 68, 'wind_speed': 12,
        'wind_dir': 'NE', 'pressure': 1015, 'description': 'Nublado',
        'icon': '⛅', 'rain_prob': 20, 'visibility': 10, 'uv_index': 3,
        'location': 'Huancayo, Junín, Perú', 'lat': lat, 'lon': lon
    }


def get_forecast_data(lat=-12.065, lon=-75.212):
    api_key = os.environ.get('OPENWEATHER_API_KEY', '')
    if api_key:
        try:
            url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=es'
            req = Request(url, headers={'User-Agent': 'CelesteAI/1.0'})
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return _parse_forecast(data)
        except Exception:
            pass

    days = ['Hoy', 'Mañana', 'Vie', 'Sáb', 'Dom', 'Lun', 'Mar']
    return [
        {'day': days[i], 'icon': ['⛅', '🌤️', '🌧️', '☁️', '⛅', '☀️', '🌤️'][i],
         'temp_high': [18, 20, 16, 17, 19, 22, 21][i],
         'temp_low': [8, 9, 7, 8, 10, 11, 10][i],
         'rain_prob': [20, 10, 70, 40, 15, 5, 10][i]}
        for i in range(7)
    ]


def _parse_weather(data):
    desc = data.get('weather', [{}])[0].get('description', '').capitalize()
    icon_map = {
        'clear': '☀️', 'clouds': '☁️', 'rain': '🌧️', 'drizzle': '🌦️',
        'thunderstorm': '⛈️', 'snow': '❄️', 'mist': '🌫️', 'haze': '🌫️'
    }
    main = data.get('weather', [{}])[0].get('main', '').lower()
    icon = icon_map.get(main, '⛅')
    return {
        'temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'humidity': data['main']['humidity'],
        'wind_speed': round(data['wind']['speed']),
        'wind_dir': _deg_to_compass(data['wind'].get('deg', 0)),
        'pressure': data['main']['pressure'],
        'description': desc,
        'icon': icon,
        'rain_prob': data.get('rain', {}).get('1h', 0) if 'rain' in data else 0,
        'visibility': round(data.get('visibility', 10000) / 1000, 1),
        'uv_index': 3,
        'location': f"{data.get('name', 'Huancayo')}, Junín, Perú",
        'lat': data['coord']['lat'],
        'lon': data['coord']['lon']
    }


def _parse_forecast(data):
    from datetime import datetime
    daily = {}
    for item in data.get('list', []):
        dt = datetime.fromtimestamp(item['dt'])
        day_key = dt.strftime('%Y-%m-%d')
        if day_key not in daily:
            daily[day_key] = {'temps': [], 'icons': [], 'rain': 0, 'count': 0}
        daily[day_key]['temps'].append(item['main']['temp'])
        daily[day_key]['icons'].append(item['weather'][0]['main'].lower())
        daily[day_key]['rain'] += item.get('pop', 0) * 100
        daily[day_key]['count'] += 1

    from datetime import timedelta
    today = datetime.now()
    days = [(today + timedelta(days=i)).strftime('%A') for i in range(7)]
    day_names = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb']
    icon_map = {
        'clear': '☀️', 'clouds': '☁️', 'rain': '🌧️', 'drizzle': '🌦️',
        'thunderstorm': '⛈️', 'snow': '❄️', 'mist': '🌫️'
    }

    result = []
    for i, dk in enumerate(sorted(daily.keys())[:7]):
        dd = daily[dk]
        avg_icon = max(set(dd['icons']), key=dd['icons'].count) if dd['icons'] else 'clouds'
        day_label = 'Hoy' if i == 0 else ('Mañana' if i == 1 else day_names[datetime.strptime(dk, '%Y-%m-%d').weekday()])
        result.append({
            'day': day_label,
            'icon': icon_map.get(avg_icon, '⛅'),
            'temp_high': round(max(dd['temps'])),
            'temp_low': round(min(dd['temps'])),
            'rain_prob': round(dd['rain'] / dd['count']) if dd['count'] else 0
        })
    return result


def _deg_to_compass(deg):
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return dirs[round(deg / 22.5) % 16]
