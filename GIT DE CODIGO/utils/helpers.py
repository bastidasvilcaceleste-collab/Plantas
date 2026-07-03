import re


def formatear_clase(clase):
    if not clase:
        return 'Desconocido'
    nombre = clase.replace('_', ' ')
    nombre = re.sub(r'\s+', ' ', nombre).strip()
    return nombre


def obtener_icono_por_clase(clase):
    if not clase:
        return '🌿'
    if clase == 'Resultado_poco_confiable':
        return '❓'
    c = clase.lower()
    iconos = {
        'apple': '🍎', 'blueberry': '🫐', 'cherry': '🍒',
        'maize': '🌽', 'corn': '🌽', 'grape': '🍇',
        'orange': '🍊', 'citrus': '🍊', 'peach': '🍑',
        'bellpepper': '🌶️', 'pepper': '🌶️', 'potato': '🥔', 'raspberry': '🍇',
        'soybean': '🫘', 'squash': '🎃', 'strawberry': '🍓',
        'tomato': '🍅', 'banana': '🍌', 'carrot': '🥕',
        'cassava': '🌿', 'chili': '🌶️', 'coffee': '☕',
        'cucumber': '🥒', 'guava': '🍈', 'mango': '🥭',
        'rice': '🌾', 'tea': '🍵', 'wheat': '🌾',
        'sugarcane': '🎋', 'pomegranate': '🍅'
    }
    for key, icono in iconos.items():
        if key in c:
            return icono
    return '🌿'
