def generar_recomendaciones(clase, es_sana):
    if es_sana:
        return {
            'tipo': 'mantenimiento',
            'titulo': 'Recomendaciones de mantenimiento',
            'recomendaciones': [
                {'icono': '💧', 'accion': 'Riego moderado',
                 'detalle': 'Mantén el suelo húmedo pero no encharcado. Riega en las mañanas.'},
                {'icono': '🌿', 'accion': 'Fertilización balanceada',
                 'detalle': 'Aplica fertilizante NPK cada 15 días durante la temporada de crecimiento.'},
                {'icono': '✂️', 'accion': 'Poda de formación',
                 'detalle': 'Elimina hojas secas o dañadas para mantener la planta vigorosa.'},
                {'icono': '🔍', 'accion': 'Monitoreo semanal',
                 'detalle': 'Inspecciona tus plantas cada semana para detectar signos tempranos de enfermedades.'}
            ]
        }

    if clase == 'Resultado_poco_confiable':
        return {
            'tipo': 'no_confiable',
            'titulo': 'Recomendaciones generales',
            'recomendaciones': [
                {'icono': '📸', 'accion': 'Toma una nueva foto',
                 'detalle': 'Con mejor iluminación y fondo uniforme para obtener un diagnóstico preciso.'},
                {'icono': '💡', 'accion': 'Mejora la iluminación',
                 'detalle': 'Fotografía la hoja con luz natural difusa, evitando sombras.'},
                {'icono': '🌱', 'accion': 'Consulta a un experto',
                 'detalle': 'Si los síntomas persisten, contacta a un agrónomo local.'}
            ]
        }

    clave = clase.lower()
    recom = []

    if 'hongo' in clave or 'mildew' in clave or 'rust' in clave or 'blight' in clave or 'scab' in clave or 'rot' in clave or 'spot' in clave or 'mold' in clave or 'scorch' in clave:
        recom.append({'icono': '🧪', 'accion': 'Aplicar fungicida',
                      'detalle': 'Usa fungicida a base de cobre o azufre. Aplica cada 7-14 días según severidad.'})
        recom.append({'icono': '💨', 'accion': 'Mejorar ventilación',
                      'detalle': 'Poda ramas para mejorar la circulación de aire entre las plantas.'})
        recom.append({'icono': '💧', 'accion': 'Reducir humedad foliar',
                      'detalle': 'Cambia a riego por goteo. Evita mojar las hojas durante el riego.'})
        recom.append({'icono': '🍂', 'accion': 'Eliminar residuos',
                      'detalle': 'Retira hojas caídas y restos de cosecha para reducir la propagación.'})

    if 'bacteria' in clave or 'bacterial' in clave or 'hlb' in clave or 'greening' in clave:
        recom.append({'icono': '🧪', 'accion': 'Aplicar bactericida',
                      'detalle': 'Usa productos a base de cobre. Aplica en otoño e invierno.'})
        recom.append({'icono': '✂️', 'accion': 'Poda sanitaria',
                      'detalle': 'Elimina ramas infectadas. Desinfecta herramientas entre cortes.'})
        recom.append({'icono': '🔄', 'accion': 'Rotación de cultivos',
                      'detalle': 'No siembres la misma familia botánica por 3 años en el mismo lugar.'})
        recom.append({'icono': '🌱', 'accion': 'Usar semilla certificada',
                      'detalle': 'Adquiere semillas libres de patógenos de proveedores certificados.'})

    if 'virus' in clave or 'mosaic' in clave or 'curl' in clave:
        recom.append({'icono': '🪰', 'accion': 'Controlar insectos vectores',
                      'detalle': 'Usa mallas anti-insectos. Controla mosca blanca y pulgones.'})
        recom.append({'icono': '🗑️', 'accion': 'Eliminar plantas infectadas',
                      'detalle': 'Retira y destruye las plantas enfermas para evitar la propagación.'})
        recom.append({'icono': '🧴', 'accion': 'Desinfectar herramientas',
                      'detalle': 'Limpia tijeras y herramientas con lejía diluida (10%) entre usos.'})
        recom.append({'icono': '🧤', 'accion': 'Lavado de manos',
                      'detalle': 'Lava tus manos antes y después de manipular plantas.'})

    if 'mite' in clave or 'spider' in clave or 'insect' in clave:
        recom.append({'icono': '🧪', 'accion': 'Aplicar acaricida',
                      'detalle': 'Usa jabón potásico o acaricida específico. Aplica en horas frescas.'})
        recom.append({'icono': '🐞', 'accion': 'Control biológico',
                      'detalle': 'Introduce depredadores naturales como Phytoseiulus persimilis.'})
        recom.append({'icono': '💨', 'accion': 'Aumentar humedad ambiental',
                      'detalle': 'Las arañas rojas prosperan en ambientes secos. Incrementa la humedad.'})

    if not recom:
        recom.append({'icono': '🧪', 'accion': 'Consulta técnica',
                      'detalle': 'Se recomienda consultar con un ingeniero agrónomo para un plan específico.'})
        recom.append({'icono': '🔄', 'accion': 'Rotación de cultivos',
                      'detalle': 'Implementa rotación para evitar la recurrencia de enfermedades.'})

    return {
        'tipo': 'tratamiento',
        'titulo': 'Recomendaciones de tratamiento',
        'recomendaciones': recom
    }


def generar_recomendaciones_riego(clase, es_sana):
    if es_sana:
        return {'frecuencia': 'Cada 2-3 días',
                'metodo': 'Riego por goteo',
                'horario': 'Mañana (6-9 AM)',
                'cantidad': '2-3 litros por planta'}
    clave = clase.lower()
    if any(s in clave for s in ['hongo', 'mildew', 'blight', 'rot', 'scab', 'mold']):
        return {'frecuencia': 'Reducir frecuencia',
                'metodo': 'Riego por goteo estricto',
                'horario': 'Mañana temprano',
                'cantidad': '1-2 litros por planta',
                'nota': 'Evita totalmente mojar el follaje. Riega solo la base.'}
    if any(s in clave for s in ['bacteria', 'bacterial', 'spot']):
        return {'frecuencia': 'Moderada',
                'metodo': 'Riego por goteo',
                'horario': 'Mañana',
                'cantidad': '2 litros por planta',
                'nota': 'Evita salpicaduras de suelo a hojas.'}
    return {'frecuencia': 'Cada 2 días',
            'metodo': 'Riego por goteo',
            'horario': 'Mañana',
            'cantidad': '2 litros por planta'}


def generar_recomendaciones_fertilizante(clase, es_sana):
    if es_sana:
        return {'tipo': 'NPK balanceado (10-10-10)',
                'frecuencia': 'Cada 15 días',
                'dosis': 'Seguir instrucciones del fabricante'}
    clave = clase.lower()
    if any(s in clave for s in ['hongo', 'mildew', 'blight', 'rot']):
        return {'tipo': 'NPK bajo en nitrógeno (5-10-10)',
                'frecuencia': 'Cada 20 días',
                'dosis': 'Reducir a 70% de la dosis normal',
                'nota': 'El exceso de nitrógeno favorece infecciones fúngicas.'}
    if any(s in clave for s in ['bacteria', 'bacterial']):
        return {'tipo': 'NPK balanceado (10-10-10)',
                'frecuencia': 'Cada 15 días',
                'dosis': 'Dosis normal',
                'nota': 'Añade potasio para fortalecer paredes celulares.'}
    if any(s in clave for s in ['virus', 'mosaic', 'curl']):
        return {'tipo': 'NPK con micronutrientes',
                'frecuencia': 'Cada 10 días',
                'dosis': 'Dosis normal',
                'nota': 'Fortalece la planta para tolerar mejor la infección viral.'}
    return {'tipo': 'NPK balanceado (10-10-10)',
            'frecuencia': 'Cada 15 días',
            'dosis': 'Seguir instrucciones del fabricante'}
