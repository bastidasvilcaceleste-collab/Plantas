from ai_models.predictor import predict_disease
from services.disease_library import buscar_por_clave
from services.clima_service import analizar_causas
from services.recomendacion_service import (
    generar_recomendaciones,
    generar_recomendaciones_riego,
    generar_recomendaciones_fertilizante
)


def diagnostico_completo(img_file):
    prediccion = predict_disease(img_file)
    principal = prediccion['principal']
    alternativas = prediccion['alternativas']

    clase = principal['clase']
    es_sana = principal['es_sana']

    info_enfermedad = buscar_por_clave(clase)

    causas = analizar_causas(clase, es_sana)

    recomendaciones = generar_recomendaciones(clase, es_sana)
    riego = generar_recomendaciones_riego(clase, es_sana)
    fertilizante = generar_recomendaciones_fertilizante(clase, es_sana)

    resultado = {
        'principal': principal,
        'alternativas': alternativas,
        'causas': causas,
        'recomendaciones': recomendaciones,
        'riego': riego,
        'fertilizante': fertilizante,
    }

    if info_enfermedad:
        resultado['enfermedad'] = {
            'nombre': info_enfermedad['nombre'],
            'cultivo': info_enfermedad['cultivo'],
            'patogeno': info_enfermedad['patogeno'],
            'nombre_cientifico': info_enfermedad['nombre_cientifico'],
            'descripcion': info_enfermedad['descripcion'],
            'sintomas': info_enfermedad['sintomas'],
            'tratamiento': info_enfermedad['tratamiento'],
            'prevencion': info_enfermedad['prevencion'],
            'icono': info_enfermedad['icono'],
        }

    return resultado
