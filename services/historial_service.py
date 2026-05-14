from models.database import Analisis
from extensions import db
from datetime import datetime


def guardar_analisis(usuario_id, clase, confianza, es_sana, imagen_b64=None):
    analisis = Analisis(
        usuario_id=usuario_id,
        clase=clase,
        confianza=confianza,
        es_sana=es_sana,
        fecha=datetime.utcnow(),
        imagen=imagen_b64
    )
    db.session.add(analisis)
    db.session.commit()
    return analisis


def obtener_historial(usuario_id, limite=None):
    query = (Analisis.query
             .filter_by(usuario_id=usuario_id)
             .order_by(Analisis.fecha.desc()))
    if limite:
        query = query.limit(limite)
    return query.all()


def obtener_detalle(analisis_id, usuario_id):
    return (Analisis.query
            .filter_by(id=analisis_id, usuario_id=usuario_id)
            .first_or_404())


def obtener_estadisticas(usuario_id):
    total = Analisis.query.filter_by(usuario_id=usuario_id).count()
    sanas = Analisis.query.filter_by(usuario_id=usuario_id, es_sana=True).count()
    enfermas = total - sanas
    return {
        'total': total,
        'sanas': sanas,
        'enfermas': enfermas
    }
