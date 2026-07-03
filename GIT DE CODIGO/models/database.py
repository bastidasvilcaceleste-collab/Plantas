import json
from extensions import db
from flask_login import UserMixin
from datetime import datetime


class Usuario(UserMixin, db.Model):
    __tablename__ = 'usuario'

    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    fecha_registro = db.Column(db.DateTime, default=datetime.utcnow)
    analisis = db.relationship('Analisis', backref='usuario', lazy=True)
    preferencias = db.relationship('UserPreferences', uselist=False, backref='usuario', lazy=True)


class UserPreferences(db.Model):
    __tablename__ = 'user_preferences'

    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), unique=True, nullable=False)

    idioma = db.Column(db.String(10), default='es')
    tema = db.Column(db.String(10), default='light')
    moneda = db.Column(db.String(10), default='PEN')
    zona_horaria = db.Column(db.String(50), default='America/Lima')
    formato_fecha = db.Column(db.String(20), default='DD/MM/YYYY')

    temp_unidad = db.Column(db.String(10), default='Celsius')
    area_unidad = db.Column(db.String(20), default='hectareas')
    peso_unidad = db.Column(db.String(10), default='kg')
    volumen_unidad = db.Column(db.String(10), default='litros')
    viento_unidad = db.Column(db.String(10), default='km/h')
    humedad_unidad = db.Column(db.String(5), default='%')

    notif_enfermedades = db.Column(db.Boolean, default=True)
    notif_ia = db.Column(db.Boolean, default=True)
    notif_clima = db.Column(db.Boolean, default=True)
    notif_recomendaciones = db.Column(db.Boolean, default=True)
    notif_noticias = db.Column(db.Boolean, default=False)
    notif_recordatorios = db.Column(db.Boolean, default=True)

    ia_confianza_min = db.Column(db.Float, default=0.60)
    ia_sensibilidad = db.Column(db.String(10), default='media')
    ia_profundidad = db.Column(db.String(20), default='normal')
    ia_aprendizaje = db.Column(db.Boolean, default=True)

    telefono = db.Column(db.String(20), nullable=True)
    ubicacion = db.Column(db.String(100), nullable=True)
    tipo_usuario = db.Column(db.String(30), default='agricultor')

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Analisis(db.Model):
    __tablename__ = 'analisis'

    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=False)
    clase = db.Column(db.String(200), nullable=False)
    confianza = db.Column(db.Float, nullable=False)
    es_sana = db.Column(db.Boolean, nullable=False)
    fecha = db.Column(db.DateTime, default=datetime.utcnow)
    imagen = db.Column(db.Text, nullable=True)


class UserCrop(db.Model):
    __tablename__ = 'user_crops'

    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=False)
    crop_name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(50), default='activo')
    last_analysis = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    usuario = db.relationship('Usuario', backref=db.backref('crops', lazy=True))
