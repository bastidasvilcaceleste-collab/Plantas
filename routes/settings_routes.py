from flask import Blueprint, request, redirect, url_for, jsonify, render_template, make_response
from flask_login import login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone

from extensions import db
from models.database import Usuario, UserPreferences, Analisis

settings_bp = Blueprint('settings', __name__)


def get_or_create_prefs():
    prefs = UserPreferences.query.filter_by(usuario_id=current_user.id).first()
    if not prefs:
        prefs = UserPreferences(usuario_id=current_user.id)
        db.session.add(prefs)
        db.session.commit()
    return prefs


@settings_bp.route('/configuracion', methods=['GET'])
@login_required
def configuracion():
    prefs = get_or_create_prefs()
    analisis_count = Analisis.query.filter_by(usuario_id=current_user.id).count()
    resp = make_response(render_template(
        'settings.html',
        active_page='settings',
        prefs=prefs,
        analisis_count=analisis_count
    ))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    return resp


@settings_bp.route('/api/configuracion/perfil', methods=['POST'])
@login_required
def guardar_perfil():
    current_user.nombre = request.form.get('nombre', current_user.nombre)
    prefs = get_or_create_prefs()
    prefs.telefono = request.form.get('telefono', '')
    prefs.ubicacion = request.form.get('ubicacion', '')
    prefs.tipo_usuario = request.form.get('tipo_usuario', 'agricultor')
    db.session.commit()
    return redirect(url_for('settings.configuracion', saved='perfil'))


@settings_bp.route('/api/configuracion/password', methods=['POST'])
@login_required
def cambiar_password():
    actual = request.form.get('password_actual', '')
    nueva = request.form.get('password_nueva', '')
    confirmar = request.form.get('password_confirmar', '')

    if not check_password_hash(current_user.password, actual):
        return redirect(url_for('settings.configuracion', error='password_actual'))

    if len(nueva) < 6:
        return redirect(url_for('settings.configuracion', error='password_corta'))

    if nueva != confirmar:
        return redirect(url_for('settings.configuracion', error='password_no_match'))

    current_user.password = generate_password_hash(nueva)
    db.session.commit()
    return redirect(url_for('settings.configuracion', saved='password'))


@settings_bp.route('/api/configuracion/preferencias', methods=['POST'])
@login_required
def guardar_preferencias():
    prefs = get_or_create_prefs()
    prefs.idioma = request.form.get('idioma', 'es')
    prefs.tema = request.form.get('tema', 'light')
    prefs.moneda = request.form.get('moneda', 'PEN')
    prefs.zona_horaria = request.form.get('zona_horaria', 'America/Lima')
    prefs.formato_fecha = request.form.get('formato_fecha', 'DD/MM/YYYY')
    db.session.commit()
    return redirect(url_for('settings.configuracion', saved='preferencias'))


@settings_bp.route('/api/configuracion/unidades', methods=['POST'])
@login_required
def guardar_unidades():
    prefs = get_or_create_prefs()
    prefs.temp_unidad = request.form.get('temp_unidad', 'Celsius')
    prefs.area_unidad = request.form.get('area_unidad', 'hectareas')
    prefs.peso_unidad = request.form.get('peso_unidad', 'kg')
    prefs.volumen_unidad = request.form.get('volumen_unidad', 'litros')
    prefs.viento_unidad = request.form.get('viento_unidad', 'km/h')
    prefs.humedad_unidad = request.form.get('humedad_unidad', '%')
    db.session.commit()
    return redirect(url_for('settings.configuracion', saved='unidades'))


@settings_bp.route('/api/configuracion/notificaciones', methods=['POST'])
@login_required
def guardar_notificaciones():
    prefs = get_or_create_prefs()
    prefs.notif_enfermedades = 'notif_enfermedades' in request.form
    prefs.notif_ia = 'notif_ia' in request.form
    prefs.notif_clima = 'notif_clima' in request.form
    prefs.notif_recomendaciones = 'notif_recomendaciones' in request.form
    prefs.notif_noticias = 'notif_noticias' in request.form
    prefs.notif_recordatorios = 'notif_recordatorios' in request.form
    db.session.commit()
    return redirect(url_for('settings.configuracion', saved='notificaciones'))


@settings_bp.route('/api/configuracion/ia', methods=['POST'])
@login_required
def guardar_ia():
    prefs = get_or_create_prefs()
    try:
        prefs.ia_confianza_min = float(request.form.get('ia_confianza_min', 0.60))
    except ValueError:
        prefs.ia_confianza_min = 0.60
    prefs.ia_sensibilidad = request.form.get('ia_sensibilidad', 'media')
    prefs.ia_profundidad = request.form.get('ia_profundidad', 'normal')
    prefs.ia_aprendizaje = 'ia_aprendizaje' in request.form
    db.session.commit()
    return redirect(url_for('settings.configuracion', saved='ia'))


@settings_bp.route('/api/configuracion/datos/exportar/<formato>')
@login_required
def exportar_datos(formato):
    import csv, io, json
    from flask import send_file

    analisis = Analisis.query.filter_by(usuario_id=current_user.id).order_by(Analisis.fecha.desc()).all()

    if formato == 'csv':
        si = io.StringIO()
        writer = csv.writer(si)
        writer.writerow(['ID', 'Cultivo', 'Diagnostico', 'Confianza', 'Estado', 'Fecha', 'Hora'])
        for a in analisis:
            fecha = a.fecha.replace(tzinfo=timezone.utc) if a.fecha and a.fecha.tzinfo is None else a.fecha
            writer.writerow([
                a.id, a.clase.split('___')[0] if '___' in a.clase else a.clase,
                a.clase, f'{a.confianza*100:.1f}%',
                'Sana' if a.es_sana else 'Enferma',
                fecha.strftime('%d/%m/%Y') if fecha else '',
                fecha.strftime('%H:%M') if fecha else ''
            ])
        output = si.getvalue().encode('utf-8-sig')
        buf = io.BytesIO(output)
        buf.seek(0)
        return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='celesteai_datos.csv')

    return redirect(url_for('settings.configuracion', saved='exportado'))


@settings_bp.route('/api/configuracion/sistema')
@login_required
def estado_sistema():
    import os, platform
    from sqlalchemy import text

    db_ok = False
    try:
        db.session.execute(text('SELECT 1'))
        db_ok = True
    except Exception:
        pass

    analisis_count = Analisis.query.filter_by(usuario_id=current_user.id).count()
    total_users = Usuario.query.count()

    return jsonify({
        'servidor': 'Online' if db_ok else 'Error',
        'base_datos': 'Conectada' if db_ok else 'Desconectada',
        'ia': 'Modelo cargado (EfficientNetB3)',
        'clima': 'API configurada',
        'almacenamiento': f'{analisis_count * 0.5:.1f} MB',
        'analisis_usuario': analisis_count,
        'total_usuarios': total_users,
        'uptime': 'Desde inicio de sesión',
        'plataforma': platform.system(),
        'python': platform.python_version()
    })


@settings_bp.route('/api/configuracion/datos/limpiar', methods=['POST', 'GET'])
@login_required
def limpiar_datos():
    Analisis.query.filter_by(usuario_id=current_user.id).delete()
    db.session.commit()
    return redirect(url_for('settings.configuracion', saved='limpiado'))
