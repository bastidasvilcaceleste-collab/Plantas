import base64
import sys
import os

from flask import Blueprint, request, redirect, url_for, jsonify, render_template, make_response
from flask_login import login_required, current_user

from extensions import db
from models.database import Analisis

# Asegurar que podemos importar desde ai_models
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ai_models'))

# IMPORTANTE: Usamos nuestro predictor directamente
from predictor import predict_disease

from services.historial_service import (
    guardar_analisis,
    obtener_historial,
    obtener_estadisticas
)
from services.disease_library import get_cultivos, filtrar
from services.guias_service import get_guias, get_categorias

analisis_bp = Blueprint('analisis', __name__)

CROP_DATA = {
    'Apple': {'icon': '🍎', 'name': 'Manzano', 'sci': 'Malus domestica'},
    'Blueberry': {'icon': '🫐', 'name': 'Arándano', 'sci': 'Vaccinium corymbosum'},
    'Cherry': {'icon': '🍒', 'name': 'Cerezo', 'sci': 'Prunus avium'},
    'Corn': {'icon': '🌽', 'name': 'Maíz', 'sci': 'Zea mays'},
    'Grape': {'icon': '🍇', 'name': 'Vid', 'sci': 'Vitis vinifera'},
    'Orange': {'icon': '🍊', 'name': 'Naranjo', 'sci': 'Citrus sinensis'},
    'Peach': {'icon': '🍑', 'name': 'Durazno', 'sci': 'Prunus persica'},
    'Pepper': {'icon': '🌶️', 'name': 'Pimiento', 'sci': 'Capsicum annuum'},
    'Potato': {'icon': '🥔', 'name': 'Papa', 'sci': 'Solanum tuberosum'},
    'Raspberry': {'icon': '🫐', 'name': 'Frambuesa', 'sci': 'Rubus idaeus'},
    'Soybean': {'icon': '🫘', 'name': 'Soya', 'sci': 'Glycine max'},
    'Squash': {'icon': '🎃', 'name': 'Calabaza', 'sci': 'Cucurbita pepo'},
    'Strawberry': {'icon': '🍓', 'name': 'Fresa', 'sci': 'Fragaria × ananassa'},
    'Tomato': {'icon': '🍅', 'name': 'Tomate', 'sci': 'Solanum lycopersicum'},
}

DISEASE_SCI = {
    'Apple_scab': 'Venturia inaequalis',
    'Black_rot': 'Botryosphaeria obtusa',
    'Cedar_apple_rust': 'Gymnosporangium juniperi-virginianae',
    'Powdery_mildew': 'Erysiphales spp.',
    'Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora zeae-maydis',
    'Common_rust_': 'Puccinia sorghi',
    'Northern_Leaf_Blight': 'Exserohilum turcicum',
    'Esca_(Black_Measles)': 'Phaeoacremonium spp.',
    'Leaf_blight_(Isariopsis_Leaf_Spot)': 'Isariopsis griseola',
    'Haunglongbing_(Citrus_greening)': 'Candidatus Liberibacter spp.',
    'Bacterial_spot': 'Xanthomonas spp.',
    'Early_blight': 'Alternaria solani',
    'Late_blight': 'Phytophthora infestans',
    'Leaf_scorch': 'Diplocarpon earlianum',
    'Leaf_Mold': 'Passalora fulva',
    'Septoria_leaf_spot': 'Septoria lycopersici',
    'Spider_mites Two-spotted_spider_mite': 'Tetranychus urticae',
    'Target_Spot': 'Corynespora cassiicola',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Begomovirus (TYLCV)',
    'Tomato_mosaic_virus': 'Tobamovirus (ToMV)',
}


# ── FILTRO POR CULTIVO ─────────────────────────────────────────────

CULTIVO_PREFIX = {
    'papa': ['Potato_'],
    'tomate': ['Tomato_'],
    'maiz': ['Corn_'],
    'manzano': ['Apple_'],
    'cerezo': ['Cherry_'],
    'vid': ['Grape_'],
    'naranjo': ['Orange_'],
    'durazno': ['Peach_'],
    'pimiento': ['Pepper_Bell_', 'Bellpepper_'],
    'fresa': ['Strawberry_'],
    'soya': ['Soybean_'],
}


def _filtrar_por_cultivo(resultado, cultivo):
    prefixes = CULTIVO_PREFIX.get(cultivo)
    if not prefixes:
        return resultado

    def match_clase(item):
        return any(item['clase'].startswith(p) for p in prefixes)

    principal = resultado['principal']
    alternativas = resultado['alternativas']

    if not match_clase(principal):
        todas = [principal] + alternativas
        coinciden = [t for t in todas if match_clase(t)]
        if not coinciden:
            resultado['principal'] = {
                'clase': 'Resultado_poco_confiable',
                'confianza': 0.0,
                'es_sana': False
            }
            resultado['alternativas'] = []
            return resultado
        resultado['principal'] = coinciden[0]
        resultado['alternativas'] = coinciden[1:5]
    else:
        resultado['alternativas'] = [a for a in alternativas if match_clase(a)][:4]

    return resultado


# ── PREDICCIÓN (VERSIÓN CORREGIDA) ─────────────────────────────────

@analisis_bp.route('/predecir', methods=['POST'])
def predecir():
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió imagen'}), 400

    archivo = request.files['image']
    cultivo = request.form.get('cultivo', '').strip().lower()

    img_bytes = archivo.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    img_data_uri = f"data:{archivo.content_type or 'image/jpeg'};base64,{img_b64}"
    archivo.seek(0)

    # === LLAMADA AL PREDICTOR ===
    try:
        # Usamos directamente predict_disease
        resultado = predict_disease(archivo)
        
        # === LOGS DE DEPURACIÓN ===
        print("\n" + "="*50)
        print("🔍 DIAGNÓSTICO COMPLETO:")
        print(f"   Clase principal: {resultado['principal']['clase']}")
        print(f"   Confianza: {resultado['principal']['confianza']}")
        print(f"   ¿Es sana? {resultado['principal'].get('es_sana', False)}")
        print(f"   Nivel: {resultado['principal'].get('nivel', 'no definido')}")
        if 'segunda' in resultado and resultado['segunda']:
            print(f"   Alternativa 2: {resultado['segunda']['clase']} ({resultado['segunda']['confianza']})")
        print("="*50 + "\n")
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Error interno del predictor: {str(e)}'}), 500

    # Filtrar por cultivo si se especificó
    if cultivo:
        resultado = _filtrar_por_cultivo(resultado, cultivo)

    principal = resultado['principal']

    # Guardar en base de datos si el usuario está autenticado
    if current_user.is_authenticated:
        guardar_analisis(
            usuario_id=current_user.id,
            clase=principal['clase'],
            confianza=principal['confianza'],
            es_sana=principal.get('es_sana', False),
            imagen_b64=img_data_uri
        )

    # Construir respuesta
    payload = {
        'principal': principal,
        'alternativas': resultado.get('alternativas', []),
        'imagen': img_data_uri,
    }

    # Si es petición AJAX, devolver JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(payload)

    # Si no, redirigir a la página de resultado con parámetros
    # IMPORTANTE: Pasamos la confianza como porcentaje para facilitar el template
    confianza_porcentaje = round(principal['confianza'] * 100, 1)
    
    params = {
        'clase': principal['clase'],
        'confianza': confianza_porcentaje,  # Ya en porcentaje (0-100)
        'confianza_raw': principal['confianza'],  # También pasamos el valor raw por si acaso
        'es_sana': str(principal.get('es_sana', False)).lower(),
        'imagen': img_data_uri
    }
    return redirect(url_for('analisis.resultado', **params))


# ── RESULTADO ──────────────────────────────────────────────

@analisis_bp.route('/resultado')
def resultado():
    clase = request.args.get('clase', 'Desconocido')
    confianza = request.args.get('confianza', 0, type=float)  # Ya viene como porcentaje
    confianza_raw = request.args.get('confianza_raw', 0, type=float)  # Valor raw 0-1
    es_sana = request.args.get('es_sana', 'false').lower() == 'true'
    imagen = request.args.get('imagen', '')
    
    # Si no vino confianza_raw, la calculamos a partir de confianza (que es porcentaje)
    if confianza_raw == 0 and confianza > 0:
        confianza_raw = confianza / 100.0
    
    print(f"📊 Renderizando resultado: clase={clase}, confianza={confianza}%, raw={confianza_raw}")
    
    return render_template('resultado.html',
                           clase=clase,
                           confianza=confianza,  # Porcentaje para mostrar
                           confianza_raw=confianza_raw,  # Valor raw 0-1 para progreso
                           es_sana=es_sana,
                           imagen=imagen)


# ── DETALLE ──────────────────────────────────────────────

@analisis_bp.route('/detalle/<int:analisis_id>')
@login_required
def detalle(analisis_id):
    item = Analisis.query.filter_by(
        id=analisis_id, usuario_id=current_user.id
    ).first_or_404()
    
    # Pasamos la confianza como porcentaje
    confianza_porcentaje = round(item.confianza * 100, 1) if item.confianza else 0
    
    return redirect(url_for('analisis.resultado',
                            clase=item.clase,
                            confianza=confianza_porcentaje,
                            confianza_raw=item.confianza,
                            es_sana=str(item.es_sana).lower(),
                            imagen=item.imagen or ''))


# ── DASHBOARD ──────────────────────────────────────────────

@analisis_bp.route('/dashboard')
@login_required
def dashboard():
    historial = obtener_historial(current_user.id, limite=3)
    stats = obtener_estadisticas(current_user.id)
    resp = make_response(render_template(
        'dashboard.html',
        historial=historial,
        active_page='dashboard',
        **stats
    ))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@analisis_bp.route('/analizar')
@login_required
def analizar():
    historial = obtener_historial(current_user.id, limite=5)
    resp = make_response(render_template('analisis.html', active_page='analizar', historial=historial))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@analisis_bp.route('/historial')
@login_required
def historial():
    items = obtener_historial(current_user.id)
    stats = obtener_estadisticas(current_user.id)

    from datetime import datetime
    now = datetime.utcnow()
    items_mes = [i for i in items if i.fecha and i.fecha.month == now.month and i.fecha.year == now.year]
    total_mes = len(items_mes)
    sanas_mes = len([i for i in items_mes if i.es_sana])

    if items:
        precision_promedio = sum(i.confianza for i in items) / len(items)
        precision_mes = sum(i.confianza for i in items_mes) / len(items_mes) if items_mes else 0
    else:
        precision_promedio = 0
        precision_mes = 0

    resp = make_response(render_template(
        'historial.html',
        historial=items,
        active_page='historial',
        crop_data=CROP_DATA,
        disease_sci=DISEASE_SCI,
        **stats,
        total_mes=total_mes,
        sanas_mes=sanas_mes,
        enfermas_mes=total_mes - sanas_mes,
        precision_promedio=precision_promedio,
        precision_mes=precision_mes
    ))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


# ── BIBLIOTECA ──────────────────────────────────────────────

@analisis_bp.route('/biblioteca')
@login_required
def biblioteca():
    resp = make_response(render_template(
        'biblioteca.html', categorias=get_categorias(), active_page='biblioteca'
    ))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, private'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


# ── APIS ──────────────────────────────────────────────

@analisis_bp.route('/api/enfermedades')
@login_required
def api_enfermedades():
    cultivo = request.args.get('cultivo')
    patogeno = request.args.get('patogeno')
    busqueda = request.args.get('busqueda')
    resultados = filtrar(cultivo=cultivo, patogeno=patogeno, busqueda=busqueda)
    return jsonify(resultados)


@analisis_bp.route('/api/guias')
@login_required
def api_guias():
    categoria = request.args.get('categoria')
    busqueda = request.args.get('busqueda')
    resultados = get_guias(categoria=categoria, busqueda=busqueda)
    return jsonify(resultados)


# ── FERTILIZANTES ──────────────────────────────────────────

@analisis_bp.route('/fertilizantes')
@login_required
def fertilizantes():
    return render_template('fertilizantes.html', active_page='fertilizantes')


# ── CLIMA ──────────────────────────────────────────────────

@analisis_bp.route('/clima')
@login_required
def clima():
    from services.clima_service import get_weather_data, get_forecast_data
    weather = get_weather_data()
    forecast = get_forecast_data()
    return render_template('clima.html', active_page='clima', weather=weather, forecast=forecast)


# ── METODOLOGÍA ────────────────────────────────────────────

@analisis_bp.route('/metodologia')
@login_required
def metodologia():
    return render_template('metodologia.html', active_page='metodologia')


# ── REPORTES ───────────────────────────────────────────────

@analisis_bp.route('/reportes')
@login_required
def reportes():
    from services.historial_service import obtener_historial, obtener_estadisticas
    from datetime import datetime, timezone
    historial = obtener_historial(current_user.id, limite=1000)
    stats = obtener_estadisticas(current_user.id)

    total = stats.get('total', 0)
    sanas = stats.get('sanas', 0)
    enfermas = stats.get('enfermas', 0)

    now = datetime.now(timezone.utc)
    items_mes = [i for i in historial if i.fecha and i.fecha.month == now.month and i.fecha.year == now.year] if historial else []
    total_mes = len(items_mes)

    precision_promedio = (sum(i.confianza for i in historial) / len(historial)) if historial else 0

    crop_counts = {}
    for item in historial:
        crop = item.clase.split('_')[0] if item.clase else 'Otros'
        crop_counts[crop] = crop_counts.get(crop, 0) + 1

    sorted_crops = sorted(crop_counts.items(), key=lambda x: -x[1])

    monthly_data = {}
    for item in historial:
        if item.fecha:
            key = item.fecha.strftime('%Y-%m')
            monthly_data[key] = monthly_data.get(key, 0) + 1

    sorted_months = sorted(monthly_data.keys())

    return render_template('reportes.html', historial=historial, active_page='reportes',
                           **stats, total_mes=total_mes, precision_promedio=precision_promedio,
                           crop_counts=sorted_crops, monthly_data=monthly_data,
                           sorted_months=sorted_months)


@analisis_bp.route('/api/crops', methods=['GET'])
@login_required
def list_crops():
    from models.database import UserCrop
    crops = UserCrop.query.filter_by(usuario_id=current_user.id).order_by(UserCrop.created_at.desc()).all()
    return jsonify([{
        'id': c.id,
        'crop_name': c.crop_name,
        'status': c.status,
        'last_analysis': c.last_analysis.strftime('%d/%m/%Y %H:%M') if c.last_analysis else None,
        'created_at': c.created_at.strftime('%d/%m/%Y') if c.created_at else None
    } for c in crops])


@analisis_bp.route('/api/crops/add', methods=['POST'])
@login_required
def add_crop():
    from models.database import UserCrop
    data = request.get_json()
    if not data or not data.get('crop_name'):
        return jsonify({'error': 'Nombre de cultivo requerido'}), 400
    crop = UserCrop(
        usuario_id=current_user.id,
        crop_name=data['crop_name'],
        status='activo'
    )
    db.session.add(crop)
    db.session.commit()
    return jsonify({'success': True, 'id': crop.id})


@analisis_bp.route('/api/crops/<int:crop_id>/delete', methods=['POST'])
@login_required
def delete_crop(crop_id):
    from models.database import UserCrop
    crop = UserCrop.query.filter_by(id=crop_id, usuario_id=current_user.id).first_or_404()
    db.session.delete(crop)
    db.session.commit()
    return jsonify({'success': True})