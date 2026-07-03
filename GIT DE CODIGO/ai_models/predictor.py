"""
Predictor EfficientNetB0 — modelo entrenado con train_simple.py
Carga: final_modelo_plantas.pth + classes.json (19 clases)
"""
import json
import os
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE, 'final_modelo_plantas.pth')
CLASSES_PATH = os.path.join(BASE, 'classes.json')

IMG_SIZE = 224
TOP_K = 5
FORMATOS_VALIDOS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'}

# Umbrales solo para etiquetar nivel (no sustituyen la clase predicha)
UMBRAL_ALTA = 0.80
UMBRAL_MODERADA = 0.50

_model = None
_class_names = None

_inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _load_class_names():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            f'No se encuentra classes.json en {CLASSES_PATH}. '
            'Ejecuta train_simple.py primero.'
        )
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError('classes.json debe ser una lista no vacía de nombres de clase.')
    return names


def reload_metadata():
    """Recarga nombres de clase desde disco (útil tras reentrenar)."""
    global _class_names, _model, CLASS_NAMES, NUM_CLASSES
    _class_names = _load_class_names()
    CLASS_NAMES = _class_names
    NUM_CLASSES = len(_class_names)
    _model = None
    return _class_names


def get_class_names():
    global _class_names, CLASS_NAMES, NUM_CLASSES
    if _class_names is None:
        _class_names = _load_class_names()
        CLASS_NAMES = _class_names
        NUM_CLASSES = len(_class_names)
    return _class_names


CLASS_NAMES = []
NUM_CLASSES = 0


def _build_model(num_classes):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f'Modelo no encontrado en {MODEL_PATH}. '
                'Ejecuta: python train_simple.py'
            )
        names = get_class_names()
        num_classes = len(names)
        model = _build_model(num_classes)
        state_dict = torch.load(
            MODEL_PATH,
            map_location=torch.device('cpu'),
            weights_only=True,
        )
        out_features = state_dict['classifier.4.weight'].shape[0]
        if out_features != num_classes:
            raise RuntimeError(
                f'Incompatibilidad: el modelo tiene {out_features} clases '
                f'y classes.json tiene {num_classes}. '
                'Vuelve a entrenar o actualiza classes.json.'
            )
        model.load_state_dict(state_dict)
        model.eval()
        _model = model
    return _model


def _extraer_bytes(archivo):
    if isinstance(archivo, (bytes, bytearray)):
        return BytesIO(archivo)
    data = archivo.read()
    if hasattr(archivo, 'seek'):
        archivo.seek(0)
    return BytesIO(data)


def validar_imagen(archivo):
    if archivo is None:
        raise ValueError('No se proporcionó ninguna imagen.')
    if hasattr(archivo, 'filename') and archivo.filename:
        ext = archivo.filename.rsplit('.', 1)[-1].lower()
        if ext not in FORMATOS_VALIDOS:
            raise ValueError(
                f'Formato no soportado: .{ext}. '
                f'Usa: {", ".join(sorted(FORMATOS_VALIDOS))}.'
            )
    buf = _extraer_bytes(archivo)
    try:
        img = Image.open(buf)
        img.verify()
    except Exception:
        raise ValueError('El archivo no es una imagen válida o está corrupto.')


def _es_clase_sana(nombre_clase):
    return 'healthy' in nombre_clase.lower()


def _nivel_confianza(confianza):
    if confianza >= UMBRAL_ALTA:
        return 'alta'
    if confianza >= UMBRAL_MODERADA:
        return 'moderada'
    return 'preliminar'


def _item_prediccion(indice, probabilidad):
    clase = get_class_names()[indice]
    confianza = round(float(probabilidad), 4)
    return {
        'clase': clase,
        'confianza': confianza,
        'es_sana': _es_clase_sana(clase),
        'nivel': _nivel_confianza(confianza),
    }


def predict_disease(img_file):
    """
    Predicción top-1 con softmax real (sin temperatura).

    Returns (compatible con Flask):
        principal: mejor clase
        alternativas: top-2..top-K
        confianza: probabilidad del top-1
        segunda / tercera: compatibilidad legacy
    """
    validar_imagen(img_file)
    buf = _extraer_bytes(img_file)
    img = Image.open(buf).convert('RGB')
    tensor = _inference_transform(img).unsqueeze(0)

    model = get_model()
    names = get_class_names()

    with torch.no_grad():
        logits = model(tensor)
        probabilidades = F.softmax(logits[0], dim=0)

    k = min(TOP_K, len(names))
    top_conf, top_idx = torch.topk(probabilidades, k)

    predicciones = [
        _item_prediccion(top_idx[i].item(), top_conf[i].item())
        for i in range(k)
    ]

    principal = predicciones[0]

    return {
        'principal': principal,
        'confianza': principal['confianza'],
        'alternativas': predicciones[1:],
        'segunda': predicciones[1] if len(predicciones) > 1 else None,
        'tercera': predicciones[2] if len(predicciones) > 2 else None,
    }


# Inicializar clases al importar el módulo
get_class_names()


if __name__ == '__main__':
    import sys

    names = reload_metadata()
    print('=== PREDICTOR EfficientNetB0 ===')
    print(f'Clases en JSON: {len(names)}')
    print(f'Modelo: {MODEL_PATH}')
    print(f'Primeras clases: {names[:3]} ... {names[-1]}')

    model = get_model()
    print(f'Backbone: EfficientNetB0')
    print(f'Salida del modelo: {model.classifier[-1].out_features} neuronas')

    img_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(BASE),
            'data_simple', 'test', 'Tomato_Healthy',
            '09c0d78a-c9ca-4dc4-aa10-e25530890b20___GH_HL_Leaf_424.1.jpg',
        )
    )

    if os.path.isfile(img_path):
        with open(img_path, 'rb') as f:
            resultado = predict_disease(f)
        print(f'\nImagen de prueba: {img_path}')
        print(f'Clase esperada (carpeta): Tomato_Healthy')
        p = resultado['principal']
        print('\n--- Ejemplo de predicción ---')
        print(f'  principal.clase:     {p["clase"]}')
        print(f'  principal.confianza: {p["confianza"]} ({p["confianza"] * 100:.1f}%)')
        print(f'  principal.nivel:     {p["nivel"]}')
        print(f'  principal.es_sana:   {p["es_sana"]}')
        print(f'  confianza (top):     {resultado["confianza"]}')
        print('\n--- Alternativas ---')
        for i, alt in enumerate(resultado['alternativas'][:3], start=2):
            print(f'  top-{i}: {alt["clase"]} — {alt["confianza"] * 100:.1f}% ({alt["nivel"]})')
    else:
        print(f'\nSin imagen de prueba en: {img_path}')
        print('Uso: python predictor.py ruta/imagen.jpg')
