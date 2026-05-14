import json
import os
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE, 'final_modelo_plantas.pth')
CLASSES_PATH = os.path.join(BASE, 'classes.json')

IMG_SIZE = 224
CONFIANZA_MINIMA = 0.60
TEMPERATURA = 1.0
FORMATOS_VALIDOS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'}

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"No se encuentra classes.json en {CLASSES_PATH}")

with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_model = None


def _build_model():
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES),
    )
    return model


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modelo no encontrado en {MODEL_PATH}. "
                "Ejecuta train_model.py para entrenar el modelo."
            )
        modelo = _build_model()
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'),
                                weights_only=True)
        modelo.load_state_dict(state_dict)
        modelo.eval()
        _model = modelo
    return _model


def _extraer_bytes(archivo):
    if isinstance(archivo, (bytes, bytearray)):
        return BytesIO(archivo)
    data = archivo.read()
    archivo.seek(0)
    return BytesIO(data)


def validar_imagen(archivo):
    if archivo is None:
        raise ValueError("No se proporcionó ninguna imagen.")
    if hasattr(archivo, 'filename') and archivo.filename:
        ext = archivo.filename.rsplit('.', 1)[-1].lower()
        if ext not in FORMATOS_VALIDOS:
            raise ValueError(
                f"Formato no soportado: .{ext}. "
                f"Usa: {', '.join(sorted(FORMATOS_VALIDOS))}."
            )
    buf = _extraer_bytes(archivo)
    try:
        from PIL import Image, UnidentifiedImageError
        img = Image.open(buf)
        img.verify()
    except (UnidentifiedImageError, Exception):
        raise ValueError("El archivo no es una imagen válida o está corrupto.")


def _nivel_confianza(confianza):
    if confianza >= 0.85:
        return 'alta'
    elif confianza >= 0.65:
        return 'moderada'
    else:
        return 'preliminar'


def predict_disease(img_file):
    validar_imagen(img_file)
    buf = _extraer_bytes(img_file)
    from PIL import Image
    img = Image.open(buf).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    model = get_model()
    with torch.no_grad():
        logits = model(img_tensor)
        logits = logits / TEMPERATURA
        probabilities = torch.softmax(logits[0], dim=0)

    top5_conf, top5_idx = torch.topk(probabilities, min(5, NUM_CLASSES))

    def build(idx, conf):
        clase = CLASS_NAMES[idx]
        conf_val = round(conf.item(), 4)
        return {
            'clase': clase,
            'confianza': conf_val,
            'es_sana': 'healthy' in clase.lower(),
            'nivel': _nivel_confianza(conf_val),
        }

    predicciones = [build(top5_idx[i].item(), top5_conf[i])
                    for i in range(len(top5_idx))]
    principal = predicciones[0]

    if principal['confianza'] < CONFIANZA_MINIMA:
        principal = {
            'clase': 'Resultado_poco_confiable',
            'confianza': round(principal['confianza'], 4),
            'es_sana': False,
            'nivel': 'preliminar',
        }

    primer_pred = predicciones[0]
    segunda_pred = predicciones[1] if len(predicciones) > 1 else None
    tercera_pred = predicciones[2] if len(predicciones) > 2 else None

    return {
        'principal': principal,
        'segunda': segunda_pred,
        'tercera': tercera_pred,
        'alternativas': (predicciones[1:] if principal.get('clase')
                         != 'Resultado_poco_confiable' else predicciones),
    }
