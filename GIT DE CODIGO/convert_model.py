import os
import json
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'ai_models')
OLD_MODEL_PATH = os.path.join(MODEL_DIR, 'final_modelo_plantas.pth')
OLD_CLASSES_PATH = os.path.join(MODEL_DIR, 'classes.json')
NEW_MODEL_PATH = os.path.join(MODEL_DIR, 'final_modelo_plantas.pth')
NEW_CLASSES_PATH = os.path.join(MODEL_DIR, 'classes.json')

TARGET_PREFIXES = ('Potato_', 'Tomato_', 'Corn_')
NUM_CLASSES = 19


def build_model(num_classes):
    m = efficientnet_b3(weights=None)
    inf = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3, True), nn.Linear(inf, 512), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(512, num_classes),
    )
    return m


def main():
    print('=== CONVIRTIENDO MODELO 107->19 CLASES ===')
    
    # Cargar classes.json original
    with open(OLD_CLASSES_PATH, 'r', encoding='utf-16') as f:
        old_classes = json.load(f)
    print(f'Clases originales: {len(old_classes)}')
    
    # Identificar las 19 clases objetivo
    target_indices = []
    target_classes = []
    for i, c in enumerate(old_classes):
        if c.startswith(TARGET_PREFIXES):
            target_indices.append(i)
            target_classes.append(c)
    print(f'Clases objetivo: {len(target_classes)}')
    for c in target_classes:
        print(f'  {c}')
    
    # Guardar nuevo classes.json
    with open(NEW_CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(target_classes, f, indent=2, ensure_ascii=False)
    print(f'Classes.json guardado: {NEW_CLASSES_PATH} ({len(target_classes)} clases)')
    
    # Construir modelo nuevo con 19 salidas
    new_model = build_model(NUM_CLASSES)
    
    # Cargar checkpoint 107 y extraer pesos
    ckpt_107 = torch.load(OLD_MODEL_PATH, map_location='cpu', weights_only=True)
    old_sd = ckpt_107 if isinstance(ckpt_107, dict) and 'model_state' not in ckpt_107 else ckpt_107.get('model_state', ckpt_107)
    
    # Filtrar state_dict: backbone igual, classifier adaptado
    new_sd = new_model.state_dict()
    
    for k in new_sd:
        if k.startswith('classifier'):
            # Saltar pesos del clasificador (se copiaran aparte)
            continue
        if k in old_sd and old_sd[k].shape == new_sd[k].shape:
            new_sd[k] = old_sd[k]
    
    # Copiar pesos del clasificador para las 19 clases objetivo
    # old classifier: classifier.4.weight [107, 512], classifier.4.bias [107]
    # new classifier: classifier.4.weight [19, 512], classifier.4.bias [19]
    old_cls_weight = old_sd['classifier.4.weight']  # [107, 512]
    old_cls_bias = old_sd['classifier.4.bias']      # [107]
    
    new_sd['classifier.4.weight'] = old_cls_weight[target_indices]
    new_sd['classifier.4.bias'] = old_cls_bias[target_indices]
    print(f'Pesos clasificador copiados: {len(target_indices)} clases')
    
    # Copiar pesos de la capa classifier.1 (Linear 1280->512)
    if 'classifier.1.weight' in old_sd and 'classifier.1.weight' in new_sd:
        new_sd['classifier.1.weight'] = old_sd['classifier.1.weight']
        new_sd['classifier.1.bias'] = old_sd['classifier.1.bias']
        print('Pesos classifier.1 copiados')
    
    new_model.load_state_dict(new_sd)
    new_model.eval()
    
    # Guardar
    torch.save(new_model.state_dict(), NEW_MODEL_PATH)
    print(f'Modelo guardado: {NEW_MODEL_PATH}')
    
    # Verificacion
    print('\n=== VERIFICACION ===')
    print(f'Modelo: EfficientNetB3')
    print(f'Salidas: {new_model.classifier[-1].out_features}')
    
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = new_model(dummy)
        print(f'Output shape: {out.shape}')
        probs = torch.softmax(out[0], dim=0)
        print(f'Probabilidades (primeras 5): {probs[:5].tolist()}')
        print(f'Suma: {probs.sum().item():.4f}')
    
    print('\n=== CONVERSION COMPLETA ===')


if __name__ == '__main__':
    main()
