"""
Entrenamiento de ResNet18 para detección de enfermedades en plantas.
Ejecutar: python train.py

Lee todas las imágenes de datasets/clean_final/, normaliza y
deduplica los nombres de carpetas, entrena con regularización
y guarda el modelo + classes.json en modelos_ia/ai_models/.
"""

import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from collections import OrderedDict
from PIL import Image

# ── CONFIGURACIÓN ──────────────────────────────────────────

EPOCHS = 15
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 0.001
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
TEMPERATURE = 2.0
PATIENCE = 5
SEED = 42

torch.manual_seed(SEED)

BASE = os.path.dirname  # helper
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, *[os.pardir]*4))
DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'clean_final')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'ai_models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {DEVICE}')
print(f'Datos: {DATA_DIR}')
print(f'Salida: {OUTPUT_DIR}')

# ── NORMALIZACIÓN DE NOMBRES DE CARPETA ────────────────────

def normalizar_nombre(nombre):
    """Limpia nombres: unifica separadores, quita redundancias."""
    n = nombre.strip()
    n = re.sub(r'_+', '_', n)      # multiple underscores -> single
    n = re.sub(r'_+$', '', n)      # trailing _
    n = re.sub(r'^_+', '', n)      # leading _
    # Capitalizar primera letra de cada palabra
    n = n.replace('_', ' ').title().replace(' ', '_')
    # Casos especiales
    n = n.replace('_Leaf__', '_Leaf_')
    n = n.replace('__', '_')
    n = re.sub(r'_+', '_', n)
    n = re.sub(r'_+$', '', n)
    return n

def agrupar_carpetas(data_dir):
    """Agrupa carpetas con nombre normalizado y devuelve
       {nombre_limpio: [lista_de_rutas_completas]}"""
    grupos = OrderedDict()
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full):
            continue
        imagenes = [f for f in os.listdir(full)
                    if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp','.tiff'))]
        if not imagenes:
            print(f'  [skip] {entry} - sin imagenes')
            continue
        limpio = normalizar_nombre(entry)
        if limpio not in grupos:
            grupos[limpio] = []
        grupos[limpio].append(full)
    return grupos

print('\nAnalizando carpetas del dataset...')
grupos = agrupar_carpetas(DATA_DIR)
print(f'Carpetas originales: {sum(len(v) for v in grupos.values())} -> '
      f'{len(grupos)} clases unicas tras normalizar')

for name, dirs in grupos.items():
    total_imgs = sum(len(os.listdir(d)) for d in dirs)
    print(f'  {name}: {len(dirs)} carpeta(s), ~{total_imgs} imagenes')

# ── TRANSFORMACIONES ───────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── DATASET CON CATEGORÍAS AGRUPADAS ───────────────────────

class GrupoImageFolder(datasets.ImageFolder):
    """ImageFolder que agrupa múltiples carpetas bajo una misma clase."""

    def __init__(self, root, grupos, transform=None):
        self.grupos = grupos
        # Construir lista ordenada de clases
        self.classes = sorted(grupos.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # Recolectar samples: (ruta_a_imagen, índice_de_clase)
        self.samples = []
        self.targets = []
        for clase, carpetas in grupos.items():
            idx = self.class_to_idx[clase]
            for carpeta in carpetas:
                for archivo in sorted(os.listdir(carpeta)):
                    if archivo.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp','.tiff')):
                        ruta = os.path.join(carpeta, archivo)
                        self.samples.append((ruta, idx))
                        self.targets.append(idx)
        self.imgs = self.samples
        self.transform = transform
        self.target_transform = None
        self.loader = datasets.folder.default_loader

dataset = GrupoImageFolder(DATA_DIR, grupos, transform=train_transform)

print(f'\nTotal clases: {len(dataset.classes)}')
print(f'Total imágenes: {len(dataset)}')

# ── STRATIFIED TRAIN/VAL SPLIT ─────────────────────────────

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(list(range(len(dataset))), dataset.targets))

train_dataset = Subset(dataset, train_idx)
train_dataset.dataset.transform = train_transform
val_dataset = Subset(dataset, val_idx)
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

print(f'Entrenamiento: {len(train_dataset)} imágenes')
print(f'Validación: {len(val_dataset)} imágenes')

# ── GUARDAR CLASSES.JSON ───────────────────────────────────

classes_path = os.path.join(OUTPUT_DIR, 'classes.json')
with open(classes_path, 'w', encoding='utf-8') as f:
    json.dump(dataset.classes, f, ensure_ascii=False, indent=2)
print(f'\nclasses.json guardado en {classes_path}')

# ── MODELO ─────────────────────────────────────────────────

NUM_CLASSES = len(dataset.classes)
modelo = resnet18(weights=None)
modelo.fc = nn.Linear(512, NUM_CLASSES)
modelo = modelo.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── ENTRENAMIENTO ──────────────────────────────────────────

best_acc = 0.0
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    modelo.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = modelo(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100.0 * correct / total
    avg_loss = train_loss / len(train_loader)

    # Validación
    modelo.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = modelo(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100.0 * val_correct / val_total
    current_lr = optimizer.param_groups[0]['lr']

    print(f'Epoch {epoch:2d}/{EPOCHS} | '
          f'LR: {current_lr:.2e} | '
          f'Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%')

    scheduler.step()

    # Early stopping + mejor modelo
    if val_acc > best_acc:
        best_acc = val_acc
        epochs_no_improve = 0
        model_path = os.path.join(OUTPUT_DIR, 'final_modelo_plantas.pth')
        torch.save(modelo.state_dict(), model_path)
        print(f'  >> Mejor modelo guardado (Val Acc: {val_acc:.2f}%)')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f'  >> Early stopping en epoch {epoch}')
            break

# ── REPORTE FINAL ──────────────────────────────────────────

print(f'\n{"="*60}')
print(f'ENTRENAMIENTO COMPLETO')
print(f'Mejor precisión en validación: {best_acc:.2f}%')
print(f'Clases totales: {NUM_CLASSES}')
print(f'Modelo guardado en: {model_path}')
print(f'classes.json: {classes_path}')
print(f'{"="*60}')

# Evaluación por clase
print('\nPrecisión por clase (top-1 en validación):')
modelo.eval()
clase_correctas = {i: 0 for i in range(NUM_CLASSES)}
clase_total = {i: 0 for i in range(NUM_CLASSES)}
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = modelo(inputs)
        _, predicted = torch.max(outputs, 1)
        for lbl, pred in zip(labels, predicted):
            clase_total[lbl.item()] += 1
            if lbl == pred:
                clase_correctas[lbl.item()] += 1

for i in range(NUM_CLASSES):
    tot = clase_total[i]
    corr = clase_correctas[i]
    pct = 100.0 * corr / tot if tot > 0 else 0
    barra = '#' * int(pct / 5) + '.' * (20 - int(pct / 5))
    print(f'  {dataset.classes[i]:45s} | {barra} {pct:5.1f}% ({corr}/{tot})')
