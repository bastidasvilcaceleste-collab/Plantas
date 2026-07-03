"""
Entrenamiento simplificado (papa, tomate, maíz) con EfficientNetB0.
Optimizado para CPU en laptop. Dataset: data_simple/
Salidas: ai_models/final_modelo_plantas.pth, ai_models/classes.json
"""
import os
import sys
import json
import time
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import numpy as np
from sklearn.metrics import classification_report
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data_simple')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE, 'ai_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'final_modelo_plantas.pth')
CLASSES_PATH = os.path.join(MODEL_DIR, 'classes.json')
OUTPUT_DIR = os.path.join(BASE, 'training_output')
CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint_simple.pt')
HISTORY_PLOT = os.path.join(OUTPUT_DIR, 'training_history_simple.png')
BACKBONE_TAG = 'efficientnet_b0'

IMG_SIZE = 224
EPOCHS_PHASE1 = 6
EPOCHS_PHASE2 = 10
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
WEIGHT_DECAY = 1e-4
UNFREEZE_LAST = 25
NUM_WORKERS = 0
PATIENCE = 4
MAX_GRAD_NORM = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESUME = '--resume' in sys.argv or '-r' in sys.argv
EXPORT = '--export' in sys.argv or '-e' in sys.argv

BATCH_SIZE = None

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if DEVICE.type == 'cpu':
    torch.set_num_threads(min(4, os.cpu_count() or 4))


def check_dataset():
    for split, path in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f'No existe {path}. Ejecuta primero: python prepare_simple_dataset.py'
            )


def get_class_names():
    return sorted(
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    )


def save_classes_json(class_names):
    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    print(f'Clases guardadas: {CLASSES_PATH} ({len(class_names)} clases)')


def compute_class_weights(class_names):
    counts = []
    for c in class_names:
        path = os.path.join(TRAIN_DIR, c)
        n = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        counts.append(max(n, 1))
    total = sum(counts)
    return torch.FloatTensor([total / n for n in counts]).to(DEVICE)


def train_transforms():
    """Augmentation moderado: suficiente variación sin costo extra de resize grande."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def val_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3, True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model


def detect_batch_size(num_classes, candidates=None):
    """Elige el batch más grande que cabe en RAM (CPU/GPU)."""
    if candidates is None:
        if DEVICE.type == 'cuda':
            candidates = [32, 24, 16, 12, 8]
        else:
            candidates = [16, 12, 8, 6, 4]

    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    dataset = datasets.ImageFolder(TRAIN_DIR, val_transforms())
    chosen = candidates[-1]

    for bs in candidates:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
        try:
            model.train()
            inputs, labels = next(iter(loader))
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR_PHASE1)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            chosen = bs
            print(f'  Batch size detectado: {bs}')
            break
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower() or 'not enough memory' in str(exc).lower():
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                print(f'  Batch {bs} no cabe, probando menor...')
                continue
            raise
        finally:
            del loader

    del model, criterion
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return chosen


def estimate_training_minutes(batch_size, train_batches, val_batches):
    """Estima duración total con 1 batch de referencia en CPU/GPU."""
    num_classes = len(get_class_names())
    model = build_model(num_classes).to(DEVICE)
    freeze_for_phase1(model)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_PHASE1)
    criterion = nn.CrossEntropyLoss()

    dataset = datasets.ImageFolder(TRAIN_DIR, train_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    t0 = time.time()
    inputs, labels = next(iter(loader))
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    optimizer.zero_grad()
    loss = criterion(model(inputs), labels)
    loss.backward()
    optimizer.step()
    train_batch_sec = max(time.time() - t0, 0.05)

    model.eval()
    val_ds = datasets.ImageFolder(VAL_DIR, val_transforms())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    t0 = time.time()
    with torch.no_grad():
        vi, vl = next(iter(val_loader))
        model(vi.to(DEVICE))
    val_batch_sec = max(time.time() - t0, 0.03)

    sec_per_epoch = train_batches * train_batch_sec + val_batches * val_batch_sec
    max_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2
    # early stopping suele cortar ~30% antes del máximo
    expected_epochs = int(max_epochs * 0.7)
    total_min = (sec_per_epoch * expected_epochs) / 60

    del model, optimizer, criterion, loader, val_loader
    return total_min, sec_per_epoch, expected_epochs


def create_loaders(batch_size):
    pin = DEVICE.type == 'cuda'
    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=pin)
    tr = DataLoader(
        datasets.ImageFolder(TRAIN_DIR, train_transforms()),
        shuffle=True, **kw,
    )
    vl = DataLoader(
        datasets.ImageFolder(VAL_DIR, val_transforms()),
        shuffle=False, **kw,
    )
    te = DataLoader(
        datasets.ImageFolder(TEST_DIR, val_transforms()),
        shuffle=False, **kw,
    )
    return tr, vl, te


def freeze_for_phase1(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def freeze_for_phase2(model):
    for p in model.parameters():
        p.requires_grad = False
    params = list(model.named_parameters())
    cutoff = max(0, len(params) - UNFREEZE_LAST)
    for i, (_, p) in enumerate(params):
        if i >= cutoff:
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parametros entrenables (fase 2): {trainable:,}')


def save_checkpoint(model, optimizer, epoch, phase, history, best_val_loss, class_names):
    torch.save({
        'epoch': epoch,
        'phase': phase,
        'best_val_loss': best_val_loss,
        'model_state': model.cpu().state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history,
        'num_classes': len(class_names),
        'class_names': class_names,
        'img_size': IMG_SIZE,
        'backbone': BACKBONE_TAG,
    }, CKPT_PATH)
    model.to(DEVICE)
    print(f'  Checkpoint -> {CKPT_PATH}')


def load_best_checkpoint(model):
    if not os.path.exists(CKPT_PATH):
        return None
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    if ckpt.get('backbone') and ckpt['backbone'] != BACKBONE_TAG:
        print('  AVISO: checkpoint de otro backbone; inicia entrenamiento limpio.')
        return None
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE)
    return ckpt


def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0

    with torch.set_grad_enabled(train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            _, predicted = torch.max(outputs, 1)
            total += bs
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, phase_name, start_epoch=0, history=None):
    if history is None:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    best_val_loss = float('inf')
    patience_counter = 0
    class_names = get_class_names()

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, train=True,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, train=False,
        )
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f'  [{phase_name}] Epoca {epoch + 1}/{num_epochs} | '
            f'Train loss:{train_loss:.4f} acc:{train_acc:.4f} | '
            f'Val loss:{val_loss:.4f} acc:{val_acc:.4f} | '
            f'{time.time() - t0:.0f}s'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, phase_name, history, best_val_loss, class_names,
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'  Early stopping en epoca {epoch + 1} (patience={PATIENCE})')
                break

    return model, history


def plot_training_history(history, path=HISTORY_PLOT):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='train_loss')
    axes[0].plot(history['val_loss'], label='val_loss')
    axes[0].set_title('Loss por epoca (EfficientNetB0)')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='train_acc')
    axes[1].plot(history['val_acc'], label='val_acc')
    axes[1].set_title('Accuracy por epoca')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Grafica guardada: {path}')


def export_from_checkpoint():
    if not os.path.exists(CKPT_PATH):
        print(f'ERROR: No hay checkpoint en {CKPT_PATH}')
        print('Entrena primero: python train_simple.py')
        sys.exit(1)
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    class_names = ckpt.get('class_names', get_class_names())
    model = build_model(len(class_names))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    save_classes_json(class_names)
    print(f'Modelo exportado: {MODEL_PATH}')


def save_final_model(model):
    model.cpu()
    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Modelo final guardado: {MODEL_PATH}')


def main():
    global BATCH_SIZE

    if EXPORT:
        export_from_checkpoint()
        return

    check_dataset()

    print('=' * 60)
    print('ENTRENAMIENTO SIMPLE - EfficientNetB0 + Transfer Learning (CPU)')
    print('=' * 60)
    print(f'Device: {DEVICE}  |  PyTorch: {torch.__version__}')
    print(f'Dataset: {DATA_DIR}')
    print(f'Imagen: {IMG_SIZE}x{IMG_SIZE}  |  Workers: {NUM_WORKERS}')

    class_names = get_class_names()
    num_classes = len(class_names)
    if num_classes != 19:
        print(f'  AVISO: se esperaban 19 clases, detectadas {num_classes}')
    print(f'\nClases detectadas: {num_classes}')
    for i, name in enumerate(class_names):
        print(f'  {i:2d}: {name}')

    save_classes_json(class_names)

    print('\nDetectando batch size optimo...')
    BATCH_SIZE = detect_batch_size(num_classes)

    train_loader, val_loader, test_loader = create_loaders(BATCH_SIZE)
    print(
        f'Batches -> train:{len(train_loader)}  '
        f'val:{len(val_loader)}  test:{len(test_loader)}'
    )

    est_min, sec_epoch, est_epochs = estimate_training_minutes(
        BATCH_SIZE, len(train_loader), len(val_loader),
    )
    print(
        f'\nTiempo estimado: ~{est_min:.0f} min '
        f'({est_epochs} epocas efectivas, ~{sec_epoch:.1f}s/epoca)'
    )
    print('  (B3 en CPU suele tardar 4-8x más; B0 reduce RAM y cómputo por batch)')

    weights = compute_class_weights(class_names)
    model = build_model(num_classes)
    model.to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f'Parametros totales: {params:,}  (B3 ~12M, B0 ~5M + cabeza)')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_epoch = 0
    resume_phase = None

    if RESUME and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        if ckpt.get('backbone', BACKBONE_TAG) == BACKBONE_TAG:
            model.load_state_dict(ckpt['model_state'])
            history = ckpt.get('history', history)
            start_epoch = ckpt.get('epoch', 0) + 1
            resume_phase = ckpt.get('phase', 'Phase 1')
            print(f'\nReanudando desde {resume_phase}, epoca {start_epoch}')
        else:
            print('\nCheckpoint incompatible (era B3 u otro). Entrenando desde cero.')

    criterion = nn.CrossEntropyLoss(weight=weights)

    if resume_phase != 'Phase 2':
        print(f'\n{"=" * 60}')
        print('FASE 1: Clasificador (backbone congelado - ImageNet)')
        print('=' * 60)
        freeze_for_phase1(model)
        opt1 = optim.Adam(
            model.classifier.parameters(), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY,
        )
        sched1 = optim.lr_scheduler.ReduceLROnPlateau(
            opt1, mode='min', patience=2, factor=0.5,
        )
        ep_start = start_epoch if resume_phase == 'Phase 1' else 0
        model, h1 = train_phase(
            model, train_loader, val_loader, criterion, opt1, sched1,
            EPOCHS_PHASE1, 'Phase 1', ep_start, history,
        )
        history = h1

    print(f'\n{"=" * 60}')
    print(f'FASE 2: Fine-tuning (ultimas {UNFREEZE_LAST} capas)')
    print('=' * 60)
    freeze_for_phase2(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt2 = optim.Adam(trainable, lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
    sched2 = optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode='min', patience=2, factor=0.5,
    )
    ep_start = start_epoch if resume_phase == 'Phase 2' else 0
    h2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model, h2 = train_phase(
        model, train_loader, val_loader, criterion, opt2, sched2,
        EPOCHS_PHASE2, 'Phase 2', ep_start, h2,
    )
    for key in history:
        history[key].extend(h2[key])

    plot_training_history(history)

    print(f'\n{"=" * 60}')
    print('Cargando mejor checkpoint y evaluando en test')
    print('=' * 60)
    ckpt = load_best_checkpoint(model)
    if ckpt:
        print(f'  Mejor val_loss: {ckpt["best_val_loss"]:.4f} ({ckpt["phase"]})')

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            _, pred = torch.max(model(inputs), 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = float(np.mean(all_preds == all_labels))
    report = classification_report(
        all_labels, all_preds, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    print(f'  Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)')
    print(f'  Precision (weighted): {report["weighted avg"]["precision"]:.4f}')
    print(f'  Recall (weighted):    {report["weighted avg"]["recall"]:.4f}')
    print(f'  F1 (weighted):        {report["weighted avg"]["f1-score"]:.4f}')

    save_final_model(model)

    results = {
        'backbone': BACKBONE_TAG,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'test_accuracy': accuracy,
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1': float(report['weighted avg']['f1-score']),
        'num_classes': num_classes,
        'class_names': class_names,
        'estimated_minutes': round(est_min, 1),
        'model_path': MODEL_PATH,
        'classes_path': CLASSES_PATH,
        'checkpoint': CKPT_PATH,
        'history_plot': HISTORY_PLOT,
        'trained_at': datetime.now().isoformat(),
        'device': str(DEVICE),
    }
    results_path = os.path.join(OUTPUT_DIR, 'training_results_simple.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n{"=" * 60}')
    print('COMPLETADO')
    print('=' * 60)
    print(f'Clases: {num_classes} (esperadas: 19)')
    print(f'Modelo: {MODEL_PATH}')
    print(f'Classes: {CLASSES_PATH}')
    print(f'Grafica: {HISTORY_PLOT}')
    print(f'Resultados: {results_path}')


if __name__ == '__main__':
    main()
