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
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'datasets')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE, 'ai_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'final_modelo_plantas.pth')
OUTPUT_DIR = os.path.join(BASE, 'training_output')
CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.pt')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 10
LR_PHASE1 = 0.001
LR_PHASE2 = 0.0001
UNFREEZE_LAST = 40
NUM_WORKERS = 2
PATIENCE = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXPORT = '--export' in sys.argv or '-e' in sys.argv
RESUME = '--resume' in sys.argv or '-r' in sys.argv

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def get_class_names():
    return sorted(d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)))


def compute_class_weights(class_names):
    counts = []
    for c in class_names:
        path = os.path.join(TRAIN_DIR, c)
        n = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        counts.append(n)
    total = sum(counts)
    return torch.FloatTensor([total / n for n in counts]).to(DEVICE)


def train_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15),
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


def create_loaders():
    tr = DataLoader(datasets.ImageFolder(TRAIN_DIR, train_transforms()),
                    BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    vl = DataLoader(datasets.ImageFolder(VAL_DIR, val_transforms()),
                    BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    te = DataLoader(datasets.ImageFolder(TEST_DIR, val_transforms()),
                    BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return tr, vl, te


def build_model(num_classes):
    m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    inf = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3, True), nn.Linear(inf, 512), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(512, num_classes),
    )
    return m


def save_checkpoint(model, optimizer, epoch, phase, history, best_val_loss):
    torch.save({
        'epoch': epoch, 'phase': phase, 'best_val_loss': best_val_loss,
        'model_state': model.cpu().state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history,
        'num_classes': model.classifier[-1].out_features,
        'img_size': IMG_SIZE,
    }, CKPT_PATH)
    model.to(DEVICE)
    print(f'  Checkpoint -> {CKPT_PATH}')


def load_checkpoint(model, optimizer=None):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    model.to(DEVICE)
    return ckpt


def export_from_checkpoint():
    if not os.path.exists(CKPT_PATH):
        print(f'ERROR: No checkpoint found at {CKPT_PATH}')
        print('Entrena primero: python train_model.py')
        sys.exit(1)
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    num_classes = ckpt.get('num_classes', len(get_class_names()))
    model = build_model(num_classes)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Modelo exportado: {MODEL_PATH}')
    print(f'  ({num_classes} clases, checkpoint época {ckpt.get("epoch","?")} {ckpt.get("phase","?")})')


def freeze_for_phase1(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def freeze_for_phase2(model):
    for p in model.parameters():
        p.requires_grad = False
    params = list(model.named_parameters())
    for i, (_, p) in enumerate(params):
        if i >= max(0, len(params) - UNFREEZE_LAST):
            p.requires_grad = True
    print(f'  Unfrozen last {UNFREEZE_LAST} / {len(params)} param groups')


def run_epoch(model, loader, criterion, optimizer, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    preds, labels = [], []
    with torch.set_grad_enabled(train):
        for inputs, lbls in loader:
            inputs, lbls = inputs.to(DEVICE), lbls.to(DEVICE)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, lbls)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()
            if not train:
                preds.extend(predicted.cpu().numpy())
                labels.extend(lbls.cpu().numpy())
    return total_loss / total, correct / total, preds, labels


def train_phase(model, loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, class_names, phase_name, start_epoch=0, history=None):
    if history is None:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(model, loader, criterion, optimizer, train=True)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, None, train=False)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'  [{phase_name}] Epoch {epoch+1}/{num_epochs} | '
              f'T L:{train_loss:.4f} A:{train_acc:.4f} | '
              f'V L:{val_loss:.4f} A:{val_acc:.4f} | {time.time()-t0:.0f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, phase_name, history, best_val_loss)

        if epoch > 5 and val_loss > min(history['val_loss'][:-5]):
            print(f'  Early stopping at epoch {epoch+1}')
            break

    return model, history


def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metrics, title in [
        (axes[0], ['train_loss', 'val_loss'], 'Loss'),
        (axes[1], ['train_acc', 'val_acc'], 'Accuracy'),
    ]:
        for m in metrics:
            ax.plot(history[m], label=m)
        ax.set_title(title)
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'history_{name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=90, fontsize=5); plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def save_final_model(model):
    model.cpu()
    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    model.to(DEVICE)
    print(f'Modelo guardado: {MODEL_PATH}')


def main():
    if EXPORT:
        export_from_checkpoint()
        return

    print(f'Device: {DEVICE}  PyTorch: {torch.__version__}')
    print(f'Train: {TRAIN_DIR}  Val: {VAL_DIR}  Test: {TEST_DIR}')

    class_names = get_class_names()
    num_classes = len(class_names)
    print(f'Classes: {num_classes}')

    weights = compute_class_weights(class_names)
    train_loader, val_loader, test_loader = create_loaders()
    print(f'Batches: Train {len(train_loader)}  Val {len(val_loader)}  Test {len(test_loader)}')

    model = build_model(num_classes)
    model.to(DEVICE)
    print(f'Model: EfficientNetB3  Params: {sum(p.numel() for p in model.parameters()):,}')

    if RESUME and os.path.exists(CKPT_PATH):
        print(f'\nResuming: {CKPT_PATH}')
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        resume_phase = ckpt.get('phase', 'Phase 1')
        history = ckpt.get('history')
        print(f'  Continuando en {resume_phase}, época {start_epoch}')
    else:
        start_epoch = 0
        resume_phase = None
        history = None
        if RESUME:
            print(f'No hay checkpoint en {CKPT_PATH}, empezando desde cero')

    if resume_phase != 'Phase 2':
        if not RESUME:
            print(f'\n{"="*60}\nPHASE 1: Classifier training (frozen backbone)\n{"="*60}')
            freeze_for_phase1(model)
            opt = optim.Adam(model.classifier.parameters(), lr=LR_PHASE1)
            sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.5)
            model, history = train_phase(
                model, train_loader, val_loader,
                nn.CrossEntropyLoss(weight=weights), opt, sched,
                EPOCHS_PHASE1, class_names, 'Phase 1',
                start_epoch if resume_phase == 'Phase 1' else 0, history)
            plot_history(history, 'phase_1')

    print(f'\n{"="*60}\nPHASE 2: Fine-tuning (unfreezing last {UNFREEZE_LAST})\n{"="*60}')
    freeze_for_phase2(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=LR_PHASE2)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.5)

    epoch_start = start_epoch if resume_phase == 'Phase 2' else 0
    h2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    model, h2 = train_phase(
        model, train_loader, val_loader,
        nn.CrossEntropyLoss(weight=weights), opt, sched,
        EPOCHS_PHASE2, class_names, 'Phase 2', epoch_start, h2)

    if history:
        for k in history: history[k].extend(h2[k])
    plot_history(history or h2, 'full')

    print(f'\n{"="*60}\nTEST EVALUATION\n{"="*60}')
    model.eval()
    all_preds, all_labels = [], []
    for inputs, lbls in test_loader:
        inputs, lbls = inputs.to(DEVICE), lbls.to(DEVICE)
        with torch.no_grad():
            _, pred = torch.max(model(inputs), 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names,
                                   output_dict=True, zero_division=0)
    print(f'  Accuracy:  {accuracy:.4f}')
    print(f'  Precision: {report["weighted avg"]["precision"]:.4f}')
    print(f'  Recall:    {report["weighted avg"]["recall"]:.4f}')
    print(f'  F1-Score:  {report["weighted avg"]["f1-score"]:.4f}')

    plot_confusion_matrix(cm, class_names)

    print(f'\n{"="*60}\nSAVING MODEL\n{"="*60}')
    save_final_model(model)

    final_results = {
        'test_accuracy': float(accuracy),
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1': float(report['weighted avg']['f1-score']),
        'classes': len(class_names),
        'trained_at': datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f'\nCompletado. Modelo: {MODEL_PATH}')
    print(f'Resultados: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
