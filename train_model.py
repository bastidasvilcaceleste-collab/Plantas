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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'datasets')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE, 'ai_models')
OUTPUT_DIR = os.path.join(BASE, 'training_output')

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
CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.pt')
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
    weights = torch.FloatTensor([sum(counts) / n for n in counts]).to(DEVICE)
    return weights


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


def save_checkpoint(model, optimizer, epoch, phase, history, best_val_loss, path=CKPT_PATH):
    torch.save({
        'epoch': epoch, 'phase': phase, 'best_val_loss': best_val_loss,
        'model_state': model.cpu().state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history,
        'num_classes': model.classifier[-1].out_features,
    }, path)
    model.to(DEVICE)
    print(f'  Checkpoint saved: {path} (epoch {epoch}, {phase})')


def load_checkpoint(model, optimizer, path=CKPT_PATH):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    model.to(DEVICE)
    return ckpt


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
    print(f'  Unfrozen last {UNFREEZE_LAST} of {len(params)} param groups')


def run_epoch(model, loader, criterion, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
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
              f'Train L:{train_loss:.4f} A:{train_acc:.4f} | '
              f'Val L:{val_loss:.4f} A:{val_acc:.4f} | {time.time()-t0:.0f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, phase_name, history, best_val_loss)

        if epoch > 5 and val_loss > min(history['val_loss'][:-5]):
            print(f'  Early stopping at epoch {epoch+1}')
            break

    return model, history


def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [
        (axes[0], ['train_loss', 'val_loss'], 'Loss'),
        (axes[1], ['train_acc', 'val_acc'], 'Accuracy'),
    ]:
        for m in metric:
            ax.plot(history[m], label=m)
        ax.set_title(title)
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'history_{name}.png'), dpi=150)
    plt.close()
    print(f'  Saved: history_{name}.png')


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=90, fontsize=5); plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f'  Saved: confusion_matrix.png')


def save_final(model, class_names):
    torch.save(model.cpu().state_dict(), os.path.join(MODEL_DIR, 'final_modelo_plantas.pth'))
    print(f'  Saved: {MODEL_DIR}/final_modelo_plantas.pth')
    with open(os.path.join(MODEL_DIR, 'classes.json'), 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f'  Saved: {MODEL_DIR}/classes.json ({len(class_names)} classes)')


def main():
    print(f'Device: {DEVICE}  PyTorch: {torch.__version__}')
    print(f'Train: {TRAIN_DIR}  Val: {VAL_DIR}  Test: {TEST_DIR}')
    print()

    class_names = get_class_names()
    num_classes = len(class_names)
    print(f'Classes: {num_classes}')

    weights = compute_class_weights(class_names)
    train_loader, val_loader, test_loader = create_loaders()
    print(f'Train: {len(train_loader)} batches  Val: {len(val_loader)}  Test: {len(test_loader)}')
    print()

    model = build_model(num_classes)
    model.to(DEVICE)
    print(f'Model: EfficientNetB3  Params: {sum(p.numel() for p in model.parameters()):,}')

    if RESUME and os.path.exists(CKPT_PATH):
        print(f'\nResuming from checkpoint: {CKPT_PATH}')
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        resume_phase = ckpt.get('phase', 'Phase 1')
        history = ckpt.get('history', None)
        print(f'  Resumed at {resume_phase}, epoch {start_epoch}/{ckpt.get("num_epochs", "?")}')
    else:
        start_epoch = 0
        resume_phase = None
        history = None
        if RESUME:
            print(f'No checkpoint found at {CKPT_PATH}, starting fresh')

    if resume_phase != 'Phase 2':
        if not RESUME:
            print(f'\n{"="*60}\nPHASE 1: Training classifier (frozen backbone)\n{"="*60}')
            freeze_for_phase1(model)
            optimizer = optim.Adam(model.classifier.parameters(), lr=LR_PHASE1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
            model, history = train_phase(
                model, train_loader, val_loader,
                nn.CrossEntropyLoss(weight=weights), optimizer, scheduler,
                EPOCHS_PHASE1, class_names, 'Phase 1',
                start_epoch if resume_phase == 'Phase 1' else 0,
                history
            )
            plot_history(history, 'phase_1')

    print(f'\n{"="*60}\nPHASE 2: Fine-tuning (unfreezing last {UNFREEZE_LAST} layers)\n{"="*60}')
    freeze_for_phase2(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=LR_PHASE2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    epoch_start = start_epoch if resume_phase == 'Phase 2' else 0
    hist2_start = len(history['train_loss']) if history and resume_phase == 'Phase 2' else 0
    phase2_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    model, h2 = train_phase(
        model, train_loader, val_loader,
        nn.CrossEntropyLoss(weight=weights), optimizer, scheduler,
        EPOCHS_PHASE2, class_names, 'Phase 2',
        epoch_start, phase2_history
    )

    if history:
        for k in history:
            history[k].extend(h2[k])
    plot_history(history or h2, 'full')

    print(f'\n{"="*60}\nTEST EVALUATION\n{"="*60}')
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for inputs, lbls in test_loader:
        inputs, lbls = inputs.to(DEVICE), lbls.to(DEVICE)
        with torch.no_grad():
            out = model(inputs)
            probs = torch.softmax(out, 1)
            _, pred = torch.max(out, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names,
                                   output_dict=True, zero_division=0)
    print(f'  Test Accuracy:  {accuracy:.4f}')
    print(f'  Precision:      {report["weighted avg"]["precision"]:.4f}')
    print(f'  Recall:         {report["weighted avg"]["recall"]:.4f}')
    print(f'  F1-Score:       {report["weighted avg"]["f1-score"]:.4f}')

    plot_confusion_matrix(cm, class_names)

    print(f'\n{"="*60}\nSAVING MODEL\n{"="*60}')
    save_final(model, class_names)

    results = {
        'test': {k: float(v) for k, v in report['weighted avg'].items() if k != 'support'},
        'test_accuracy': float(accuracy),
        'classes': len(class_names),
        'trained_at': datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nDone. Outputs in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
