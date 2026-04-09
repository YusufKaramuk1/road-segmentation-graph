import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

# =========================================================
# AYARLAR
# =========================================================
DATA_DIR   = "data/train"
SAVE_DIR   = "outputs"

IMG_SIZE   = 512
BATCH_SIZE = 4
EPOCHS     = 30
LR         = 1e-4
MAX_IMGS   = 6226
SEED       = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Cihaz: {DEVICE}")

# =========================================================
# DATASET
# =========================================================
class RoadDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = [p.replace("_sat.jpg", "_mask.png") for p in img_paths]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Maske okunamadı: {mask_path}")

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            out = self.transform(image=img, mask=mask)
            img = out["image"]
            mask = out["mask"].unsqueeze(0)  # (1, H, W)

        return img, mask


# =========================================================
# AUGMENTATION
# =========================================================
train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),

    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    # ShiftScaleRotate yerine Affine kullandık
    A.Affine(
        scale=(0.95, 1.05),
        translate_percent=(-0.05, 0.05),
        rotate=(-15, 15),
        shear=(-5, 5),
        p=0.4
    ),

    A.RandomBrightnessContrast(p=0.3),
    A.ColorJitter(p=0.2),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),
])

# =========================================================
# VERİYİ BUL / SPLIT
# =========================================================
all_imgs = sorted(glob.glob(os.path.join(DATA_DIR, "*_sat.jpg")))[:MAX_IMGS]
assert len(all_imgs) > 0, f"Görüntü bulunamadı: {DATA_DIR}/*_sat.jpg"

random.shuffle(all_imgs)

split_idx = int(len(all_imgs) * 0.8)
train_imgs = all_imgs[:split_idx]
val_imgs   = all_imgs[split_idx:]

train_ds = RoadDataset(train_imgs, transform=train_tf)
val_ds   = RoadDataset(val_imgs, transform=val_tf)

# Windows'ta en güvenlisi 0, sorun çıkmaz
num_workers = 0

train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(DEVICE == "cuda")
)

val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(DEVICE == "cuda")
)

print(f"Eğitim: {len(train_ds)} | Doğrulama: {len(val_ds)} görüntü")

# =========================================================
# MODEL
# =========================================================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

# =========================================================
# LOSS
# =========================================================
bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

def focal_loss_with_logits(logits, target, alpha=0.8, gamma=2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = target * probs + (1 - target) * (1 - probs)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

def combined_loss(logits, target):
    # Ana yolları ve ince yolları birlikte daha iyi öğrenmesi için
    bce   = bce_loss(logits, target)
    dice  = dice_loss(logits, target)
    focal = focal_loss_with_logits(logits, target)
    return 0.5 * bce + 0.3 * dice + 0.2 * focal

# =========================================================
# METRİK
# =========================================================
def iou_score(logits, target, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / (union + 1e-6)).item()

# =========================================================
# OPTIMIZER / SCHEDULER / AMP
# =========================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

use_amp = (DEVICE == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# =========================================================
# EĞİTİM
# =========================================================
history = {
    "train_loss": [],
    "val_loss": [],
    "val_iou": []
}

best_iou = -1.0
best_epoch = 0

for epoch in range(EPOCHS):
    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0.0

    train_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
    for imgs, masks in train_bar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss = combined_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_dl)

    # ---------------- VAL ----------------
    model.eval()
    val_loss = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss = combined_loss(logits, masks)

            val_loss += loss.item()
            val_iou += iou_score(logits, masks, threshold=0.5)

    val_loss /= len(val_dl)
    val_iou /= len(val_dl)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)

    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}", end="")

    if val_iou > best_iou:
        best_iou = val_iou
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print("  ← best ✔")
    else:
        print()

print(f"\nEn iyi model: Epoch {best_epoch} | Val IoU: {best_iou:.4f}")

# Son modeli de ayrıca kaydet
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pth"))

# =========================================================
# BEST MODEL YÜKLE + THRESHOLD TUNING
# =========================================================
print("\nThreshold tuning yapılıyor...")

model.load_state_dict(
    torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=DEVICE)
)
model.eval()

threshold_candidates = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
best_thresh = 0.50
best_thresh_iou = -1.0

with torch.no_grad():
    for thresh in threshold_candidates:
        total_iou = 0.0

        for imgs, masks in val_dl:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            logits = model(imgs)
            total_iou += iou_score(logits, masks, threshold=thresh)

        avg_iou = total_iou / len(val_dl)
        print(f"  Threshold {thresh:.2f} -> IoU {avg_iou:.4f}")

        if avg_iou > best_thresh_iou:
            best_thresh_iou = avg_iou
            best_thresh = thresh

print(f"\nEn iyi threshold: {best_thresh:.2f} | IoU: {best_thresh_iou:.4f}")

with open(os.path.join(SAVE_DIR, "best_threshold.txt"), "w", encoding="utf-8") as f:
    f.write(str(best_thresh))

# =========================================================
# GRAFİKLER
# =========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["train_loss"], label="Train Loss")
ax1.plot(history["val_loss"], label="Val Loss")
ax1.axvline(best_epoch - 1, color="green", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(history["val_iou"], color="green", label="Val IoU")
ax2.axvline(best_epoch - 1, color="red", linestyle="--", alpha=0.6, label=f"Best epoch ({best_epoch})")
ax2.set_title("IoU Skoru")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
plt.show()

# =========================================================
# ÖRNEK TAHMİNLER
# =========================================================
model.eval()

sample_ds = RoadDataset(val_imgs[:3], transform=val_tf)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes[0, 0].set_title("Uydu Görüntüsü")
axes[0, 1].set_title("Gerçek Maske")
axes[0, 2].set_title(f"Model Tahmini (t={best_thresh:.2f})")

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

for i in range(3):
    img_t, mask_t = sample_ds[i]

    with torch.no_grad():
        logits = model(img_t.unsqueeze(0).to(DEVICE))
        pred = (torch.sigmoid(logits) > best_thresh).float().squeeze().cpu().numpy()

    img_show = img_t.permute(1, 2, 0).numpy()
    img_show = np.clip(img_show * std + mean, 0, 1)

    axes[i, 0].imshow(img_show)
    axes[i, 1].imshow(mask_t.squeeze().numpy(), cmap="gray")
    axes[i, 2].imshow(pred, cmap="gray")

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "predictions.png"), dpi=150)
plt.show()

# =========================================================
# 20 ADET MASK KAYDET
# =========================================================
mask_out_dir = os.path.join(SAVE_DIR, "masks")
os.makedirs(mask_out_dir, exist_ok=True)

save_count = min(20, len(val_imgs))

for i in range(save_count):
    img_path = val_imgs[i]
    img_t, _ = RoadDataset([img_path], transform=val_tf)[0]

    with torch.no_grad():
        logits = model(img_t.unsqueeze(0).to(DEVICE))
        pred = (torch.sigmoid(logits) > best_thresh).float().squeeze().cpu().numpy()

    pred_uint8 = (pred * 255).astype(np.uint8)
    save_name = os.path.basename(img_path).replace("_sat.jpg", "_mask.png")
    cv2.imwrite(os.path.join(mask_out_dir, save_name), pred_uint8)

print(f"{save_count} maske kaydedildi -> outputs/masks/")
print("\nAşama 1 tamamlandı! Sıradaki: python 2_skeleton.py")