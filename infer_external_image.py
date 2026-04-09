import os
import math
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

IMAGE_PATH   = "test_images/test1.png"
MODEL_PATH   = "outputs/best_model.pth"
THRESH_PATH  = "outputs/best_threshold.txt"
SAVE_DIR     = "outputs/external"

PATCH_SIZE   = 512
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Cihaz: {DEVICE}")

if os.path.exists(THRESH_PATH):
    with open(THRESH_PATH, "r", encoding="utf-8") as f:
        best_thresh = float(f.read().strip())
else:
    best_thresh = 0.45

print(f"Kullanılan threshold: {best_thresh}")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

test_tf = A.Compose([
    A.Resize(PATCH_SIZE, PATCH_SIZE),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),
])

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Görsel bulunamadı: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
orig_h, orig_w = img_rgb.shape[:2]

print(f"Orijinal boyut: {orig_w}x{orig_h}")

pad_h = math.ceil(orig_h / PATCH_SIZE) * PATCH_SIZE
pad_w = math.ceil(orig_w / PATCH_SIZE) * PATCH_SIZE

padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
padded[:orig_h, :orig_w] = img_rgb

mask_full = np.zeros((pad_h, pad_w), dtype=np.uint8)

patch_count = 0

for y in range(0, pad_h, PATCH_SIZE):
    for x in range(0, pad_w, PATCH_SIZE):
        patch = padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        out = test_tf(image=patch)
        img_t = out["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img_t)
            probs = torch.sigmoid(logits)
            pred = (probs > best_thresh).float().squeeze().cpu().numpy()

        pred_uint8 = (pred * 255).astype(np.uint8)
        mask_full[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = pred_uint8
        patch_count += 1

print(f"Toplam patch sayısı: {patch_count}")

mask_cropped = mask_full[:orig_h, :orig_w]

input_save = os.path.join(SAVE_DIR, "latest_input.png")
mask_save  = os.path.join(SAVE_DIR, "latest_mask.png")
mask_npy   = os.path.join(SAVE_DIR, "latest_mask.npy")

cv2.imwrite(input_save, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(mask_save, mask_cropped)
np.save(mask_npy, mask_cropped)

print(f"Girdi kaydedildi  -> {input_save}")
print(f"Maske kaydedildi  -> {mask_save}")
print(f"Maske npy kaydedildi -> {mask_npy}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_rgb)
axes[0].set_title(f"Orijinal Görsel ({orig_w}x{orig_h})")
axes[0].axis("off")

axes[1].imshow(mask_cropped, cmap="gray")
axes[1].set_title("Tahmin Edilen Yol Maskesi")
axes[1].axis("off")

overlay = img_rgb.copy()
overlay_mask = mask_cropped > 127
overlay[overlay_mask] = [255, 255, 255]

axes[2].imshow(overlay)
axes[2].set_title("Mask Overlay")
axes[2].axis("off")

plt.tight_layout()
panel_path = os.path.join(SAVE_DIR, "latest_prediction_panel.png")
plt.savefig(panel_path, dpi=150)
plt.show()

print(f"Panel kaydedildi -> {panel_path}")
print("\nDış görsel inference tamamlandı.")
print("Sıradaki: python 2_skeleton.py")