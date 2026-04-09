import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

SAVE_DIR = "outputs"
EXT_DIR = os.path.join(SAVE_DIR, "external")

MASK_PATH = os.path.join(EXT_DIR, "latest_mask.png")
INPUT_PATH = os.path.join(EXT_DIR, "latest_input.png")

if not os.path.exists(MASK_PATH):
    raise FileNotFoundError("latest_mask.png bulunamadı. Önce infer_external_image.py çalıştır.")

mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise ValueError("Maske okunamadı.")

_, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

skeleton = skeletonize(mask_clean > 0).astype(np.uint8)

print(f"Skeleton oluşturuldu. Yol pikseli: {skeleton.sum()}")

np.save(os.path.join(SAVE_DIR, "skeleton.npy"), skeleton)

img = cv2.imread(INPUT_PATH)
if img is not None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

skeleton_rgb = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
skeleton_rgb[skeleton > 0] = [255, 0, 0]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
if img is not None:
    plt.title("Uydu Görüntüsü")
    plt.imshow(img)
else:
    plt.title("Yol Maskesi")
    plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Temizlenmiş Maske")
plt.imshow(mask_clean, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Skeleton")
plt.imshow(skeleton_rgb)
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "skeleton.png"), dpi=150)
plt.show()

print("✔ Skeleton tamamlandı")
print("Sıradaki: python 3_graph.py")