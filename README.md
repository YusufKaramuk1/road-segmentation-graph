# road-segmentation-graph

Bu proje, uydu görüntülerinden yolları otomatik olarak tespit eden ve elde edilen yol maskelerini iskelet (skeleton) ve graph yapısına dönüştürerek yol ağı analizi yapan uçtan uca bir derin öğrenme pipeline’ıdır.

Pipeline şu adımlardan oluşur:

1. Uydu görüntüsünden yol segmentasyonu  
2. Yol maskesinden skeleton çıkarımı  
3. Skeleton’dan graph oluşturma  
4. Graph metrikleri ve görsel rapor üretimi  

Proje hem eğitim veri seti üzerinde hem de dışarıdan verilen rastgele uydu görüntüleri üzerinde çalışacak şekilde düzenlenmiştir.

---

## Özellikler

- U-Net + ResNet34 tabanlı yol segmentasyonu
- DeepGlobe Road Extraction veri seti ile eğitim
- Best model checkpoint kaydı
- Threshold tuning
- Harici görüntü için patch-based inference
- Skeleton çıkarımı
- Node / edge tabanlı graph üretimi
- Graph metrikleri
- Sunum için tek sayfalık final rapor görseli

---

## Kullanılan Veri Seti

DeepGlobe Road Extraction Dataset  
https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

---

## Kurulum

pip install torch torchvision  
pip install segmentation-models-pytorch  
pip install opencv-python albumentations  
pip install scikit-image networkx matplotlib pillow tqdm  

---

## Çalıştırma (Harici Görüntü Testi)

python infer_external_image.py  
python 2_skeleton.py  
python 3_graph.py  
python 4_report.py  

---

## Çıktılar

outputs/  
- latest_mask.png  
- skeleton.npy  
- graph.png  
- final_report.png  

---

## Açıklama

Bu proje sadece segmentation değil, aynı zamanda yol ağını graph yapısına çeviren bir sistemdir.  
Gerçek dünya uydu görüntüleri üzerinde test edilmiştir ve ana yol yapısını başarılı şekilde çıkarmaktadır.
