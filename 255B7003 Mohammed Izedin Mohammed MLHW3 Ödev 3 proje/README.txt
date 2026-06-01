## Açıklama

Makine Öğrenmesi (BLM5110) dersi kapsamında Scikit-learn kullanılarak geliştirilen
k-Means müşteri segmentasyonu projesi. Online Retail II (UCI) veri kümesi kullanılarak
RFM (Recency, Frequency, Monetary) analizi ile müşteri segmentasyonu yapılmıştır.

Projede uygulanan adımlar:
- Özellik Mühendisliği: RFM değerlerinin hesaplanması
- Dağılım Kontrolü: Skewness analizi ve log dönüşümü
- k-Means Kümeleme: Elbow ve Silhouette analizi ile optimal k belirleme
- Görselleştirme: PCA ile 2 boyutlu küme görselleştirmesi

## Gereksinimler

Gerekli kütüphaneleri yüklemek için:

```
pip install -r requirements.txt
```

Veya manuel olarak:
```
pip install numpy pandas scikit-learn matplotlib seaborn openpyxl
```

## Veri Seti

`dataset/online_retail_II.xlsx` dosyası Online Retail II veri kümesini içerir.
Veri kümesi 2 yıllık (2009-2011) e-ticaret işlemlerini içermektedir.

Veri kümesine aşağıdaki linkten erişebilirsiniz:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

İndirilen dosyayı `dataset/` klasörüne koyun.

## Çalıştırma

### Eğitim:

Tüm analiz adımlarını çalıştırmak için:

```
python train.py
```

Bu komut:
- Veriyi yükler ve RFM değerlerini hesaplar
- Skewness analizi yapar ve gerekirse log dönüşümü uygular
- Standardizasyon uygular
- k=2'den k=10'a kadar Elbow ve Silhouette analizi yapar
- Optimal k ile final model oluşturur
- PCA ile 2D görselleştirme yapar
- Sonuçları `results/` klasörüne kaydeder

### Değerlendirme:

Küme analizini detaylı incelemek için:

```
python eval.py
```

Bu komut:
- Kaydedilmiş segmentleri yükler
- Küme istatistiklerini hesaplar
- Her küme için detaylı analiz sunar
- Örnek müşterileri gösterir

## Dosya Düzeni

```
MLHW3/
├── train.py              # Ana eğitim scripti
├── eval.py               # Değerlendirme scripti
├── dataset.py            # Veri yükleme ve RFM hesaplama
├── preprocessing.py      # Skewness, log dönüşümü, standardizasyon
├── clustering.py         # k-Means, Elbow, Silhouette analizi
├── utils.py              # Görselleştirme fonksiyonları
├── requirements.txt      # Gerekli Python kütüphaneleri
├── README.txt            # Bu dosya
├── dataset/
│   └── online_retail_II.xlsx  # Veri seti
└── results/
    ├── metrics.txt                  # Detaylı sonuçlar
    ├── rfm_data.csv                 # RFM verileri
    ├── rfm_histograms_original.png  # Orijinal RFM histogramları
    ├── rfm_histograms_log.png       # Log dönüşümlü histogramlar
    ├── elbow_plot.png               # Elbow metodu grafiği
    ├── silhouette_plot.png          # Silhouette analizi grafiği
    ├── combined_analysis.png        # Birleşik analiz grafiği
    ├── pca_clusters.png             # PCA ile küme görselleştirmesi
    ├── cluster_characteristics.png  # Küme özellikleri
    ├── clustering_analysis.csv      # k değerleri analizi
    ├── customer_segments.csv        # Müşteri segmentleri
    └── cluster_interpretation.csv   # Küme yorumları
```

## İşlem Adımları

### 1. Özellik Mühendisliği (RFM Hesaplama)
- **Recency**: Son işlemden bu yana geçen gün sayısı
- **Frequency**: Toplam benzersiz fatura sayısı
- **Monetary**: Toplam harcama tutarı

### 2. Dağılım Kontrolü ve Düzenleme
- Fisher-Pearson skewness katsayısı hesaplanır
- g1 > 0 ise sağa çarpık, g1 < 0 ise sola çarpık
- 2 veya daha fazla özellik çarpık ise log(1+x) dönüşümü uygulanır
- Z-score standardizasyonu uygulanır

### 3. k-Means ile Bölütleme
- k = 2 ile 10 arasında test edilir
- Elbow metodu: WCSS (Within-Cluster Sum of Squares) analizi
- Silhouette analizi: Küme kalitesi ölçümü
- En iyi k değeri belirlenir

### 4. Küme Analizi ve Görselleştirme
- PCA ile 3 boyuttan 2 boyuta indirgeme
- Kümelerin 2D scatter plot görselleştirmesi
- Her kümenin RFM özellikleri yorumlanır

## Model Detayları

- **Algoritma:** k-Means (Scikit-learn)
- **Başlatma:** k-means++ (akıllı başlatma)
- **Mesafe Metriği:** Öklid mesafesi
- **n_init:** 10 (farklı başlangıç noktalarıyla çalıştırma)
- **max_iter:** 300
- **Boyut İndirgeme:** PCA (2 bileşen)

## Değerlendirme Metrikleri

- **Inertia (WCSS):** Küme içi kare toplamı (düşük = daha iyi)
- **Silhouette Score:** Küme kalitesi (-1 ile 1 arası, yüksek = daha iyi)
- **Explained Variance Ratio:** PCA'nın açıkladığı varyans oranı

## Notlar

- Tüm işlemler için Scikit-learn kütüphanesi kullanılmıştır
- Görselleştirmeler Matplotlib ve Seaborn ile oluşturulmuştur
- Sonuçlar `results/` klasörüne otomatik kaydedilir
- Kod modüler yapıda organize edilmiştir

## Yazar
Mohammed Izedin Mohammed STUDENT_ID_REMOVED
Makine Öğrenmesi Dersi - 3. Ödev (Proje)
2025-2026 Güz Yarıyılı
