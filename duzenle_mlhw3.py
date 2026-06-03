import os
import shutil

# 1. Yeni modern klasör hiyerarşisini kur
folders = ['data', 'src', 'reports', 'reports/figures']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 2. Dosya taşıma ve isimlendirme haritası
move_map = {
    # Kodlar -> src/
    'clustering.py': 'src/clustering.py',
    'dataset.py': 'src/dataset.py',
    'eval.py': 'src/eval.py',
    'generate_report.py': 'src/generate_report.py',
    'preprocessing.py': 'src/preprocessing.py',
    'train.py': 'src/train.py',
    'utils.py': 'src/utils.py',
    
    # Raporlar -> reports/
    'Mohammed Izedin Mohammed k-Means Müşteri Segmentasyonu Raporu.pdf': 'reports/customer_segmentation_report.pdf',
    'report.html': 'reports/report_v1.html',
    'report2.html': 'reports/report_v2.html',
    
    # Veri Seti -> data/
    'dataset/online_retail_II.xlsx': 'data/online_retail_II.xlsx'
}

# 3. Dosyaları güvenli bir şekilde taşı
for src, dst in move_map.items():
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"✅ Taşındı: {src} -> {dst}")

# 4. Mevcut grafik çıktılarını reports/figures altına taşı
results_dir = 'results'
if os.path.exists(results_dir):
    for item in os.listdir(results_dir):
        src_item = os.path.join(results_dir, item)
        dst_item = os.path.join('reports/figures', item)
        shutil.move(src_item, dst_item)
    os.rmdir(results_dir)
    print("✅ 'results' klasöründeki tüm grafikler 'reports/figures/' altına taşındı.")

# Eski boş veri klasörünü temizle
if os.path.exists('dataset') and not os.listdir('dataset'):
    os.rmdir('dataset')

print("\n🎉 MLHW3 projesi başarıyla modernize edildi!")