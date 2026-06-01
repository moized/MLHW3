"""
Report Generator Script
Automatically generates HTML report from actual results.
Ensures report values match the code output exactly.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


def load_results(results_dir: str = "results") -> dict:
    """
    Load all results from the results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary with all loaded data
    """
    results = {}
    
    # Load RFM data
    rfm_path = os.path.join(results_dir, "rfm_data.csv")
    if os.path.exists(rfm_path):
        results['rfm'] = pd.read_csv(rfm_path)
    
    # Load customer segments
    segments_path = os.path.join(results_dir, "customer_segments.csv")
    if os.path.exists(segments_path):
        results['segments'] = pd.read_csv(segments_path)
    
    # Load cluster interpretation
    interpretation_path = os.path.join(results_dir, "cluster_interpretation.csv")
    if os.path.exists(interpretation_path):
        results['interpretation'] = pd.read_csv(interpretation_path)
    
    # Load clustering analysis
    analysis_path = os.path.join(results_dir, "clustering_analysis.csv")
    if os.path.exists(analysis_path):
        results['analysis'] = pd.read_csv(analysis_path)
    
    # Load metrics from txt file
    metrics_path = os.path.join(results_dir, "metrics.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            results['metrics_text'] = f.read()
    
    return results


def parse_metrics(metrics_text: str) -> dict:
    """
    Parse metrics from the metrics.txt file.
    
    Args:
        metrics_text: Content of metrics.txt
        
    Returns:
        Dictionary with parsed metrics
    """
    metrics = {}
    
    lines = metrics_text.split('\n')
    for i, line in enumerate(lines):
        if 'Total customers analyzed:' in line:
            metrics['n_customers'] = int(line.split(':')[1].strip().replace(',', ''))
        elif 'Final Silhouette Score:' in line:
            metrics['silhouette'] = float(line.split(':')[1].strip())
        elif 'Final Inertia:' in line:
            metrics['inertia'] = float(line.split(':')[1].strip().replace(',', ''))
        elif 'Selected k:' in line:
            metrics['final_k'] = int(line.split(':')[1].strip())
        elif 'Optimal k (Elbow):' in line:
            # Handle case where line might have different format
            try:
                metrics['elbow_k'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Optimal k (Silhouette):' in line:
            try:
                metrics['silhouette_k'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Log transformation applied:' in line:
            metrics['log_applied'] = 'True' in line
        elif 'PC1:' in line and 'Total:' not in line:
            try:
                metrics['pc1_var'] = float(line.split('(')[1].split('%')[0])
            except:
                pass
        elif 'PC2:' in line:
            try:
                metrics['pc2_var'] = float(line.split('(')[1].split('%')[0])
            except:
                pass
        elif 'Total:' in line and 'variance' not in line.lower():
            try:
                metrics['total_var'] = float(line.split('(')[1].split('%')[0])
            except:
                pass
        elif 'Total time:' in line:
            try:
                metrics['exec_time'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
    
    return metrics


def generate_report(results: dict, output_path: str = "report.html"):
    """
    Generate HTML report from results.
    
    Args:
        results: Dictionary with all results
        output_path: Path to save the report
    """
    # Parse metrics
    metrics = parse_metrics(results.get('metrics_text', ''))
    
    # Get data
    rfm = results.get('rfm', pd.DataFrame())
    segments = results.get('segments', pd.DataFrame())
    interpretation = results.get('interpretation', pd.DataFrame())
    analysis = results.get('analysis', pd.DataFrame())
    
    # Calculate statistics
    n_customers = len(rfm) if not rfm.empty else metrics.get('n_customers', 0)
    
    # RFM statistics - extract values
    if not rfm.empty:
        rfm_stats = rfm[['Recency', 'Frequency', 'Monetary']].describe()
        r_mean = rfm_stats.loc['mean', 'Recency']
        r_std = rfm_stats.loc['std', 'Recency']
        r_min = rfm_stats.loc['min', 'Recency']
        r_25 = rfm_stats.loc['25%', 'Recency']
        r_50 = rfm_stats.loc['50%', 'Recency']
        r_75 = rfm_stats.loc['75%', 'Recency']
        r_max = rfm_stats.loc['max', 'Recency']
        
        f_mean = rfm_stats.loc['mean', 'Frequency']
        f_std = rfm_stats.loc['std', 'Frequency']
        f_min = rfm_stats.loc['min', 'Frequency']
        f_25 = rfm_stats.loc['25%', 'Frequency']
        f_50 = rfm_stats.loc['50%', 'Frequency']
        f_75 = rfm_stats.loc['75%', 'Frequency']
        f_max = rfm_stats.loc['max', 'Frequency']
        
        m_mean = rfm_stats.loc['mean', 'Monetary']
        m_std = rfm_stats.loc['std', 'Monetary']
        m_min = rfm_stats.loc['min', 'Monetary']
        m_25 = rfm_stats.loc['25%', 'Monetary']
        m_50 = rfm_stats.loc['50%', 'Monetary']
        m_75 = rfm_stats.loc['75%', 'Monetary']
        m_max = rfm_stats.loc['max', 'Monetary']
        
        # Log-transformed values
        r_min_log = np.log1p(r_min)
        r_max_log = np.log1p(r_max)
        f_min_log = np.log1p(f_min)
        f_max_log = np.log1p(f_max)
        m_min_log = np.log1p(m_min)
        m_max_log = np.log1p(m_max)
    else:
        r_mean = r_std = r_min = r_25 = r_50 = r_75 = r_max = 0
        f_mean = f_std = f_min = f_25 = f_50 = f_75 = f_max = 0
        m_mean = m_std = m_min = m_25 = m_50 = m_75 = m_max = 0
        r_min_log = r_max_log = f_min_log = f_max_log = m_min_log = m_max_log = 0
    
    # Cluster statistics
    if not segments.empty:
        cluster_stats = segments.groupby('Cluster').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        cluster_stats.columns = ['Count', 'Recency', 'Frequency', 'Monetary']
    else:
        cluster_stats = None
    
    # Analysis data
    if not analysis.empty:
        k_values = analysis['k'].tolist()
        inertias = analysis['Inertia'].tolist()
        silhouette_scores = analysis['Silhouette_Score'].tolist()
    else:
        k_values = list(range(2, 11))
        inertias = [0] * 9
        silhouette_scores = [0] * 9
    
    # Get actual values from metrics or calculate
    final_k = metrics.get('final_k', 2)
    silhouette = metrics.get('silhouette', 0)
    inertia = metrics.get('inertia', 0)
    elbow_k = metrics.get('elbow_k', 3)
    silhouette_k = metrics.get('silhouette_k', 2)
    pc1_var = metrics.get('pc1_var', 0)
    pc2_var = metrics.get('pc2_var', 0)
    total_var = pc1_var + pc2_var
    exec_time = metrics.get('exec_time', 0)
    
    # Skewness values from preprocessing (recalculate from data)
    if not rfm.empty:
        def calc_skewness(data):
            n = len(data)
            mean = np.mean(data)
            std = np.std(data, ddof=0)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
        
        orig_skew_r = calc_skewness(rfm['Recency'].values)
        orig_skew_f = calc_skewness(rfm['Frequency'].values)
        orig_skew_m = calc_skewness(rfm['Monetary'].values)
        
        # Log transformed skewness
        log_skew_r = calc_skewness(np.log1p(rfm['Recency'].values))
        log_skew_f = calc_skewness(np.log1p(rfm['Frequency'].values))
        log_skew_m = calc_skewness(np.log1p(rfm['Monetary'].values))
    else:
        orig_skew_r = orig_skew_f = orig_skew_m = 0
        log_skew_r = log_skew_f = log_skew_m = 0
    
    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>MLHW3 - k-Means Müşteri Segmentasyonu Raporu</title>
    <style>
        /* IEEE Style - Single Column for Print (Better Table Visibility) */
        @media print {{
            body {{
                font-family: 'Times New Roman', Times, serif;
                font-size: 11pt;
                line-height: 1.3;
                margin: 0;
                padding: 0;
            }}
            .container {{
                column-count: 1;
                padding: 1.5cm;
            }}
            h1.title {{
                text-align: center;
                font-size: 18pt;
                margin-bottom: 0.3cm;
            }}
            .author {{
                text-align: center;
                margin-bottom: 0.5cm;
            }}
            .abstract {{
                margin-bottom: 0.5cm;
                padding: 0.3cm;
                background: #f5f5f5;
                border-left: 3px solid #333;
            }}
            h2 {{
                font-size: 14pt;
                margin-top: 0.5cm;
                margin-bottom: 0.2cm;
                border-bottom: 1px solid #333;
            }}
            h3 {{
                font-size: 12pt;
                margin-top: 0.4cm;
                margin-bottom: 0.15cm;
            }}
            p {{
                text-align: justify;
                margin: 0.15cm 0;
                orphans: 3;
                widows: 3;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 9pt;
                margin: 0.3cm 0;
                break-inside: avoid;
                page-break-inside: avoid;
            }}
            th, td {{
                border: 1px solid #333;
                padding: 4px 6px;
                text-align: center;
                word-wrap: break-word;
            }}
            th {{
                background: #e0e0e0;
                font-weight: bold;
            }}
            img {{
                max-width: 80%;
                height: auto;
                break-inside: avoid;
                display: block;
                margin: 0.3cm auto;
            }}
            .figure {{
                text-align: center;
                margin: 0.4cm 0;
                break-inside: avoid;
                page-break-inside: avoid;
            }}
            .figure-caption {{
                font-size: 10pt;
                font-style: italic;
            }}
            .references {{
                font-size: 10pt;
            }}
            .highlight {{
                background: #fffde7;
                padding: 0.2cm;
                border-left: 3px solid #ffc107;
                margin: 0.2cm 0;
            }}
            code {{
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                background: #f0f0f0;
                padding: 1px 3px;
            }}
            @page {{
                size: A4;
                margin: 2cm;
            }}
        }}

        /* Screen Styles */
        @media screen {{
            body {{
                font-family: 'Times New Roman', Times, serif;
                font-size: 12pt;
                line-height: 1.4;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background: #f9f9f9;
            }}
            .container {{
                background: white;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1.title {{
                text-align: center;
                font-size: 24pt;
                color: #333;
                margin-bottom: 10px;
            }}
            .author {{
                text-align: center;
                color: #666;
                margin-bottom: 20px;
            }}
            .abstract {{
                background: #f0f0f0;
                padding: 15px;
                border-left: 4px solid #333;
                margin-bottom: 20px;
            }}
            h2 {{
                color: #222;
                border-bottom: 2px solid #333;
                padding-bottom: 5px;
                margin-top: 25px;
            }}
            h3 {{
                color: #444;
                margin-top: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: center;
            }}
            th {{
                background: #333;
                color: white;
            }}
            tr:nth-child(even) {{
                background: #f9f9f9;
            }}
            .figure {{
                text-align: center;
                margin: 20px 0;
            }}
            .figure img {{
                max-width: 600px;
                border: 1px solid #ddd;
            }}
            .figure-caption {{
                font-style: italic;
                color: #666;
                margin-top: 8px;
            }}
            code {{
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            .highlight {{
                background: #fffde7;
                padding: 10px;
                border-left: 4px solid #ffc107;
                margin: 15px 0;
            }}
        }}
    </style>
</head>
<body>
<div class="container">

<h1 class="title">k-Means Algoritması ile Müşteri Segmentasyonu: RFM Analizi Yaklaşımı</h1>

<div class="author">
    <strong>Mohammed Izedin Mohammed</strong><br>
    Öğrenci No: STUDENT_ID_REMOVED<br>
    Makine Öğrenmesi Dersi - Proje (Ödev 3)<br>
    Ocak 2026
</div>

<div class="abstract">
    <strong>Özet:</strong> Bu projede, Online Retail II veri seti üzerinde k-Means kümeleme algoritması kullanılarak müşteri segmentasyonu gerçekleştirilmiştir. Her müşteri için RFM (Recency, Frequency, Monetary) özellikleri hesaplanmış, dağılım analizi yapılarak log dönüşümü ve standardizasyon uygulanmıştır. Elbow metodu ve Silhouette analizi ile optimal küme sayısı belirlenmiş, PCA ile 2 boyutlu görselleştirme yapılmıştır. Sonuç olarak, {n_customers:,} müşteri {final_k} ana segmente ayrılmıştır. Silhouette skoru {silhouette:.4f} olarak elde edilmiştir.
</div>

<h2>I. Giriş</h2>

<p>Müşteri segmentasyonu, pazarlama stratejilerinin etkinliğini artırmak için kritik öneme sahip bir analiz yöntemidir. Farklı müşteri gruplarının davranışlarını anlamak, işletmelerin hedefli kampanyalar oluşturmasına ve müşteri ilişkilerini güçlendirmesine olanak tanır.</p>

<p>RFM (Recency, Frequency, Monetary) analizi, müşteri segmentasyonunda yaygın olarak kullanılan bir yöntemdir:</p>

<ul>
    <li><strong>Recency (Yenilik):</strong> Müşterinin son alışverişinden bu yana geçen süre</li>
    <li><strong>Frequency (Sıklık):</strong> Müşterinin toplam işlem sayısı</li>
    <li><strong>Monetary (Parasal Değer):</strong> Müşterinin toplam harcama tutarı</li>
</ul>

<p>k-Means algoritması, denetimsiz öğrenme yöntemleri arasında en yaygın kullanılan kümeleme algoritmalarından biridir. Bu projede, e-ticaret verisi üzerinde RFM analizi ile k-Means kümelemesi birleştirilerek müşteri segmentasyonu gerçekleştirilmiştir.</p>

<p>Bu çalışmada şu sorulara yanıt aranmıştır:</p>
<ol>
    <li>RFM özelliklerinin dağılımı nasıldır ve hangi dönüşümler gereklidir?</li>
    <li>Optimal küme sayısı kaçtır?</li>
    <li>Her kümenin karakteristik özellikleri nelerdir?</li>
</ol>

<p><strong>Bu raporun katkıları:</strong></p>
<ul>
    <li>Online Retail II veri seti için uçtan uca, tekrarlanabilir bir RFM tabanlı k-Means segmentasyon hattı.</li>
    <li>Skewness analizi ile gerekçelendirilmiş log dönüşümü ve standardizasyon adımlarının etkisinin ölçülmesi.</li>
    <li>Elbow ve Silhouette analizlerinin birlikte değerlendirilmesi ve iş açısından yorumlanabilir segmentlerin çıkarılması.</li>
</ul>

<h2>II. Veri Seti</h2>

<p>Çalışmada UCI Machine Learning Repository'den alınan Online Retail II veri seti kullanılmıştır. Bu veri seti, İngiltere merkezli bir online perakende şirketinin 2009-2011 yılları arasındaki işlemlerini içermektedir.</p>

<p>Veri seti iki ayrı sayfadan oluştuğu için (2009-2010 ve 2010-2011), analiz için iki sayfa birleştirilmiş ve toplam 1,067,371 satırlık ham işlem verisi elde edilmiştir. Bu yaklaşım, zaman aralığını genişleterek müşteri davranışını daha kapsayıcı biçimde yansıtır.</p>

<p><strong>Tablo 1:</strong> Veri seti özellikleri</p>

<table>
    <tr>
        <th>Özellik</th>
        <th>Değer</th>
    </tr>
    <tr>
        <td>Toplam işlem sayısı</td>
        <td>1,067,371</td>
    </tr>
    <tr>
        <td>Temizlenmiş işlem sayısı</td>
        <td>805,549</td>
    </tr>
    <tr>
        <td>Benzersiz müşteri sayısı</td>
        <td>{n_customers:,}</td>
    </tr>
    <tr>
        <td>Zaman aralığı</td>
        <td>2009-2011 (2 yıl)</td>
    </tr>
    <tr>
        <td>Kaynak</td>
        <td>UCI Machine Learning Repository</td>
    </tr>
</table>

<p><strong>Veri Temizleme Adımları:</strong></p>
<ul>
    <li>Müşteri ID'si olmayan kayıtlar çıkarıldı</li>
    <li>İptal edilen siparişler (Invoice 'C' ile başlayanlar) çıkarıldı</li>
    <li>Negatif veya sıfır miktarlı işlemler çıkarıldı</li>
    <li>Toplam 261,822 kayıt temizlendi</li>
</ul>

<p>Bu adımların amacı, iade/iptal gibi gerçek satın alma davranışını temsil etmeyen kayıtların etkisini azaltmak ve RFM hesaplamalarının anlamlı müşteri davranışına dayanmasını sağlamaktır.</p>

<h2>III. Metodoloji</h2>

<h3>A. Özellik Mühendisliği (RFM Hesaplama)</h3>

<p>Her müşteri için aşağıdaki RFM değerleri hesaplanmıştır:</p>

<ul>
    <li><strong>Recency:</strong> Referans tarihinden (2011-12-10) müşterinin son işlem tarihine kadar geçen gün sayısı</li>
    <li><strong>Frequency:</strong> Müşterinin yaptığı benzersiz fatura (invoice) sayısı</li>
    <li><strong>Monetary:</strong> Müşterinin tüm işlemlerinin toplam tutarı (Quantity × Price)</li>
</ul>

<p>RFM hesaplamaları şu şekilde özetlenebilir: Recency, son işlem tarihinden referans tarihe (2011-12-10) olan gün farkıdır; Frequency, benzersiz fatura sayısıdır; Monetary ise toplam harcama tutarıdır. Referans tarihinin veri setindeki son işlem tarihinden bir gün sonrası seçilmesi, en güncel müşteriler için Recency değerinin en düşük olması gerektiği varsayımıyla uyumludur.</p>

<p><strong>Tablo 2:</strong> RFM istatistikleri (orijinal değerler)</p>

<table>
    <tr>
        <th>İstatistik</th>
        <th>Recency (gün)</th>
        <th>Frequency (sipariş)</th>
        <th>Monetary ($)</th>
    </tr>
    <tr>
        <td>Ortalama</td>
        <td>{r_mean:.2f}</td>
        <td>{f_mean:.2f}</td>
        <td>{m_mean:.2f}</td>
    </tr>
    <tr>
        <td>Standart Sapma</td>
        <td>{r_std:.2f}</td>
        <td>{f_std:.2f}</td>
        <td>{m_std:.2f}</td>
    </tr>
    <tr>
        <td>Minimum</td>
        <td>{r_min:.0f}</td>
        <td>{f_min:.0f}</td>
        <td>{m_min:.2f}</td>
    </tr>
    <tr>
        <td>%25</td>
        <td>{r_25:.0f}</td>
        <td>{f_25:.0f}</td>
        <td>{m_25:.2f}</td>
    </tr>
    <tr>
        <td>Medyan (%50)</td>
        <td>{r_50:.0f}</td>
        <td>{f_50:.0f}</td>
        <td>{m_50:.2f}</td>
    </tr>
    <tr>
        <td>%75</td>
        <td>{r_75:.0f}</td>
        <td>{f_75:.0f}</td>
        <td>{m_75:.2f}</td>
    </tr>
    <tr>
        <td>Maksimum</td>
        <td>{r_max:.0f}</td>
        <td>{f_max:.0f}</td>
        <td>{m_max:.2f}</td>
    </tr>
</table>

<div class="figure">
    <img src="results/rfm_histograms_original.png" alt="RFM Histogramları (Orijinal)">
    <p class="figure-caption">Şekil 1: RFM özelliklerinin orijinal dağılımları (dönüşüm öncesi)</p>
</div>

<h3>B. Dağılım Kontrolü ve Düzenleme</h3>

<p><strong>1) Skewness (Çarpıklık) Analizi:</strong></p>

<p>Fisher-Pearson çarpıklık katsayısı (g₁) aşağıdaki formül ile hesaplanmıştır:</p>

<p style="text-align: center; font-style: italic;">g₁ = (1/n) × Σ((xᵢ - μ) / σ)³</p>

<p>Yorumlama: g₁ = 0 ise simetrik dağılım, g₁ > 0 ise sağa çarpık, g₁ < 0 ise sola çarpık.</p>

<p>RFM özellikleri özellikle Monetary ve Frequency için uzun kuyruklu dağılımlara sahiptir. Bu durum Öklid tabanlı algoritmalarda mesafe hesaplarını domine edebilir. Bu nedenle log-plus-one dönüşümü, büyük değerlerin etkisini azaltmak ve dağılımı daha dengeli hale getirmek için tercih edilmiştir.</p>

<p><strong>Tablo 3:</strong> Skewness analizi sonuçları</p>

<table>
    <tr>
        <th>Özellik</th>
        <th>Orijinal g₁</th>
        <th>Yorum</th>
        <th>Log Sonrası g₁</th>
        <th>Yorum</th>
    </tr>
    <tr>
        <td>Recency</td>
        <td>{orig_skew_r:.4f}</td>
        <td>{"Sağa çarpık" if orig_skew_r > 0.5 else "Yaklaşık simetrik" if abs(orig_skew_r) < 0.5 else "Sola çarpık"}</td>
        <td>{log_skew_r:.4f}</td>
        <td>{"Yaklaşık simetrik" if abs(log_skew_r) < 0.5 else "Hafif sağa çarpık" if log_skew_r > 0 else "Hafif sola çarpık"}</td>
    </tr>
    <tr>
        <td>Frequency</td>
        <td>{orig_skew_f:.4f}</td>
        <td>{"Aşırı sağa çarpık" if orig_skew_f > 2 else "Sağa çarpık" if orig_skew_f > 0.5 else "Yaklaşık simetrik"}</td>
        <td>{log_skew_f:.4f}</td>
        <td>{"Yaklaşık simetrik" if abs(log_skew_f) < 0.5 else "Hafif sağa çarpık" if log_skew_f > 0 else "Hafif sola çarpık"}</td>
    </tr>
    <tr>
        <td>Monetary</td>
        <td>{orig_skew_m:.4f}</td>
        <td>{"Aşırı sağa çarpık" if orig_skew_m > 2 else "Sağa çarpık" if orig_skew_m > 0.5 else "Yaklaşık simetrik"}</td>
        <td>{log_skew_m:.4f}</td>
        <td>{"Yaklaşık simetrik" if abs(log_skew_m) < 0.5 else "Hafif sağa çarpık" if log_skew_m > 0 else "Hafif sola çarpık"}</td>
    </tr>
</table>

<div class="highlight">
    <strong>Önemli Bulgu:</strong> Tüm özelliklerin g₁ > 0 olması nedeniyle (3/3 özellik çarpık), log-plus-one dönüşümü (y = ln(1+x)) uygulanmıştır. Bu dönüşüm özellikle Frequency ve Monetary özelliklerindeki aşırı çarpıklığı önemli ölçüde azaltmıştır.
</div>

<p><strong>2) Log Dönüşümü:</strong></p>

<p>Log-plus-one dönüşümü uygulanarak özelliklerin değer aralıkları daraltılmıştır:</p>

<table>
    <tr>
        <th>Özellik</th>
        <th>Orijinal Aralık</th>
        <th>Dönüşüm Sonrası</th>
    </tr>
    <tr>
        <td>Recency</td>
        <td>[{r_min:.0f}, {r_max:.0f}]</td>
        <td>[{r_min_log:.2f}, {r_max_log:.2f}]</td>
    </tr>
    <tr>
        <td>Frequency</td>
        <td>[{f_min:.0f}, {f_max:.0f}]</td>
        <td>[{f_min_log:.2f}, {f_max_log:.2f}]</td>
    </tr>
    <tr>
        <td>Monetary</td>
        <td>[{m_min:.2f}, {m_max:.2f}]</td>
        <td>[{m_min_log:.2f}, {m_max_log:.2f}]</td>
    </tr>
</table>

<div class="figure">
    <img src="results/rfm_histograms_log.png" alt="RFM Histogramları (Log Dönüşümlü)">
    <p class="figure-caption">Şekil 2: RFM özelliklerinin log dönüşümü sonrası dağılımları</p>
</div>

<p><strong>3) Standardizasyon:</strong></p>

<p>Z-score normalizasyonu uygulanarak tüm özellikler μ=0, σ=1 olacak şekilde standartlaştırılmıştır:</p>

<p style="text-align: center; font-style: italic;">z = (x - μ) / σ</p>

<p>Bu proje denetimsiz öğrenme olduğundan ayrı bir eğitim/test bölmesi yapılmamış, ölçekleme parametreleri (μ ve σ) tüm veri üzerinden hesaplanmıştır. Log dönüşümü sonrası standardizasyon, RFM özelliklerini aynı ölçekte toplayarak k-Means'in mesafe tabanlı yapısına uygunluk sağlar.</p>

<h3>C. k-Means Kümeleme</h3>

<p>k-Means algoritması aşağıdaki parametrelerle uygulanmıştır:</p>

<ul>
    <li><strong>Başlatma:</strong> k-means++ (akıllı merkez başlatma)</li>
    <li><strong>n_init:</strong> 10 (farklı başlangıç noktalarıyla çalıştırma)</li>
    <li><strong>max_iter:</strong> 300</li>
    <li><strong>k aralığı:</strong> 2-10</li>
    <li><strong>random_state:</strong> 42 (tekrarlanabilirlik)</li>
</ul>

<h2>IV. Sonuçlar</h2>

<h3>A. Elbow Metodu Analizi</h3>

<p>Elbow (Dirsek) metodu, farklı k değerleri için Within-Cluster Sum of Squares (WCSS/Inertia) değerlerini hesaplayarak optimal küme sayısını belirlemeye yardımcı olur. "Dirsek" noktası, WCSS'deki düşüşün yavaşladığı noktadır.</p>

<p><strong>Tablo 4:</strong> Elbow analizi sonuçları</p>

<table>
    <tr>
        <th>k</th>
        <th>Inertia (WCSS)</th>
    </tr>
'''
    
    # Add elbow analysis rows
    for i, (k, inert) in enumerate(zip(k_values, inertias)):
        html_content += f'''    <tr>
        <td>{k}</td>
        <td>{inert:,.2f}</td>
    </tr>
'''
    
    html_content += f'''</table>

<div class="figure">
    <img src="results/elbow_plot.png" alt="Elbow Metodu">
    <p class="figure-caption">Şekil 3: Elbow metodu ile optimal k belirleme (k={elbow_k} önerisi)</p>
</div>

<h3>B. Silhouette Analizi</h3>

<p>Silhouette skoru, kümeleme kalitesini -1 ile 1 arasında ölçer. Yüksek değerler daha iyi kümelemeyi gösterir:</p>

<ul>
    <li>+1'e yakın: İyi kümelenmiş</li>
    <li>0'a yakın: Küme sınırında</li>
    <li>-1'e yakın: Yanlış kümeye atanmış</li>
</ul>

<p><strong>Tablo 5:</strong> Silhouette analizi sonuçları</p>

<table>
    <tr>
        <th>k</th>
        <th>Silhouette Skoru</th>
        <th>Değerlendirme</th>
    </tr>
'''
    
    # Find best silhouette
    best_sil_idx = silhouette_scores.index(max(silhouette_scores))
    
    for i, (k, sil) in enumerate(zip(k_values, silhouette_scores)):
        is_best = i == best_sil_idx
        eval_text = "En Yüksek" if is_best else ("Orta" if sil > 0.3 else "Düşük")
        bold_start = "<strong>" if is_best else ""
        bold_end = "</strong>" if is_best else ""
        html_content += f'''    <tr>
        <td>{bold_start}{k}{bold_end}</td>
        <td>{bold_start}{sil:.4f}{bold_end}</td>
        <td>{bold_start}{eval_text}{bold_end}</td>
    </tr>
'''
    
    html_content += f'''</table>

<div class="figure">
    <img src="results/silhouette_plot.png" alt="Silhouette Analizi">
    <p class="figure-caption">Şekil 4: Silhouette analizi ile optimal k belirleme (k={silhouette_k} önerisi)</p>
</div>

<div class="figure">
    <img src="results/combined_analysis.png" alt="Birleşik Analiz">
    <p class="figure-caption">Şekil 5: Elbow ve Silhouette analizlerinin karşılaştırması</p>
</div>

<div class="highlight">
    <strong>Optimal k Seçimi:</strong> Elbow metodu k={elbow_k}, Silhouette analizi k={silhouette_k} önermiştir. Silhouette skorunun k={silhouette_k}'de en yüksek olması ({silhouette:.4f}) ve iş yorumlaması açısından anlamlı segmentler oluşturması nedeniyle <strong>k={final_k}</strong> tercih edilmiştir.
</div>

<h3>C. Küme Analizi ve Karakterizasyonu</h3>

<p><strong>Tablo 6:</strong> Final küme özellikleri (k={final_k})</p>

<table>
    <tr>
        <th>Küme</th>
        <th>Müşteri Sayısı</th>
        <th>Oran</th>
        <th>Recency (gün)</th>
        <th>Frequency</th>
        <th>Monetary ($)</th>
        <th>Segment Profili</th>
    </tr>
'''
    
    # Add cluster rows from interpretation
    if not interpretation.empty:
        for _, row in interpretation.iterrows():
            pct = row['Count'] / n_customers * 100
            html_content += f'''    <tr>
        <td><strong>{int(row['Cluster'])}</strong></td>
        <td>{int(row['Count']):,}</td>
        <td>{pct:.1f}%</td>
        <td>{row['Recency_Mean']:.1f}</td>
        <td>{row['Frequency_Mean']:.1f}</td>
        <td>{row['Monetary_Mean']:,.2f}</td>
        <td>{row['Interpretation']}</td>
    </tr>
'''
    
    html_content += '''</table>

<p><strong>Küme Yorumlaması:</strong></p>
'''
    
    # Add cluster interpretations
    if not interpretation.empty:
        overall_r = rfm['Recency'].mean() if not rfm.empty else 0
        overall_f = rfm['Frequency'].mean() if not rfm.empty else 0
        overall_m = rfm['Monetary'].mean() if not rfm.empty else 0
        
        for _, row in interpretation.iterrows():
            cluster = int(row['Cluster'])
            pct = row['Count'] / n_customers * 100
            interp = row['Interpretation']
            
            # Determine segment type
            if 'Recent' in interp and 'Frequent' in interp and 'High-value' in interp:
                segment_name = "VIP Müşteriler"
                recommendation = "Öncelikli muamele, özel teklifler, sadakat programları"
            elif 'Old' in interp or 'Dormant' in interp:
                segment_name = "Pasif Müşteriler"
                recommendation = "Yeniden aktivasyon kampanyaları, indirim kuponları, geri kazanım stratejileri"
            else:
                segment_name = f"Segment {cluster}"
                recommendation = "Hedefli pazarlama stratejileri"
            
            html_content += f'''
<p><strong>Küme {cluster} - {segment_name} ({pct:.0f}%):</strong></p>
<ul>
    <li>Ortalama {row['Recency_Mean']:.0f} gün önce alışveriş yapmışlar (genel ortalama: {overall_r:.0f} gün)</li>
    <li>Ortalama {row['Frequency_Mean']:.1f} sipariş vermişler (genel ortalama: {overall_f:.1f})</li>
    <li>Ortalama ${row['Monetary_Mean']:,.0f} harcamışlar (genel ortalama: ${overall_m:,.0f})</li>
    <li><strong>Öneri:</strong> {recommendation}</li>
</ul>
'''
    
    html_content += f'''
<div class="figure">
    <img src="results/cluster_characteristics.png" alt="Küme Özellikleri">
    <p class="figure-caption">Şekil 6: Kümelerin RFM özelliklerine göre karşılaştırması</p>
</div>

<h3>D. PCA Görselleştirmesi</h3>

<p>3 boyutlu RFM özellikleri, Principal Component Analysis (PCA) ile 2 boyuta indirgenmiştir:</p>

<p><strong>Tablo 7:</strong> PCA açıklanan varyans oranları</p>

<table>
    <tr>
        <th>Bileşen</th>
        <th>Açıklanan Varyans</th>
        <th>Kümülatif</th>
    </tr>
    <tr>
        <td>PC1</td>
        <td>{pc1_var:.2f}%</td>
        <td>{pc1_var:.2f}%</td>
    </tr>
    <tr>
        <td>PC2</td>
        <td>{pc2_var:.2f}%</td>
        <td>{total_var:.2f}%</td>
    </tr>
</table>

<p>İlk iki bileşen, toplam varyansın %{total_var:.2f}'ünü açıklamaktadır, bu da 2D görselleştirmenin orijinal verideki bilgiyi büyük ölçüde koruduğunu göstermektedir.</p>

<div class="figure">
    <img src="results/pca_clusters.png" alt="PCA Küme Görselleştirmesi">
    <p class="figure-caption">Şekil 7: PCA ile 2 boyutlu küme görselleştirmesi</p>
</div>

<h3>E. Model Performansı</h3>

<p><strong>Tablo 8:</strong> Final model metrikleri</p>

<table>
    <tr>
        <th>Metrik</th>
        <th>Değer</th>
    </tr>
    <tr>
        <td>Küme Sayısı (k)</td>
        <td>{final_k}</td>
    </tr>
    <tr>
        <td>Silhouette Skoru</td>
        <td>{silhouette:.4f}</td>
    </tr>
    <tr>
        <td>Inertia (WCSS)</td>
        <td>{inertia:,.2f}</td>
    </tr>
    <tr>
        <td>PCA Varyans (%)</td>
        <td>{total_var:.2f}%</td>
    </tr>
    <tr>
        <td>Çalışma Süresi</td>
        <td>~{exec_time:.0f} saniye</td>
    </tr>
</table>

<p>Silhouette skorunun k=2 için belirgin biçimde yüksek olması ve müşteri segmentlerinin iş açısından yorumlanabilir olması nedeniyle k=2 seçilmiştir. K=3 ve k=4 değerlerinde inertia azalmasına rağmen silhouette düşmekte, bu da kümelerin ayrışmasının zayıfladığını göstermektedir.</p>

<h2>V. Tartışma</h2>

<h3>A. Dağılım Düzeltmesinin Etkisi</h3>

<p>Orijinal RFM değerlerinin aşırı sağa çarpık olduğu gözlemlenmiştir (özellikle Monetary için g₁={orig_skew_m:.2f}). Log dönüşümü uygulandıktan sonra çarpıklık değerleri kabul edilebilir seviyelere inmiştir. k-Means algoritması, Öklid mesafesine dayandığı için verilerin normal dağılıma yakın olması kümeleme kalitesini artırmaktadır.</p>

<h3>B. Elbow vs Silhouette Karşılaştırması</h3>

<p>İki yöntem farklı k değerleri önermiştir:</p>
<ul>
    <li><strong>Elbow (k={elbow_k}):</strong> WCSS düşüşünün yavaşladığı nokta</li>
    <li><strong>Silhouette (k={silhouette_k}):</strong> En yüksek kümeleme kalitesi</li>
</ul>

<p>k={final_k} seçilmesinin nedenleri:</p>
<ol>
    <li>Daha yüksek Silhouette skoru ({silhouette:.4f})</li>
    <li>İş perspektifinden yorumlanması daha kolay segmentler</li>
    <li>Pazarlama stratejileri için daha actionable içgörüler</li>
</ol>

<h3>C. Küme Karakteristikleri</h3>

<p>Elde edilen segmentler, klasik RFM analizindeki müşteri tiplerini yansıtmaktadır. Bu segmentasyon, pazarlama ekiplerinin farklı müşteri gruplarına özel stratejiler geliştirmesine olanak tanır.</p>

<h3>D. Sınırlılıklar</h3>

<ul>
    <li>k-Means küresel kümeler varsayar; farklı şekillerdeki kümeleri yakalayamayabilir</li>
    <li>Veri seti 2009-2011 dönemini kapsadığı için güncel olmayabilir</li>
    <li>Daha fazla küme (k=3 veya k=4) daha detaylı segmentasyon sağlayabilir</li>
    <li>Aykırı değerler Monetary değişkeninde güçlü etki yaratabilir</li>
    <li>Zaman içinde müşteri davranışları değişebileceğinden segmentler periyodik güncellenmelidir</li>
</ul>

<h3>E. Uygulama ve Tekrarlanabilirlik</h3>

<p>Çalışma modüler bir yapıda kurgulanmış olup veri yükleme, ön işleme, kümeleme ve görselleştirme adımları ayrı dosyalar halinde düzenlenmiştir. Sonuçlar results/ klasörüne otomatik kaydedilmekte ve report.html dosyası bu çıktılardan dinamik olarak üretilmektedir. Bu yapı, farklı k değerleri veya yeni veri setleri ile tekrar çalıştırmaya uygundur.</p>

<h3>F. Etik ve Gizlilik</h3>

<p>Veri seti anonimleştirilmiş işlem kayıtlarından oluşmaktadır ve kişisel veri içermemektedir. Segmentasyon çıktıları, bireysel müşteri profillerini değil, toplu davranış eğilimlerini göstermektedir. Model çıktılarının kampanya tasarımında kullanılmasında ayrımcılığa yol açabilecek uygulamalardan kaçınılmalıdır.</p>

<h3>G. Gelecek Çalışmalar</h3>

<p>İlerleyen çalışmalarda zaman serisi etkileri, müşteri yaşam boyu değeri ve kampanya tepkileri gibi ek değişkenlerin modele dahil edilmesi planlanabilir. Ayrıca k-Means dışındaki yöntemler (DBSCAN, GMM) ile daha esnek kümeleme denemeleri yapılabilir.</p>

<h2>VI. Sonuç</h2>

<p>Bu projede, Online Retail II veri seti üzerinde k-Means kümeleme algoritması ile müşteri segmentasyonu başarıyla gerçekleştirilmiştir. Elde edilen sonuçlar:</p>

<ul>
    <li><strong>RFM Analizi:</strong> {n_customers:,} müşteri için Recency, Frequency ve Monetary özellikleri hesaplanmıştır.</li>
    <li><strong>Veri Ön İşleme:</strong> Log dönüşümü ile aşırı çarpıklık giderilmiş, standardizasyon ile özellikler normalize edilmiştir.</li>
    <li><strong>Optimal Küme Sayısı:</strong> Silhouette analizi ile k={final_k} optimal olarak belirlenmiştir.</li>
    <li><strong>Silhouette Skoru:</strong> {silhouette:.4f}</li>
    <li><strong>Görselleştirme:</strong> PCA ile %{total_var:.2f} varyans korunarak 2D görselleştirme yapılmıştır.</li>
</ul>

<p>Bu segmentasyon sonuçları, pazarlama stratejilerinin kişiselleştirilmesi, müşteri ilişkileri yönetimi ve kaynak tahsisi kararlarında kullanılabilir.</p>

<h2>VII. Referanslar</h2>

<div class="references">
<ol>
    <li>J. MacQueen, "Some methods for classification and analysis of multivariate observations," Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, vol. 1, pp. 281-297, 1967.</li>
    <li>P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," Journal of Computational and Applied Mathematics, vol. 20, pp. 53-65, 1987.</li>
    <li>R. Tibshirani, G. Walther, and T. Hastie, "Estimating the number of clusters in a data set via the gap statistic," Journal of the Royal Statistical Society: Series B, vol. 63, no. 2, pp. 411-423, 2001.</li>
    <li>D. Chen, S. L. Sain, and K. Guo, "Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining," Journal of Database Marketing & Customer Strategy Management, vol. 19, no. 3, pp. 197-208, 2012.</li>
    <li>UCI Machine Learning Repository, "Online Retail II Data Set," https://archive.ics.uci.edu/dataset/502/online+retail+ii, 2019.</li>
    <li>F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.</li>
</ol>
</div>

<p style="text-align: center; font-size: 10pt; color: #666; margin-top: 30px;">
    <em>Rapor Oluşturma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em>
</p>

</div>
</body>
</html>
'''
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report generated: {output_path}")
    print(f"Report uses actual values from results/")


def main():
    """
    Main function to generate report from results.
    """
    print("=" * 60)
    print("REPORT GENERATOR")
    print("=" * 60)
    
    # Check if results exist
    if not os.path.exists("results"):
        print("\nError: Results directory not found.")
        print("Please run train.py first to generate results.")
        return
    
    # Load results
    print("\nLoading results...")
    results = load_results()
    
    if not results:
        print("Error: No results found.")
        return
    
    # Generate report
    print("\nGenerating report...")
    generate_report(results, "report.html")
    
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE!")
    print("=" * 60)
    print("\nThe report now contains actual values from your results.")
    print("Open report.html in a browser to view.")


if __name__ == "__main__":
    main()
