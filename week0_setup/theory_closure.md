# Week 0 — Kapanış Dökümanı

**NovaDev v1.0 - Self-Assessment & Final Check**

> "Kendi cümlelerimle, kendi anlayışımla. Week 0'ı gerçekten özümsedim mi?"

---

## 🎯 Bu Döküman Hakkında

**Amaç:** Week 0 sonunda temel kavramların **kendi cümlelerimle** netleştiğini göstermek

**Format:** 
- ✅ Son kontrol listesi (soru-cevap)
- 🧠 Mini-ödev çözümleri
- 📊 Tablo formatında karşılaştırmalar
- 🎓 Kendime notlar

**Kullanım:**
- Week 0 bittiğinde oku
- Cevapları kendi kelimelerinle yaz
- Boşluk varsa geri dön ilgili theory'ye
- Week 1'e geçmeden önce rahat hisset

---

## ✅ Son Kontrol Listesi — Cevaplarım

### 1️⃣ Train / Validation / Test (Fark ve Amaç)

#### Train Set
```
Amaç: Model burada ÖĞRENİR
Ne yapar: Ağırlıklar bu veri üzerinde optimize edilir
θ ← θ - η∇L (train data)

Analoji: Ders çalışma dönemi
```

#### Validation Set
```
Amaç: AYAR SEÇ + Erken Durdur
Ne yapar: 
  - Hiperparametre seçimi (LR, L2, epoch, batch)
  - Early stopping sinyali
  - Model bu veriyi ÖĞRENMEz, sadece performans ölçer

Analoji: Ara sınav (çalışma stratejini ayarla)
```

#### Test Set
```
Amaç: FİNAL DEĞERLENDİRME
Ne yapar: Gerçek dünya performansını TEK SEFERDE göster
ASLA: Test'e bakıp ayar yapma!

Analoji: Final sınavı (bir kez, dokunma)
```

#### Neden Ayırıyoruz?

**Ana Sebep:** Ezberi (overfit) yakalamak ve gerçek performansı ölçmek

```
Senario 1: Ayrım yok
  → Model train'i ezberler
  → Gerçek dünyada felaket

Senario 2: Sadece train/test
  → Test'e bakarak ayar yaparsın
  → Test'i "ezberlersin" (indirect)

Senario 3: Train/Val/Test ✓
  → Val ile ayar yap
  → Test sadece final rapor
  → Dürüst değerlendirme
```

---

### 2️⃣ Loss Seçimi (Ne Zaman Hangisi?)

#### Regresyon Loss'ları

**MSE (Mean Squared Error)**
```
Formül: L = (1/N) Σ (y - ŷ)²

Ne zaman:
  ✓ Normal dağılımlı hatalar
  ✓ Büyük hatalar gerçekten kötü
  ✓ Veri temiz, aykırı az
  ✓ İlk tercih (standard)

Özellik:
  ✓ Büyük hataları AĞIR cezalar
  ✗ Aykırı değerlere HASSAS

Probabilistik köken:
  Gaussian noise → MSE (MLE)
```

**MAE (Mean Absolute Error)**
```
Formül: L = (1/N) Σ |y - ŷ|

Ne zaman:
  ✓ Aykırı değer ÇOK
  ✓ Robust tahmin gerekli
  ✓ Tüm hatalar eşit önemde

Özellik:
  ✓ Aykırılara DAYANIKLI
  ✗ Sıfırda türev tanımsız (köşeli)

Median predict eder
```

**Huber Loss**
```
Formül: 
  Küçük hata: MSE gibi (smooth)
  Büyük hata: MAE gibi (robust)

Ne zaman:
  ✓ DENGE arıyorum
  ✓ Hem smooth hem robust istiyorum
  
EN İYİ PRATİK DENGE!
```

#### Sınıflama Loss'ları

**Cross-Entropy (CE)**
```
Binary: -(y log p + (1-y) log(1-p))
Multi-class: -Σ y_k log p_k

Ne zaman:
  ✓ Sınıflama (her zaman!)
  ✓ Olasılık tahmini istiyorum

Özellik:
  Yanlış sınıfa YÜKSEK GÜVENde
  → Ekstra AĞIR ceza

Probabilistik köken:
  Bernoulli/Categorical → CE (MLE)
```

#### Özet Reçete

```
Regresyon:
  Temiz veri → MSE
  Aykırı çok → MAE
  Denge → Huber

Sınıflama:
  Cross-Entropy (standart)
  İmbalanced → + class weight / Focal Loss
```

---

### 3️⃣ Learning Rate (LR) — Tanım ve Semptomlar

#### Tanım

```
θ_new = θ_old - η × ∇L
                 ↑
            Learning Rate
            (Adım büyüklüğü)
```

**Analoji:** Dağdan inerken attığın adımın boyu

#### Semptomlar Tablosu

**LR Çok Büyük**
```
Belirtiler:
  ❌ Loss ZIPlar (testere dişi)
  ❌ NaN / Inf
  ❌ Ağırlık normları patlar
  ❌ Val metrikleri kaotik

Grafik:
  Loss
    │ ╱╲╱╲╱╲
    │╱      ╲╱╲
    └────────────→ Epoch

İlk Yardım:
  1. LR'ı yarıya indir (0.01 → 0.001)
  2. Özellik ölçekle (StandardScaler)
  3. L2 ekle (weight decay)
  4. Gradient clipping (gerekirse)
```

**LR Biraz Büyük**
```
Belirtiler:
  ⚠️ Yakınsıyor ama salınım var
  ⚠️ Val istikrarsız
  ⚠️ Sweet spot'u bulmuyor

Grafik:
  Loss
    │╲   ╱╲
    │ ╲ ╱  ╲╱
    │  ╲╱
    └────────────→ Epoch

Çözüm:
  1. LR'ı azalt (0.01 → 0.005)
  2. ReduceLROnPlateau kullan
  3. Cosine decay dene
```

**LR Uygun ✓**
```
Belirtiler:
  ✅ Train düşüyor (smooth)
  ✅ Val düzenli iyileşiyor
  ✅ Aşırı salınım yok

Grafik:
  Loss
    │╲___
    │    ╲___
    │        ╲___
    └────────────→ Epoch

Aksiyon:
  1. Bu ayarı KAYDET
  2. Early stopping eşiği tanımla
  3. Week 1'de kullan
```

**LR Çok Küçük**
```
Belirtiler:
  ⏱️ Çok YAVAŞ
  ⏱️ Loss azalıyor ama ağır çekim
  ⏱️ Epoch sayısı yetersiz

Grafik:
  Loss
    │╲
    │ ╲
    │  ╲
    │   ╲  (hala inmekte...)
    └────────────→ Epoch

Çözüm:
  1. LR'ı artır (0.0001 → 0.001)
  2. Warmup + decay dene
  3. Epoch sayısını artır
```

#### İlk Yardım Protokolü

```
1. LR'ı YARIYLA indir
   → %70 durumda düzelir

2. Özellikleri STANDARTLAŞTIR
   → Loss yüzeyi yuvarlanır

3. L2 EKLE (weight decay)
   → Patlamayı yumuşatır

4. zero_grad() KONTROL
   → Gradient accumulation olmasın

5. Loss/Metric DOĞRU mu?
   → Regresyon ≠ CE

6. Shape/Dtype/Device?
   → print(x.shape, x.dtype, x.device)
```

---

### 4️⃣ Overfit vs Underfit — Sinyal ve İlk Yardım

#### Underfit (Öğrenemedi)

**Belirtiler:**
```
Train Loss: YÜKSEK 📈
Val Loss:   YÜKSEK 📈

Model yetersiz, öğrenememiş
```

**Grafiksel:**
```
Loss
  │ ─────────── Train (yüksek)
  │
  │ ─────────── Val (yüksek)
  │
  └────────────→ Epoch

İkisi de kötü → UNDERFIT
```

**Çözümler:**
```
1. Kapasite ↑
   - Daha karmaşık model
   - Daha fazla katman/nöron

2. Daha uzun eğitim
   - Epoch artır
   - LR schedule düzelt

3. Daha iyi özellikler
   - Feature engineering
   - Daha iyi temsil

4. LR ayarı
   - Çok küçükse artır
   - Optimizer değiştir (AdamW)
```

#### Overfit (Ezber)

**Belirtiler:**
```
Train Loss: ÇOK DÜŞÜK 📉
Val Loss:   YÜKSEK 📈

Model ezberlemiş, genelleyemiyor
```

**Grafiksel:**
```
Loss
  │
  │ ──────────╲___ Train (mükemmel)
  │            ╲
  │ ────────────╱── Val (kötüleşiyor)
  │             ↑
  │        Overfit başladı
  │            (burda dur!)
  └────────────→ Epoch
```

**Çözümler (Regularization):**
```
1. Early Stopping ⭐
   Val loss kötüleşince DUR
   → En basit ve etkili

2. L2 (Weight Decay)
   λ = 1e-4, 1e-3, 1e-2
   → Ağırlıkları küçük tut

3. Daha çok veri
   → En iyi çözüm (varsa)

4. Data Augmentation
   → Yapay veri çeşitliliği

5. Dropout (derin ağlarda)
   p = 0.5 (hidden), 0.2 (input)
   → Nöron bağımlılığını kır

6. Kapasite ↓
   → Model basitleştir (son çare)
```

#### Val Eğrisi Kuralı

```
Train ↓↓ ama Val ↑↑ → OVERFIT

Aksiyon:
  1. Hemen early stopping
  2. En iyi val checkpoint'e dön
  3. Regularization artır
  4. Bir daha aynı noktayı geçme
```

---

### 5️⃣ Tensor Refleksi — Shape / Dtype / Device Kontrolü

#### Shape Kontrolü

**Neden Önemli:**
```
En sık hata: Shape mismatch!
"RuntimeError: size mismatch (10,5) vs (10,3)"
```

**Kontrol Noktaları:**
```python
# Input
print(f"Input: {x.shape}")  # (64, 784)

# Layer output
print(f"After layer1: {h1.shape}")  # (64, 128)
print(f"After layer2: {h2.shape}")  # (64, 64)
print(f"Output: {y.shape}")  # (64, 10)

# Loss computation
print(f"Target: {target.shape}")  # (64,) or (64, 10)
```

**Alışkanlık:**
```
Her katman sonrası:
  assert h.shape == expected_shape
  
Kritik noktalarda:
  logger.info(f"Shape checkpoint: {x.shape}")
```

#### Dtype Kontrolü

**Kurallar:**
```
Float işlemler:
  ✓ Loss hesabı: float32
  ✓ Aktivasyonlar: float32
  ✓ Ağırlıklar: float32

Int işlemler:
  ✓ Sınıf etiketleri: int64 (long)
  ✓ Index'ler: int64

❌ Karışırsa: Sessiz hata veya yavaşlama
```

**Örnek:**
```python
# ❌ YANLIŞ
target = torch.tensor([0, 1, 2], dtype=torch.float32)
loss = F.cross_entropy(logits, target)  # HATA!

# ✅ DOĞRU
target = torch.tensor([0, 1, 2], dtype=torch.long)
loss = F.cross_entropy(logits, target)  # OK
```

#### Device Kontrolü

**Kurallar:**
```
Tüm tensor'ler AYNI cihazda:
  ✓ Model: device='mps'
  ✓ Input: device='mps'
  ✓ Target: device='mps'

❌ Karışırsa:
  - CPU'da kalan tensor → Sessiz YAVAŞLAMA
  - Farklı device'lar → RuntimeError
```

**Best Practice:**
```python
# Başta device belirle
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Model
model = MyModel().to(device)

# Her batch
for x, y in dataloader:
    x = x.to(device)
    y = y.to(device)
    
    # Training...
```

#### Gradient Akışı

**Kritik Kontrol:**
```python
# Her iterasyon
optimizer.zero_grad()  # Temizle
loss.backward()        # Hesapla
optimizer.step()       # Güncelle

# ❌ zero_grad() unutulursa:
# Gradientler BİRİKİR → Patlama!
```

#### Checklist (Her Yeni Kod)

```
□ Shape'leri logladım mı?
□ Dtype'lar doğru mu? (float loss, long target)
□ Device'lar aynı mı? (hepsi MPS/CPU/CUDA)
□ zero_grad() her iterasyonda mı?
□ requires_grad=True parametrelerde mi?
□ no_grad() güncellemede mi?
```

---

## 🧠 Mini-Ödev — Cevaplar

### A) Tek Cümlelik Tanımlar

```
┌─────────────────────────────────────────────────┐
│ TERİM           │ TEK CÜMLE TANIM               │
├─────────────────────────────────────────────────┤
│ Model           │ Girdiyi çıktıya dönüştüren,   │
│                 │ parametreleri ayarlanabilir    │
│                 │ FONKSIYON (f_θ)               │
├─────────────────────────────────────────────────┤
│ Parametre       │ Modelin ÖĞRENİLEN iç sayıları │
│                 │ (ağırlıklar, bias'lar)        │
├─────────────────────────────────────────────────┤
│ Hiperparametre  │ Eğitim sürecinin KULLANICI    │
│                 │ SEÇİMİ ayarları (LR, L2,      │
│                 │ batch, katman sayısı)         │
├─────────────────────────────────────────────────┤
│ Loss (Kayıp)    │ Tahminin YANLIŞLIĞINI tek     │
│                 │ sayıda ölçen fonksiyon        │
├─────────────────────────────────────────────────┤
│ Optimizer       │ Kayıp azalacak şekilde        │
│                 │ parametreleri GÜNCELLEYEN     │
│                 │ algoritma (SGD, AdamW)        │
├─────────────────────────────────────────────────┤
│ Regularization  │ Ezberi (OVERFIT) FRENLEYEN    │
│                 │ teknikler (L2, early stopping,│
│                 │ dropout, augmentation)        │
└─────────────────────────────────────────────────┘
```

---

### B) Üç Problem: Doğru Metrik & Olası Leakage

#### Problem 1: Ev Fiyatı Tahmini (Regresyon)

```
┌─────────────────────────────────────────────┐
│ METRIK                                      │
├─────────────────────────────────────────────┤
│ Birincil: RMSE                              │
│   → Birim anlamlı (TL cinsinden)            │
│                                             │
│ İkincil: MAE                                │
│   → Aykırılara robust                       │
│                                             │
│ R² (opsiyonel)                              │
│   → Varyans açıklama oranı                  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ LOSS                                        │
├─────────────────────────────────────────────┤
│ Temel: MSE                                  │
│ Aykırı varsa: Huber / MAE                  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ LEAKAGE RİSKLERİ                            │
├─────────────────────────────────────────────┤
│ ❌ "İlan sonrası" bilgiler                  │
│    → Pazarlık detayı, satış tarihi          │
│                                             │
│ ❌ Bölgesel ortalama etiketten türetme      │
│    → "Mahalledeki ortalama fiyat" özelliği  │
│    → Eğer etiket = fiyat ise SIZINTI!       │
│                                             │
│ ❌ Gelecek bilgi                            │
│    → "6 ay sonraki mahalle değeri"          │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ DOĞRU AYIRMA                                │
├─────────────────────────────────────────────┤
│ Standard: Rastgele split (70/15/15)         │
│                                             │
│ Coğrafi kayma varsa:                        │
│   → Bölge bazlı split                       │
│   → Train: İstanbul                         │
│   → Test: Ankara (distribution shift test) │
│                                             │
│ Zaman serisi ise:                           │
│   → Temporal split (geçmiş→gelecek)         │
└─────────────────────────────────────────────┘
```

#### Problem 2: Dolandırıcılık Tespiti (Dengesiz Sınıflama)

```
┌─────────────────────────────────────────────┐
│ METRIK                                      │
├─────────────────────────────────────────────┤
│ Birincil: Recall ⭐                         │
│   → Dolandırıcılık kaçırmak PAHALI          │
│   → "Gerçek pozitifin %'sini yakaladık?"    │
│                                             │
│ İkincil: Precision                          │
│   → Yanlış alarm maliyeti                   │
│                                             │
│ Denge: F1 Score                             │
│   → Precision + Recall harmonik ort.        │
│                                             │
│ Eğri: PR-AUC ⭐                             │
│   → İmbalanced data'da ROC-AUC'dan iyi     │
│                                             │
│ ❌ KULLANMA: Accuracy                       │
│   → %99 negatif varsa yanıltır!             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ LOSS                                        │
├─────────────────────────────────────────────┤
│ Cross-Entropy + Class Weight                │
│   → Azınlık sınıfa ağırlık ver              │
│                                             │
│ Alternatif: Focal Loss                      │
│   → Kolay örnekleri down-weight             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ LEAKAGE RİSKLERİ                            │
├─────────────────────────────────────────────┤
│ ❌ Etiket kurallarından türetilen özellikler│
│    → "Manuel inceleme sonucu" flag'i        │
│    → Eğer etiket bununla oluşturulduysa!    │
│                                             │
│ ❌ Müdahale sonrası bilgi                   │
│    → "Hesap bloklandı" durumu               │
│    → "Kullanıcı şikayet etti"               │
│                                             │
│ ❌ Gelecek davranış                         │
│    → "30 gün sonra chargeback oldu"         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ DOĞRU AYIRMA                                │
├─────────────────────────────────────────────┤
│ Stratified Split ⭐                         │
│   → Sınıf oranını KORU                      │
│   → Train: %1 pozitif                       │
│   → Val: %1 pozitif                         │
│   → Test: %1 pozitif                        │
│                                             │
│ Zaman kayması varsa:                        │
│   → Temporal split                          │
│   → Dolandırıcılık pattern'leri değişir     │
└─────────────────────────────────────────────┘
```

#### Problem 3: Satış Tahmini (Zaman Serisi)

```
┌─────────────────────────────────────────────┐
│ METRIK                                      │
├─────────────────────────────────────────────┤
│ Birincil: MAE / RMSE                        │
│   → MAE: Outlier'a robust                   │
│   → RMSE: Büyük hata önemliyse              │
│                                             │
│ İkincil: MAPE (Mean Abs % Error)           │
│   → Yüzde hata (karşılaştırılabilir)        │
│   → Dikkat: y=0 ise tanımsız!               │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ LOSS                                        │
├─────────────────────────────────────────────┤
│ MSE veya MAE                                │
│   → Outlier çoksa MAE tercih                │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ LEAKAGE RİSKLERİ ⚠️ EN YÜKSEK!             │
├─────────────────────────────────────────────┤
│ ❌ GELECEĞİ geçmişe karıştırmak (EN SİK!)   │
│    → Gelecek ay kampanyası                  │
│    → Gelecek haftaki stok durumu            │
│    → Future rolling mean/std                │
│                                             │
│ ❌ Normalization'da leak                    │
│    → Tüm veriden mean/std                   │
│    → DOĞRUSU: Sadece geçmişten öğren        │
│                                             │
│ ❌ Feature engineering'de leak              │
│    → "Bu ayın sonu toplam satış" özelliği   │
│    → Ayın sonu henüz gelmedi!               │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ DOĞRU AYIRMA ⭐ KRİTİK                      │
├─────────────────────────────────────────────┤
│ Temporal Split (MUTLAK)                     │
│   → Train: 2019-01 ~ 2019-12                │
│   → Val:   2020-01 ~ 2020-03                │
│   → Test:  2020-04 ~ 2020-06                │
│                                             │
│ Rolling Window                              │
│   → Her tahmin için: geçmiş → gelecek       │
│   → Asla geriye bakma!                      │
│                                             │
│ Expanding Window                            │
│   → Train seti zamanla büyür                │
│   → Ama asla gelecek dahil edilmez          │
└─────────────────────────────────────────────┘
```

---

### C) LR'ı 10× Büyütürsen / 10× Küçültürsen

```
┌──────────────┬────────────────────────────┬───────────────────────────────┬─────────────────────────────┐
│   DURUM      │        SEMPTOM             │      HIZLI TEŞHİS             │          ÇÖZÜM              │
├──────────────┼────────────────────────────┼───────────────────────────────┼─────────────────────────────┤
│ LR ×10       │ ❌ Loss ZIPlar             │ Ağırlık normları artıyor      │ 1. LR'ı yarıya/onda bire    │
│ (ÇOK BÜYÜK)  │ ❌ NaN / Inf               │ Gradient'lar patlıyor         │ 2. L2 ekle (weight decay)   │
│              │ ❌ Val metrikleri kaotik   │ Loss grafik testere dişi      │ 3. Gradient clipping        │
│              │ ❌ Ağırlıklar aşırı büyür  │ Diverge ediyor                │ 4. Özellikleri ölçekle      │
├──────────────┼────────────────────────────┼───────────────────────────────┼─────────────────────────────┤
│ LR ÷10       │ ⏱️ Loss ÇOK YAVAŞ azalıyor│ Aynı epoch'ta belirgin        │ 1. LR'ı kademeli artır      │
│ (ÇOK KÜÇÜK)  │ ⏱️ Eğitim sürüyor ama     │ iyileşme yok                  │ 2. LR warmup + decay dene   │
│              │    ilerlemiyor             │ Gradient'lar çok küçük        │ 3. Epoch sayısını artır     │
│              │ ⏱️ Ağır çekim              │ Momentum yok                  │ 4. AdamW dene (adaptif LR)  │
├──────────────┼────────────────────────────┼───────────────────────────────┼─────────────────────────────┤
│ Biraz Büyük  │ ⚠️ Yakınsıyor ama salınım │ LR azaltınca stabil oluyor    │ 1. LR'ı azalt               │
│              │ ⚠️ Val istikrarsız        │ Sweet spot bulamıyor          │ 2. ReduceLROnPlateau        │
│              │ ⚠️ Converge edemiyor      │ Oscillation var               │ 3. Cosine decay             │
├──────────────┼────────────────────────────┼───────────────────────────────┼─────────────────────────────┤
│ TAM İSABET ✓ │ ✅ Train düşüyor (smooth) │ Eğriler sakin                 │ 1. Bu ayarı KAYDET          │
│              │ ✅ Val düzenli iyileşiyor │ Aşırı salınım yok             │ 2. Early stopping eşiği     │
│              │ ✅ Stabil convergence     │ Sweet spot'ta!                │    tanımla                  │
│              │                            │                               │ 3. Week 1'de kullan         │
└──────────────┴────────────────────────────┴───────────────────────────────┴─────────────────────────────┘
```

**Grafiksel Karşılaştırma:**

```
LR ×10 (Çok Büyük):
  Loss
    │ ╱╲╱╲╱╲╱╲
    │╱        ╲╱
    └────────────→ Epoch
    Testere dişi, diverge

LR ÷10 (Çok Küçük):
  Loss
    │╲
    │ ╲
    │  ╲
    │   ╲  (hala inmekte)
    └────────────→ Epoch
    Ağır çekim

LR Tam (✓):
  Loss
    │╲___
    │    ╲___
    │        ╲___
    └────────────→ Epoch
    Smooth, stabil
```

---

## 📝 Ek Notlar (Kendime)

### Ölçekleme (Standardization)

```
x' = (x - μ) / σ

Neden?
  → Loss yüzeyini YUVARLAR
  → GD zikzak yapmaz
  → LR hassasiyeti azalır
  → Convergence HIZLANIR

Ne zaman?
  → Gradyan tabanlı modellerde ŞAR T
  → Linear, MLP, CNN, Transformer
  → Tree-based'de gereksiz (RF, XGBoost)
```

### Early Stopping = Implicit Regularizer

```
Matematiksel:
  GD her step: w_t = w_0 - ηt × Σ∇L
  
  t küçük → w küçük → Implicit L2
  t büyük → w büyük → Overfit risk

Pratik:
  → En güçlü regularization
  → Kolay implement
  → %90 durumda yeterli
  
Hiperparametre:
  patience = 10  (10 epoch iyileşme yoksa dur)
```

### Doğru Metrik = İş Hedefi

```
Dengesiz Sınıf:
  ❌ Accuracy (yanıltır)
  ✅ Precision/Recall/F1
  ✅ PR-AUC

Regresyon:
  ✅ RMSE (birim anlamlı)
  ✅ MAE (robust)
  ⚠️ R² (dikkatli yorumla)

Eşik Seçimi:
  → İş maliyetine göre
  → Precision vs Recall dengesi
  → ROC curve'de optimal nokta
```

### Test Set Kutsaldır

```
❌ ASLA:
  - Test'e bakıp ayar yapma
  - Test performansı kötüyse yeniden dene
  - Test'i "development" için kullanma

✅ SADECE:
  - Val ile ayarla
  - Val'de memnunsan
  - Test'e BİR KEZ bak
  - Final rapor

Neden?
  → Test overfitting'i önle
  → Dürüst değerlendirme
  → Gerçek dünya performansı
```

---

## 🎯 Week 0 Başarı Kriterleri

### Self-Check (Hepsine ✓ koyabilir misin?)

```
□ Train/Val/Test farkını açıklayabiliyorum
□ MSE/MAE/Huber ne zaman kullanılır biliyorum
□ LR semptomlarını tanıyabiliyorum
□ Overfit/Underfit'i teşhis edip çözebiliyorum
□ Shape/Dtype/Device kontrolünü alışkanlık yaptım
□ Data leakage tespiti yapabiliyorum
□ Doğru metrik seçebiliyorum (problem tipine göre)
□ Early stopping implement edebilirim
□ L2 regularization'ın ne işe yaradığını biliyorum
□ Probabilistik kökenleri anlıyorum (MSE←Gaussian, L2←Prior)

Hepsi ✓ ise → Week 0 BAŞARILI! 🎓
```

### Week 1 Hedefi

```
Lineer Regresyon:
  1. Sentetik veri oluştur (y = wx + b + ε)
  2. Manuel GD implement (autograd'sız)
  3. nn.Module ile training
  4. Train/Val split
  5. Early stopping
  6. L2 regularization
  7. Metrik: Val MSE < 0.5

Hedef:
  ✓ Kod yazarken "neden" bilmek
  ✓ Her satırı theory'ye bağlamak
  ✓ Debug yaparken sistematik düşünmek
```

---

## 💬 Sonuç

Week 0'nın hedefi olan temel kavramları özümsedim:

```
✅ Model = Parametrik fonksiyon (f_θ)
✅ Loss = Hata ölçüsü (MSE, CE, ...)
✅ Optimizer = Güncelleme algoritması (GD, Adam)
✅ Train/Val/Test = Dürüst değerlendirme sistemi
✅ Overfit/Underfit = Ezber vs Yetersizlik
✅ Tensor = Shape + Dtype + Device
✅ LR = Hayat memat meselesi
✅ Regularization = Overfit frenleyici
```

**Yarın Week 1:**
- Lineer regresyon (theory → practice)
- Bu temelleri KOD'a dökme
- Val MSE < 0.5 hedefi
- Nedenini açıklayarak!

**Hazırım! 💪**

---

