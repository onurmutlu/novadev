# Week 0: Matematiksel Temeller - Hocanın Tahtası

**NovaDev v1.0 - Derin Matematiksel Bakış**

> "Hız yok, kopyala-yapıştır yok. Temeli taş taş döşeyelim."

---

## 🎯 Bu Döküman Hakkında

Bu notlar, makine öğrenmesinin **matematiksel ve probabilistik temellerini** açıklar. 
Amaç: "Neden bu formül?", "Nereden geldi?" sorularına **net cevaplar**.

**Seviye:** Orta-İleri (theory_foundations.md'yi okuduktan sonra)
**Süre:** 90-120 dakika
**Hedef:** "Neden?" sorusuna akıcı cevap verebilmek

---

## 0️⃣ Çerçeve: ML Gerçekte Ne Yapar?

### 🎯 Temel İddia

**Bir fonksiyon arıyoruz.**

```
Girdi (x) → Fonksiyon → Çıktı (ŷ = f_θ(x))
```

Burada:
- **x**: Girdi (features)
- **f_θ**: Öğrenilen fonksiyon
- **θ** (theta): Ayarlanabilir **parametreler**
- **ŷ**: Tahmin

### 📐 Matematiksel Formülasyon

```
Hedef: θ* = argmin_θ E[(y - f_θ(x))²]
                      ↑
                 Beklenti (tüm olası veri)
```

**Pratikte:** Elimizdeki örneklerden yaklaşık hesaplarız:

```
θ* ≈ argmin_θ (1/N) Σ L(y_i, f_θ(x_i))
                        ↑
                    Empirical loss
```

### 💡 Amaç: Gerçek Dünya İlişkisini Taklit Etmek

```
Gerçek dünya:  y = g(x) + ε  (gürültülü)
                    ↑      ↑
                Bilinmeyen Kaçınılmaz
                 fonksiyon  gürültü

Bizim model:  ŷ = f_θ(x)
               ↑
          Yaklaştırma
```

### ✅ Bir Cümle Özet

> **Eğitim = Hatayı ölçen bir pusulaya (loss) bakarak parametreleri ayarlama işidir.**

**Bileşenler:**
- **Veri**: Örnekler {(x_i, y_i)}
- **Loss**: Hatayı tek sayıya çevirir L(θ)
- **Optimizasyon**: L(θ)'yı azaltmak için θ'yı değiştir
- **Genelleme**: Ezber değil, **görmediğin veride** de işe yarama

---

## 1️⃣ Veri: Neden Kutsal? (Ve Nasıl Bozulur)

### 📊 i.i.d. Varsayımı (Temelin Temeli)

**i.i.d.** = **i**ndependent and **i**dentically **d**istributed

**Anlamı:** Eğitim ve test örnekleri **aynı dağılımdan** gelmeli.

```
p_train(x, y) = p_test(x, y)
```

**Neden Önemli?**
- Eğer dağılımlar farklıysa → "Eğitimde iyi, gerçekte kötü"

### 🔴 i.i.d. İhlalleri (Gerçek Hayatta Sık)

#### 1. Covariate Shift (Kovaryat Kayması)
```
p(x) değişir ama p(y|x) sabit

Örnek: Kamera açısı değişir ama
        "kedi" tanımı aynı kalır
```

**Belirtiler:**
- Test verisi farklı görünüyor
- Özellik dağılımları shift etmiş

**Çözüm:**
- Domain adaptation
- Feature standardization
- Data augmentation

#### 2. Concept Drift (Konsept Kayması)
```
p(y|x) değişir ama p(x) sabit

Örnek: "Spam" tanımı zamanla evriliyor
        ama email format'ı aynı
```

**Belirtiler:**
- Model performansı zamanla düşüyor
- Etiket anlamı değişmiş

**Çözüm:**
- Periodic retraining
- Online learning
- Temporal validation

#### 3. Prior Shift (Önsel Kayması)
```
p(y) değişir (sınıf oranları)

Örnek: Training'de %50 pozitif,
        Production'da %10 pozitif
```

**Belirtiler:**
- Class imbalance production'da farklı
- Metrics yanıltıcı

**Çözüm:**
- Stratified sampling
- Class weighting
- Calibration

### 🎯 Veri Bölme Stratejisi

#### Standard Split
```
Tüm Veri (100%)
    ↓
├─ Train (70%)      → Parametreler öğren
├─ Val (15%)        → Hiperparametre ayarla
└─ Test (15%)       → Final değerlendirme
```

#### Time Series Split
```
[────Train────][─Val─][Test]
     Geçmiş    Yakın  Gelecek
                Gelecek
```

**Dikkat:** Zamansal sızıntı olmasın! Future information geçmişe sızmamalı.

### ⚠️ Kural: Validation vs Test

```
❌ YANLIŞ:
for hp in hyperparameters:
    score = evaluate_on_test(hp)  # Test'i kirletiyorsun!
    
✅ DOĞRU:
for hp in hyperparameters:
    score = evaluate_on_val(hp)   # Val ile seç
final_score = evaluate_on_test(best_hp)  # Test'e bir kez bak
```

### 🔍 Data Leakage Örnekleri (Saha Deneyimi)

#### Leakage Tipi 1: Temporal
```python
# ❌ YANLIŞ: Geleceği kullanıyorsun
df['mean_price'] = df.groupby('user')['price'].transform('mean')
# Bu hesap tüm veriyi kullanır!

# ✅ DOĞRU: Sadece geçmişi kullan
df['mean_price'] = df.groupby('user')['price'].expanding().mean()
```

#### Leakage Tipi 2: Target Encoding
```python
# ❌ YANLIŞ: Hedef bilgisi özelliğe sızdı
category_means = train['category'].map(
    train.groupby('category')['target'].mean()
)
# Test kategorilerinin target ortalaması train'den!

# ✅ DOĞRU: Cross-validation ile
from sklearn.model_selection import KFold
# Her fold için target encoding yap
```

#### Leakage Tipi 3: Proxy Variables
```
Özellikler: [birim_fiyat, adet, TOPLAM_FİYAT]
Hedef: adet

→ TOPLAM_FİYAT / birim_fiyat = adet (sızıntı!)
```

**Antidot:**
- Feature correlation matrix kontrol et
- Domain knowledge kullan
- Leakage detection tools (SHAP values kontrol)

---

## 2️⃣ Özellikler: Ham Veri → Öğrenilebilir Temsil

### 🎚️ Ölçekleme: Neden Bu Kadar Kritik?

#### Matematiksel Açıklama

Kayıp yüzeyi **özellik ölçeklerine** bağlı:

```
L(w) = Σ (y - Σ w_j x_j)²

Eğer x_1 ∈ [0, 1] ve x_2 ∈ [0, 1000]:
  → w_1'in etkisi küçük görünür
  → w_2'nin etkisi abartılır
  → Loss yüzeyi ÇARPIK (eliptik)
```

**Eliptik Yüzey Problemi:**
```
        w_2
         ↑
    ╱╲  │  ╱╲     ← Çok dik
   ╱  ╲ │ ╱  ╲
  ╱    ╲│╱    ╲
 ────────────────→ w_1
        ← Çok yassı

GD bu yüzeyde ZİKZAK yapar!
```

**Ölçeklenmiş Yüzey:**
```
        w_2
         ↑
       ╱─╲        ← Dairevi
      │   │
      ╲───╱
 ─────────────→ w_1

GD dümdüz iner!
```

### 📏 Ölçekleme Yöntemleri

#### Z-Score (Standardization)
```
x' = (x - μ) / σ

Sonuç: μ=0, σ=1
Avantaj: Outlier'ları korur
Kullanım: Çoğu durumda ilk tercih
```

#### Min-Max Scaling
```
x' = (x - x_min) / (x_max - x_min)

Sonuç: [0, 1] aralığı
Avantaj: Bounded range
Dezavantaj: Outlier'a hassas
```

#### Robust Scaling
```
x' = (x - median) / IQR

IQR = Q3 - Q1 (interquartile range)
Avantaj: Outlier'a robust
Kullanım: Aykırı değer çok varsa
```

### 🎯 Kategorik Değişkenler

#### One-Hot Encoding
```
Renk: ['kırmızı', 'mavi', 'yeşil']
     ↓
[[1, 0, 0],  # kırmızı
 [0, 1, 0],  # mavi
 [0, 0, 1]]  # yeşil
```

**Avantaj:** Sıra varsayımı yok
**Dezavantaj:** High cardinality'de boyut patlaması

#### Target Encoding
```
Kategori → Mean(target | kategori)

Örnek:
'istanbul' → 0.85  (bu kategoride ortalama target)
'ankara'   → 0.62
```

**Avantaj:** Tek boyut
**⚠️ DİKKAT:** Leakage riski! Cross-validation ile yap

#### Learnable Embeddings
```
Kategori → Düşük boyutlu vektör (öğrenilir)

'istanbul' → [0.5, -0.3, 0.8]
'ankara'   → [0.2, 0.4, -0.1]
```

**Avantaj:** Model kendisi öğrenir
**Kullanım:** Derin ağlarda, NLP'de

### 📝 Metin İşleme

#### Klasik: Bag of Words
```
"makine öğrenmesi zor" → [1, 1, 0, 1, ...]
                           m  ö  a  z
```

**Dezavantaj:** Sıra bilgisi kaybolur

#### Modern: Pre-trained Embeddings
```
"makine öğrenmesi" → BERT/GPT embedding
                     [0.23, -0.45, ..., 0.78]  (768-dim)
```

**Avantaj:** Semantik ilişkiler korunur

### 🔴 Outlier Yönetimi

#### MSE ve Outlier İlişkisi

```
MSE = (1/N) Σ (y - ŷ)²
                 ↑
            Kare alıyor!
```

**Büyük hata → Kare → Çok büyük ceza**

**Örnek:**
```
Hatalar: [1, 1, 1, 1, 10]
MSE = (1+1+1+1+100)/5 = 20.8  ← Tek outlier dominates!
MAE = (1+1+1+1+10)/5 = 2.8    ← Daha robust
```

#### Robust Loss: Huber
```
         { (1/2)x²        if |x| ≤ δ
L(x) = {
         { δ(|x| - δ/2)   if |x| > δ

Küçük hata: MSE gibi (smooth)
Büyük hata: MAE gibi (linear)
```

---

## 3️⃣ Kayıp Fonksiyonları: Oyunun Kuralı

### 🎯 Temel Prensip

> "Ne ölçüyorsan ona dönüşürsün."

Loss fonksiyonu **neyi optimize ettiğini** tanımlar.

### 📐 Regresyon Loss'ları

#### Mean Squared Error (MSE)
```
L_MSE = (1/N) Σ (y_i - ŷ_i)²
```

**Özellikleri:**
- **Kare**: Büyük hataları ağır cezalandırır
- **Smooth**: Türevi her yerde var
- **Convex**: Tek minimum (linear model'de)

**Ne Zaman Kullan:**
- Normal dağılımlı hatalar
- Outlier az
- Standard regression

**Dezavantaj:**
- Outlier'a aşırı hassas

#### Mean Absolute Error (MAE)
```
L_MAE = (1/N) Σ |y_i - ŷ_i|
```

**Özellikleri:**
- **Linear**: Outlier'a daha toleranslı
- **Robust**: Median predict eder

**Ne Zaman Kullan:**
- Outlier çok
- Robust tahmin lazım

**Dezavantaj:**
- Köşeli (optimizasyon biraz hassas)
- Sıfırda türev tanımsız

#### Huber Loss (Hibrit)
```
         { (1/2)(y-ŷ)²      if |y-ŷ| ≤ δ
L_Huber = {
         { δ|y-ŷ| - δ²/2    if |y-ŷ| > δ
```

**Pratikte En İyi Denge:**
- Küçük hata: MSE (smooth)
- Büyük hata: MAE (robust)

### 🎲 Sınıflandırma Loss'ları

#### Cross-Entropy (Log Loss)
```
L_CE = -Σ y_i × log(ŷ_i)

Binary: -(y log(p) + (1-y) log(1-p))
```

**Sezgisel Açıklama:**
```
Model %10 der ama gerçek 1 → Büyük ceza
Model %90 der ve gerçek 1  → Küçük ceza
Model %99 der ve gerçek 1  → Çok küçük ceza

-log(0.1) = 2.3  ← Ağır
-log(0.9) = 0.1  ← Hafif
-log(0.99) = 0.01 ← Çok hafif
```

**Ne Zaman Kullan:**
- Sınıflandırma (her zaman!)
- Olasılık tahminleri

#### Focal Loss
```
L_focal = -(1-p)^γ × log(p)
             ↑
        Modülasyon faktörü
```

**Özellik:**
- Kolay örnekler → Düşük ağırlık
- Zor örnekler → Yüksek ağırlık

**Ne Zaman Kullan:**
- Class imbalance
- Hard negative mining

### 📊 Metrik Seçimi vs Loss Seçimi

```
┌─────────────┬──────────────┬─────────────┐
│   Problem   │     Loss     │   Metric    │
├─────────────┼──────────────┼─────────────┤
│ Regression  │ MSE/Huber    │ RMSE, MAE   │
│ Binary Cls  │ BCE          │ F1, AUC     │
│ Multi Cls   │ CE           │ Accuracy    │
│ Imbalanced  │ Focal/Weighted│ PR-AUC, F1  │
│ Ranking     │ Pairwise     │ nDCG, MRR   │
└─────────────┴──────────────┴─────────────┘
```

**Önemli:** Loss optimize edilir, Metric raporlanır.

---

## 4️⃣ Optimizasyon: Eğim Neden Yeterli?

### 🗻 Loss Yüzeyi Analojisi

```
Kayıp fonksiyonu = Dağlık arazi

L(θ)
  ↑
  │     ╱╲    ╱╲
  │   ╱    ╲╱    ╲
  │ ╱              ╲
  └──────────────────→ θ
  
  Bulunduğun nokta: θ_current
  Hedef: Vadiye in (minimum)
  Elindeki: Gradient (eğim vektörü)
```

### 📐 Gradient Descent Formülü

```
θ_new = θ_old - η × ∇L(θ_old)
         ↑       ↑      ↑
       Mevcut   LR   Gradient
```

**Sezgi:**
- **Gradient**: En dik iniş yönü
- **LR (η)**: Adım büyüklüğü
- **Negatif**: Yukarı değil aşağı git

### 🎚️ Learning Rate: Hayat Memat Meselesi

```
LR çok büyük (η = 1.0):
  θ ───→ ┼ ←─── θ'
         ↑
    Minimum'u aştı!
    Zıpla zıpla diverge olur

LR uygun (η = 0.1):
  θ ───→ · ───→ · ───→ ●
         ↓   Smooth
        Minimum'a yaklaş

LR çok küçük (η = 0.001):
  θ → · → · → · → ...
      Çok yavaş, zaman kaybı
```

### 🔄 Mini-Batch Gradient Descent

```
Batch GD:      Tüm veriyi gör → Güncelle
               ✓ Stabil
               ✗ Yavaş

Stochastic GD: Her örneği gör → Güncelle
                ✓ Hızlı
                ✗ Gürültülü

Mini-batch GD: 32-256 örnek → Güncelle
               ✓ Hız + stabilite dengesi
               ✓ GPU parallelizmi
               ← PRAKTİK STANDART
```

**Gürültü Avantajı:**
```
Gürültülü gradient bazen iyi!
  ↓
Saddle point'lerden kaçabilir
Local minimum'lardan atlar
```

### ⚡ Momentum & Nesterov

#### Standard Momentum
```
v_t = β × v_{t-1} + ∇L(θ)
θ_t = θ_{t-1} - η × v_t
      ↑
  Geçmişin ortalaması
```

**Analoji:** Top vadiden aşağı yuvarlanıyor
- İvme birikir
- Küçük tepeleri aşabilir
- Oscillation azalır

**β = 0.9:** %90 geçmiş + %10 şimdi

#### Nesterov Momentum (NAG)
```
"Önce atlayacağın yeri tahmin et,
 sonra oradan gradient ölç"

θ_lookahead = θ - β × v
v_t = β × v_{t-1} + ∇L(θ_lookahead)
θ_t = θ_{t-1} - η × v_t
```

**Avantaj:** Daha proaktif, zıplama azalır

### 🧠 Adam / AdamW

#### Adam (Adaptive Moment Estimation)
```
m_t = β1 × m_{t-1} + (1-β1) × ∇L     [First moment]
v_t = β2 × v_{t-1} + (1-β2) × (∇L)²  [Second moment]

m̂_t = m_t / (1 - β1^t)  [Bias correction]
v̂_t = v_t / (1 - β2^t)

θ_t = θ_{t-1} - η × m̂_t / (√v̂_t + ε)
                          ↑
                    Her parametreye
                    adaptif LR
```

**Sezgi:**
- **m**: Momentum (yön)
- **v**: Variance (ölçek)
- Her parametre kendi LR'ını alır

**Hiperparametreler:**
- β1 = 0.9 (momentum)
- β2 = 0.999 (variance)
- ε = 1e-8 (sayısal stabilite)

#### AdamW (Weight Decay düzeltilmiş)
```
Adam'da L2 düzgün çalışmıyor!

❌ Adam: Gradient'e L2 ekle
✅ AdamW: Direkt ağırlıkları küçült

θ_t = θ_{t-1} - η × (m̂_t / √v̂_t + λ × θ_{t-1})
                                    ↑
                              Doğru L2
```

### 📈 Learning Rate Schedule

#### Cosine Decay
```
η_t = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2

  η
  │╲
  │ ╲___
  │     ╲___
  │         ╲___
  └──────────────→ epoch
  
  Başta hızlı, sonra yumuşak
```

#### Step Decay
```
η_t = η_0 × γ^(floor(t/s))

  η
  │────┐
  │    └────┐
  │         └────┐
  └──────────────→ epoch
  
  Belli epoch'larda düşür
```

#### ReduceLROnPlateau
```
if val_loss not improving for N epochs:
    η = η / 10
```

#### Warmup
```
Başta küçük LR → Yavaş yavaş arttır

η_t = η_target × min(1, t / warmup_steps)

Neden? Büyük model'de patlamayı önler
```

---

**(Devam theory_mathematical_part2.md'de...)**

---

## 📚 Ara Özet

Bu bölümde öğrendiklerimiz:

✅ **Matematiksel çerçeve:** ML = fonksiyon arama
✅ **Veri disiplini:** i.i.d., leakage, bölme stratejisi
✅ **Feature engineering:** Ölçekleme matematiği
✅ **Loss fonksiyonları:** MSE/MAE/Huber/CE seçimi
✅ **Optimizasyon:** GD, momentum, Adam derinliği

**Sonraki bölüm (part2):**
- Sayısal koşullar & curvature
- Probabilistik bakış (MLE, MAP)
- Bias-variance matematik
- Regularization teorisi
- Değerlendirme metrikleri

---

**Durum:** 📖 Part 1 tamamlandı → Part 2'ye geç
