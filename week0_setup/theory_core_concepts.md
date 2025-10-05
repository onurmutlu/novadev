# Week 0: Temel Kavramlar - Üniversite Seviyesi

**NovaDev v1.0 - Akademik Yaklaşım**

> "Adım adım, kavramları tanımlayarak, sezgi + matematik dengesiyle. Week 1'de ne yaptığını NEDEN yaptığını bileceksin."

---

## 🎯 Bu Döküman Hakkında

**Hedef Kitle:** 
- ✅ theory_intro.md'yi okudum, temel sezgi var
- ✅ Şimdi daha formal/akademik açıklamalar istiyorum
- ✅ Hafif matematik korkutmuyor, formülleri anlamak istiyorum
- ✅ "Neden böyle?" sorularına daha derin cevaplar arıyorum

**Seviye:** Üniversite Giriş (Orta)
**Süre:** 90-120 dakika
**Stil:** Tanım → Sezgi → Hafif Matematik → Pratik
**Hedef:** Week 1 kod yazarken "bu formül nereden geldi?" bilmek

---

## 1️⃣ Makine Öğrenmesi Nedir? (Formal Tanım)

### 📐 Matematiksel Çerçeve

**Tanım:** Makine öğrenmesi, veriden **örüntü** öğrenip yeni veriler için **tahmin** yapan yöntemlerin genel adıdır.

**Bileşenler:**
```
Girdi (x):        Özellik vektörü, x ∈ ℝᵈ
Çıktı (ŷ):        Tahmin, ŷ = f_θ(x)
Parametreler (θ): Ayarlanabilir ağırlıklar
Model (f_θ):      x ↦ ŷ fonksiyonu
```

### 🎯 Amaç Formülasyonu

**Hedef:** Parametreleri öyle ayarla ki tahminler gerçeğe yakın olsun.

**Matematiksel:**
```
θ* = argmin_θ E[(y - f_θ(x))²]
              ↑
         Beklenen hata (tüm olası veri üzerinde)
```

**Pratikte:**
```
θ* ≈ argmin_θ (1/N) Σᵢ L(yᵢ, f_θ(xᵢ))
                      ↑
              Ampirik kayıp (elimizdeki veri)
```

### 💡 Sezgisel Açıklama

**Üniversite dersi analojisi:**
```
Girdi (x): Öğrencinin önceki notları, devam, ödev skoru
Model (f_θ): "Final notu tahmin" formülü
Parametreler (θ): Formüldeki ağırlıklar (devam %30, ödev %40, vize %30)
Eğitim: Geçmiş öğrencilere bakarak ağırlıkları ayarla
Test: Yeni öğrenci gelince final notunu tahmin et
```

---

## 2️⃣ Veri ve Veri Kümesi (Dataset)

### 📊 Terminoloji

#### Örnek (Sample)
**Tanım:** Tek bir veri satırı

**Notasyon:** 
```
(xᵢ, yᵢ) ← i'nci örnek
xᵢ ∈ ℝᵈ  ← d-boyutlu özellik vektörü
yᵢ ∈ ℝ   ← hedef değer (regresyon)
yᵢ ∈ {0,1,...,K-1} ← sınıf etiketi (classification)
```

#### Özellik (Feature)
**Tanım:** Örneği tanımlayan nitelikler

**Tipler:**
- **Sayısal (Numerical):** Sürekli (fiyat, ağırlık) veya kesikli (sayım)
- **Kategorik (Categorical):** Sonlu küme (renk, şehir)
- **Sıralı (Ordinal):** Sıra var (düşük/orta/yüksek)

#### Etiket (Label)
**Tanım:** Doğru cevap (supervised learning'de)

**Gözetimli vs Gözetimsiz:**
```
Gözetimli:    D = {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}
Gözetimsiz:   D = {x₁, x₂, ..., xₙ}  (etiket yok)
```

### 🎓 Öğrenme Paradigmaları

#### 1. Supervised Learning (Gözetimli)
```
Hedef: Etiketli örneklerden f: X → Y öğren

Alt türler:
  - Regression: Y = ℝ (sayısal tahmin)
  - Classification: Y = {0,1,...,K-1} (sınıf tahmini)
  
Örnekler:
  - Ev fiyatı tahmini (regression)
  - Spam sınıflama (binary classification)
  - Rakam tanıma (multi-class classification)
```

#### 2. Unsupervised Learning (Gözetimsiz)
```
Hedef: Verinin yapısını/örüntüsünü keşfet

Alt türler:
  - Clustering: Benzer grupları bul
  - Dimensionality Reduction: Boyut azalt (PCA, t-SNE)
  - Density Estimation: Dağılımı öğren
  
Örnekler:
  - Müşteri segmentasyonu (clustering)
  - Veri görselleştirme (t-SNE)
  - Anomaly detection
```

#### 3. Reinforcement Learning (Pekiştirmeli)
```
Hedef: Ödül maksimize eden politika öğren

Bileşenler:
  - Agent: Karar verici
  - Environment: Çevre
  - State: Durum
  - Action: Aksiyon
  - Reward: Ödül
  
Örnekler:
  - Oyun oynayan AI (AlphaGo)
  - Robot kontrolü
  - Otonom araçlar
```

---

## 3️⃣ Eğitim–Doğrulama–Test Ayrımı

### 🔄 Neden Bölüyoruz?

**Ana Sebep:** **Genelleme** kabiliyetini ölçmek

**Tehlike:** Eğitim verisine "ezber" (overfit)

### 📦 Veri Bölme Stratejisi

#### Standard Split
```
D (tüm veri, N örnek)
  ↓
├─ D_train (70%, 0.7N)  → Parametreleri öğren
├─ D_val   (15%, 0.15N) → Hiperparametreleri seç
└─ D_test  (15%, 0.15N) → Final performans
```

**Matematiksel:**
```
θ* = argmin_θ L_train(θ)           ← Eğitim
λ* = argmin_λ L_val(θ*(λ))         ← Validation (hiperparametre)
Performans = L_test(θ*(λ*))        ← Test (bir kez!)
```

#### Stratified Split (Sınıflama)
```
Sınıf oranlarını koru:

Orijinal: %80 negatif, %20 pozitif
  ↓
Train: %80 negatif, %20 pozitif
Val:   %80 negatif, %20 pozitif
Test:  %80 negatif, %20 pozitif
```

**Neden?** Dengesiz sınıflarda rastgele split tehlikeli!

#### Temporal Split (Zaman Serisi)
```
Zaman →
[─────Train─────][─Val─][Test]
   Geçmiş      Yakın   Gelecek
               Gelecek
```

**Kritik:** Geleceği geçmişe ASLA sızdırma!

### ⚠️ Altın Kurallar

```
1. Test setine sadece BİR KEZ bak (final rapor)
2. Validation ile hiperparametre seç
3. Test'e bakıp ayar yapma → kendini kandırırsın
4. Cross-validation (k-fold) veri azsa kullan
```

### 📊 i.i.d. Varsayımı

**Tanım:** **i**ndependent and **i**dentically **d**istributed

**Matematiksel:**
```
(xᵢ, yᵢ) ~ p(x, y)  bağımsız ve özdeş dağılımlı

Yani:
  p_train(x, y) = p_test(x, y)
```

**Gerçek Hayatta İhlaller:**

#### Covariate Shift
```
p_train(x) ≠ p_test(x)  ama  p(y|x) aynı

Örnek: Kamera modeli değişti (görüntü dağılımı kaydı)
       ama "kedi" tanımı aynı
```

#### Concept Drift
```
p_train(y|x) ≠ p_test(y|x)  ama  p(x) aynı

Örnek: "Spam" tanımı zamanla evrildi
       ama email formatı aynı
```

#### Prior Shift
```
p_train(y) ≠ p_test(y)  (sınıf oranları)

Örnek: Eğitimde %50 pozitif
       Gerçekte %5 pozitif
```

---

## 4️⃣ Amaç Fonksiyonu: Kayıp (Loss) ve Risk

### 📐 Kayıp Fonksiyonu (Loss Function)

**Tanım:** Tek bir örnek için hata ölçüsü

**Notasyon:** ℓ(y, ŷ) veya L(y, f(x))

### 🎯 Regresyon Loss'ları

#### Mean Squared Error (MSE)
```
L_MSE = (1/N) Σᵢ (yᵢ - ŷᵢ)²

Türev:
∂L/∂ŷᵢ = -2(yᵢ - ŷᵢ)

Özellikler:
  ✓ Konveks (tek minimum)
  ✓ Smooth (her yerde türevlenebilir)
  ✓ Büyük hataları ağır cezalar
  ✗ Outlier'a hassas
```

**Ne Zaman:**
- Normal dağılımlı hatalar
- Büyük hatalar gerçekten kötü
- Standard regression

#### Mean Absolute Error (MAE)
```
L_MAE = (1/N) Σᵢ |yᵢ - ŷᵢ|

Türev:
∂L/∂ŷᵢ = -sign(yᵢ - ŷᵢ)

Özellikler:
  ✓ Outlier'a robust
  ✓ Median predict eder
  ✗ Sıfırda türev tanımsız (köşeli)
```

**Ne Zaman:**
- Outlier çok
- Robust tahmin gerekli

#### Huber Loss (Hibrit)
```
         { (1/2)(y-ŷ)²      if |y-ŷ| ≤ δ
L_Huber = {
         { δ|y-ŷ| - δ²/2    if |y-ŷ| > δ

Özellikler:
  ✓ Küçük hata: MSE (smooth)
  ✓ Büyük hata: MAE (robust)
  ✓ En iyi denge
```

### 🎲 Sınıflama Loss'ları

#### Binary Cross-Entropy (BCE)
```
L_BCE = -(1/N) Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]

Burada pᵢ = σ(f(xᵢ)), σ = sigmoid

Türev:
∂L/∂f = p - y  (sigmoid ile güzel!)

Özellikler:
  ✓ Olasılık tahmini
  ✓ Yanlış özgüvene ağır ceza
  ✓ Konveks
```

#### Categorical Cross-Entropy
```
L_CE = -(1/N) Σᵢ Σₖ yᵢₖ log(pᵢₖ)

Burada pᵢ = softmax(f(xᵢ))

Multi-class için standard loss
```

### 📊 Risk Kavramı

**Expected Risk (Gerçek Amaç):**
```
R(f) = E_(x,y)~p(x,y) [ℓ(y, f(x))]
       ↑
    Tüm olası veriler üzerinde beklenti
```

**Problem:** p(x,y)'yi bilmiyoruz!

**Çözüm: Empirical Risk Minimization (ERM)**
```
R̂(f) = (1/N) Σᵢ ℓ(yᵢ, f(xᵢ))
        ↑
    Elimizdeki örneklerle yaklaşıklık

Hedef: θ* = argmin_θ R̂(f_θ)
```

---

## 5️⃣ Optimizasyon: Gradient Descent

### 🗻 Geometrik Sezgi

**Kayıp yüzeyi = Dağ arazisi**

```
L(θ)
  ↑
  │     ╱╲    ╱╲
  │   ╱    ╲╱    ╲
  │ ╱              ╲
  └──────────────────→ θ
  
Hedef: En dip noktaya in (global minimum)
Araç: Gradient (eğim vektörü)
```

### 📐 Gradient Descent Formülü

**Güncelleme Kuralı:**
```
θ_{t+1} = θ_t - η ∇_θ L(θ_t)
          ↑     ↑    ↑
       Yeni   LR  Gradient
```

**Gradient:**
```
∇_θ L = [∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₐ]ᵀ

En dik YÜKSELIŞ yönü → Negatifini al = İNİŞ
```

### 🎚️ Learning Rate (η)

**Kritik Hiperparametre!**

```
η çok büyük:
  θ ───→ ┼ ←─── θ'
         ↑
    Minimum'u aşar, diverge!
    
η optimal:
  θ ───→ · ───→ · ───→ ●
         Smooth convergence
         
η çok küçük:
  θ → · → · → · → ...
      Çok yavaş, zaman kaybı
```

**Teorik:** η < 2/L (L = Lipschitz sabiti) garanti eder, ama pratikte deneysel seçeriz.

### 🎒 Varyantlar

#### Batch Gradient Descent
```
g_t = (1/N) Σᵢ ∇_θ ℓ(yᵢ, f_θ(xᵢ))

θ_{t+1} = θ_t - η g_t

Artı: Stabil, deterministik
Eksi: Yavaş (N büyükse)
```

#### Stochastic Gradient Descent (SGD)
```
Rastgele bir örnek i seç:
g_t = ∇_θ ℓ(yᵢ, f_θ(xᵢ))

θ_{t+1} = θ_t - η g_t

Artı: Hızlı, memory-efficient
Eksi: Gürültülü, zikzak
```

#### Mini-Batch GD ⭐
```
Rastgele B örnek seç (batch):
g_t = (1/B) Σᵢ∈batch ∇_θ ℓ(yᵢ, f_θ(xᵢ))

θ_{t+1} = θ_t - η g_t

Artı: Hız + stabilite dengesi, GPU parallel
Standart: B = 32, 64, 128, 256
```

### 🚀 Gelişmiş Optimizerler

#### Momentum
```
v_t = β v_{t-1} + g_t
θ_t = θ_{t-1} - η v_t

β tipik: 0.9

Sezgi: Top yuvarlanıyor, ivme birikir
      → Vadide sallanma azalır
      → Küçük tepeleri aşabilir
```

#### Nesterov Momentum
```
"Önce atla, sonra bak" prensibi

θ_lookahead = θ - β v
g_lookahead = ∇L(θ_lookahead)
v_t = β v_{t-1} + g_lookahead
θ_t = θ_{t-1} - η v_t

Daha proaktif, oscillation azalır
```

#### Adam (Adaptive Moment Estimation) ⭐
```
m_t = β₁ m_{t-1} + (1-β₁) g_t     [First moment, momentum]
v_t = β₂ v_{t-1} + (1-β₂) g_t²    [Second moment, variance]

m̂_t = m_t / (1 - β₁ᵗ)             [Bias correction]
v̂_t = v_t / (1 - β₂ᵗ)

θ_t = θ_{t-1} - η m̂_t / (√v̂_t + ε)

Hiperparametreler:
  β₁ = 0.9 (momentum)
  β₂ = 0.999 (variance)
  ε = 1e-8 (numerical stability)

Her parametreye adaptif LR!
```

#### AdamW
```
Adam'ın L2 düzeltilmiş versiyonu

θ_t = θ_{t-1} - η (m̂_t / √v̂_t + λ θ_{t-1})
                                   ↑
                            Doğru weight decay

Pratik Standart: AdamW + küçük λ (1e-4)
```

### 📈 LR Schedule (Learning Rate Programı)

#### ReduceLROnPlateau
```
if val_loss not improving for patience epochs:
    η = η / factor
    
Tipik: patience=10, factor=10
```

#### Cosine Annealing
```
η_t = η_min + (η_max - η_min) × (1 + cos(πt/T)) / 2

  η
  │╲
  │ ╲___
  │     ╲___
  │         ╲___
  └──────────────→ t
  
Smooth decay
```

#### Step Decay
```
η_t = η_0 × γ^⌊t/s⌋

γ = 0.1, s = 30 epoch

Belli epoch'larda LR'ı düşür
```

#### Warmup (Büyük Modellerde)
```
İlk W epoch'ta LR'ı yavaş yavaş artır:

η_t = η_target × min(1, t/W)

Patlamayı önler
```

---

## 6️⃣ Tensors, Şekiller ve Autograd

### 🧊 Tensor Nedir?

**Tanım:** Çok boyutlu sayı dizisi + metadata

```
Tensor = numpy array + dtype + device

x = torch.randn(64, 3, 224, 224, device='mps', dtype=torch.float32)
                 ↑   ↑   ↑    ↑      ↑         ↑
              Batch RGB  H    W   Device   Data type
```

**Boyutlar:**
- 0D: Scalar (5)
- 1D: Vector ([1, 2, 3])
- 2D: Matrix ([[1,2],[3,4]])
- 3D+: Tensor

### 📏 Shape Operations

#### Önemli İşlemler
```python
x.shape           # Boyutları döndür
x.view(...)       # Yeniden şekillendir (memory'de contiguous olmalı)
x.reshape(...)    # View'e benzer ama copy yapabilir
x.permute(...)    # Boyutları yer değiştir
x.transpose(...)  # İki boyutu yer değiştir
x.squeeze()       # Size=1 boyutları kaldır
x.unsqueeze(dim)  # Yeni boyut ekle
```

**En Sık Hata:**
```
RuntimeError: size mismatch
→ print(x.shape) her kritik noktada!
```

### 📡 Broadcasting Rules

**Kural:**
1. Sağdan sola hizala
2. Boyutlar eşit VEYA biri 1 olmalı
3. Eksik boyut 1 kabul edilir

**Örnekler:**
```
(3, 1, 5) + (4, 5) → (3, 4, 5) ✓
  3 1 5
    4 5
  ─────
  3 4 5

(3, 2) + (3, 1) → (3, 2) ✓
  3 2
  3 1
  ───
  3 2

(3, 2) + (2, 3) → ERROR ✗
  3 2
  2 3
  ───
  X X  (uyuşmuyor!)
```

### 🔄 Computational Graph & Autograd

**Sezgi:** Her işlem bir "fatura" keser

#### Forward Pass
```
x (requires_grad=True)
  ↓ [işlem: square]
a = x²
  ↓ [işlem: multiply]
b = a × 3
  ↓ [işlem: sum]
L = Σb

Her işlem grafiğe kaydedilir
```

#### Backward Pass (Zincir Kuralı)
```
L.backward()

∂L/∂L = 1
∂L/∂b = ∂L/∂L × ∂L/∂b = 1 × 1 = 1
∂L/∂a = ∂L/∂b × ∂b/∂a = 1 × 3 = 3
∂L/∂x = ∂L/∂a × ∂a/∂x = 3 × 2x = 6x

x.grad = 6x ← Sonuç burada
```

### ⚙️ Autograd Detaylar

```python
# 1. Gradient tracking açık
x = torch.tensor([1.0], requires_grad=True)

# 2. Forward (graph oluşur)
y = x ** 2

# 3. Backward (gradient hesapla)
y.backward()

# 4. Gradient'i oku
print(x.grad)  # 2x = 2.0

# 5. Sıfırla (yoksa birikir!)
x.grad.zero_()

# 6. Güncellemede tracking kapat
with torch.no_grad():
    x -= lr * x.grad
```

**Kritik:** `zero_grad()` unutma → gradient accumulation!

---

## 7️⃣ Overfit & Underfit: Bias-Variance

### 📊 İki Uç

```
Underfit:               Overfit:
  
Train: Kötü           Train: Mükemmel
Val:   Kötü           Val:   Kötü

Model yetersiz        Model ezber yaptı
```

### 🎯 Teşhis

**Loss Eğrileri:**
```
Loss
  │
  │ Underfit:
  │ ─────────── Train
  │ ─────────── Val
  │   (ikisi de yüksek)
  │
  │ Overfit:
  │ ──────────╲___ Train (düşük)
  │            ╱── Val (yüksek)
  │           ↑
  │      Overfit başladı
  └────────────────→ Epoch
```

### 🛡️ Regularization (Düzenleme)

#### L2 Regularization (Ridge)
```
L_total = L_data + λ/2 × ||θ||²
                   ↑
              Regularization term

λ: regularization katsayısı (1e-4, 1e-3, 1e-2)

Etki: Büyük ağırlıkları cezalar → sade model
```

#### L1 Regularization (Lasso)
```
L_total = L_data + λ × ||θ||₁

Etki: Bazı ağırlıkları TAM SIFIR yapar (sparsity)
     → Özellik seçimi
```

#### Elastic Net
```
L_total = L_data + λ₁||θ||₁ + λ₂||θ||²

L1 + L2 kombinasyonu
```

#### Dropout (Derin Ağlarda)
```
Training:
  Her nöron p olasılıkla "kapanır"
  
Test:
  Tüm nöronlar aktif
  Çıktılar p ile scale edilir

Etki: Co-adaptation'ı kırar
      Ensemble effect
```

#### Early Stopping ⭐
```
best_val_loss = ∞
patience_counter = 0

for epoch in epochs:
    train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        break  # DUR!

En basit ve etkili regularization!
```

### 📐 Bias-Variance Trade-off

**Matematiksel:**
```
E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²
               ↑          ↑       ↑
           Systematic Sensitivity  Irreducible
           error      to data     noise
```

**Denge:**
```
Error
  │
  │╲              ← Total Error
  │ ╲
  │  ╲  ╱        
  │   ╲╱         ← Bias (underfit bölgesi)
  │    ╲         ← Variance (overfit bölgesi)
  │     ╲___
  └──────────────→ Model Complexity
  Simple      Complex
  
  Underfit  Sweet   Overfit
            Spot
```

---

## 8️⃣ Feature Engineering & Scaling

### 📏 Neden Ölçekleme?

**Problem: Çarpık Loss Yüzeyi**

```
Özellik 1: [0, 1]
Özellik 2: [0, 10000]

Hessian:
H = [  1      0   ]
    [  0   10⁸   ]

Condition number κ = λ_max/λ_min = 10⁸ → KÖTÜ!

GD zikzak yapar:
        θ₂
         ↑
    ╱╲  │  ╱╲     ← Çok dik
   ╱  ╲ │ ╱  ╲
  ╱    ╲│╱    ╲
 ────────────────→ θ₁
```

**Çözüm: Standardization**

```
x' = (x - μ) / σ

Sonuç: μ=0, σ=1
κ ≈ 1 → Yuvarlak yüzey → GD düz iner!
```

### 📊 Ölçekleme Yöntemleri

#### Z-Score (Standardization)
```
x' = (x - μ) / σ

μ = sample mean
σ = sample std

Sonuç: μ'=0, σ'=1
Avantaj: Outlier'ları korur
```

#### Min-Max Scaling
```
x' = (x - x_min) / (x_max - x_min)

Sonuç: [0, 1] aralığı
Dezavantaj: Outlier'a hassas
```

#### Robust Scaling
```
x' = (x - median) / IQR

IQR = Q₃ - Q₁ (interquartile range)

Avantaj: Outlier'a robust
```

### 🏷️ Kategorik Değişkenler

#### One-Hot Encoding
```
Renk: ['kırmızı', 'mavi', 'yeşil']

     kırmızı  mavi  yeşil
     [1,      0,    0]
     [0,      1,    0]
     [0,      0,    1]

Avantaj: Sıra varsayımı yok
Dezavantaj: High cardinality'de boyut patlaması
```

#### Label Encoding
```
['kırmızı', 'mavi', 'yeşil'] → [0, 1, 2]

Dikkat: Sıra anlamı verir! (mavi > kırmızı?)
Tree-based modellerde OK, linear'da dikkat
```

#### Target Encoding
```
Kategori → Mean(target | kategori)

Dikkat: LEAKAGE riski!
Çözüm: Cross-validation ile yap
```

#### Learnable Embeddings
```
Kategori → Düşük boyutlu dense vektör (öğrenilir)

'istanbul' → [0.5, -0.3, 0.8, ...]  (d=128)
'ankara'   → [0.2, 0.4, -0.1, ...]

Derin ağlarda standart
```

---

## 9️⃣ Değerlendirme Metrikleri

### 📐 Regresyon Metrikleri

#### MSE / RMSE
```
MSE = (1/N) Σ (y - ŷ)²
RMSE = √MSE

Avantaj: Birim anlamlı (RMSE)
Dezavantaj: Outlier'a hassas
```

#### MAE
```
MAE = (1/N) Σ |y - ŷ|

Avantaj: Outlier'a robust
Dezavantaj: Türev sıfırda tanımsız
```

#### R² (Coefficient of Determination)
```
R² = 1 - SS_res / SS_tot

SS_res = Σ(y - ŷ)²
SS_tot = Σ(y - ȳ)²

Yorum: R²=0.85 → Model varyansın %85'ini açıklıyor
R² < 0 → Model ortalamadan kötü!
```

### 🎯 Classification Metrikleri

#### Confusion Matrix
```
                Predicted
             Pos      Neg
Actual Pos   TP   |   FN
       Neg   FP   |   TN
```

#### Precision & Recall
```
Precision = TP / (TP + FP)
  "Pozitif dediğimin ne kadarı doğru?"

Recall = TP / (TP + FN)
  "Gerçek pozitiflerin ne kadarını yakaladım?"
```

#### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Harmonik ortalama
→ Küçük değerlere ağırlık verir
```

#### ROC-AUC vs PR-AUC
```
ROC: TPR vs FPR
  → Dengeli sınıflarda iyi

PR: Precision vs Recall
  → İmbalanced data'da daha bilgilendirici

İmbalanced (örn. %1 pozitif):
  → PR-AUC kullan!
```

#### Calibration
```
Model: "%80 olasılıkla pozitif"
Gerçek: 100 örneğin 80'i pozitif mi?

Calibration curve:
  Mükemmel: y = x doğrusu
  Sapma var: Recalibration gerekli
```

---

## 🔟 Data Leakage (Veri Sızıntısı)

### 🚨 Tanım

**Leakage:** Test/validation bilgisinin eğitime sızması

**Sonuç:** Yapay yüksek performans, production'da felaket

### 📅 Temporal Leakage

```
❌ YANLIŞ:
  Rastgele split (2019, 2020 karışık)
  → Geleceği bilerek geçmişi tahmin!

✅ DOĞRU:
  Zamansal split:
  Train: 2019-01 ~ 2019-12
  Val:   2020-01 ~ 2020-03
  Test:  2020-04 ~ 2020-06
```

### 🔗 Target Leakage

```
Hedef: Kredi geri ödenecek mi?

❌ Özellik: "geri_ödeme_planı"
  → Bu bilgi ancak ödeme BAŞLADIĞINDA var!
  → Tahmin anında bilinmez

✅ Özellik: "gelir", "yaş", "kredi_skoru"
  → Tahmin anında bilinir
```

### 🧮 Preprocessing Leakage

```python
# ❌ YANLIŞ
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)  # Tüm veri!
X_train, X_test = train_test_split(X_all_scaled)

# ✅ DOĞRU
X_train, X_test = train_test_split(X_all)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Sadece train!
X_test_scaled = scaler.transform(X_test)        # Aynı scaler, transform only
```

### 🔍 Leakage Tespiti

**Checklist:**
```
1. Bu özellik tahmin anında bilinir mi?
2. Correlation > 0.95 hedefle? (şüpheli!)
3. Zaman sırasını bozduk mu?
4. Test'e dokunduk mu (preprocessing'de)?
5. Cross-validation düzgün mü? (grup bazlı gerekebilir)
```

---

## 1️⃣1️⃣ Probabilistik Bakış

### 🎲 Maximum Likelihood Estimation (MLE)

**Fikir:** Gözlenen verinin olasılığını maksimize et

```
θ* = argmax_θ p(Data | θ)
             ↑
        Likelihood

Log-likelihood (çalışmak kolay):
θ* = argmax_θ Σᵢ log p(yᵢ | xᵢ, θ)
```

### 📐 MSE'nin Probabilistik Kökeni

**Varsayım:** Gaussian noise

```
y = f_θ(x) + ε,  ε ~ N(0, σ²)

→ p(y|x,θ) = N(y; f_θ(x), σ²)

Log-likelihood:
log p(y|x,θ) = -(y - f_θ(x))² / (2σ²) + const

Maksimize et:
argmax Σ log p(yᵢ|xᵢ,θ)
= argmin Σ (yᵢ - f_θ(xᵢ))²
           ↑
          MSE!
```

**Sonuç:** MSE minimize = Gaussian MLE

### 🎯 Cross-Entropy'nin Kökeni

**Varsayım:** Bernoulli (binary classification)

```
y ∈ {0, 1}
p(y=1|x,θ) = σ(f_θ(x))

Log-likelihood:
log p(y|x,θ) = y log p + (1-y) log(1-p)

Maksimize et:
argmax Σ [yᵢ log pᵢ + (1-yᵢ) log(1-pᵢ)]
= argmin -Σ [yᵢ log pᵢ + (1-yᵢ) log(1-pᵢ)]
            ↑
    Binary Cross-Entropy!
```

**Sonuç:** BCE minimize = Bernoulli MLE

### 📊 MAP (Maximum A Posteriori)

**Bayes Kuralı:**
```
p(θ|D) ∝ p(D|θ) × p(θ)
  ↑        ↑        ↑
Posterior  Like   Prior
```

**MAP:**
```
θ* = argmax_θ p(θ|D)
   = argmax_θ [log p(D|θ) + log p(θ)]
   = argmin_θ [-log p(D|θ) - log p(θ)]
                    ↑            ↑
                  Loss      Regularization!
```

#### L2 = Gaussian Prior
```
Prior: θ ~ N(0, 1/λ)

log p(θ) = -λ/2 × ||θ||²

MAP:
argmin [Loss + λ/2 × ||θ||²]
                ↑
            L2 (Ridge)!
```

#### L1 = Laplace Prior
```
Prior: θ ~ Laplace(0, 1/λ)

log p(θ) = -λ × ||θ||₁

MAP:
argmin [Loss + λ × ||θ||₁]
                ↑
            L1 (Lasso)!
```

**Mesaj:**
> Loss ve regularization rastgele değil,
> probabilistik varsayımların sonucu!

---

## 1️⃣2️⃣ Sayısal Stabilite

### 🔴 Problemler

#### 1. NaN / Inf
```
Sebepler:
  - LR çok büyük
  - Gradient explosion
  - Sayısal taşma (exp, log)

Belirtiler:
  Loss → NaN
  Parametre → ±Inf
```

#### 2. Gradient Vanishing
```
Derin ağlarda:
  - Küçük gradientler çarpılır
  - Sonuç → 0

Çözüm:
  - ReLU (sigmoid yerine)
  - Batch/LayerNorm
  - Skip connections (Week 2+)
```

#### 3. Gradient Explosion
```
Gradientler büyür → patlar

Çözüm:
  - Gradient clipping
  - Küçük LR
  - Normalizasyon
```

### 🚑 İlk Yardım Protokolü

```
Problem: Training unstable, NaN

1. LR'ı yarıya indir
   → Çoğu zaman düzelir

2. Özellikleri standardize et
   → Loss yüzeyi yuvarlanır

3. zero_grad() kontrol et
   → Gradient accumulation olmasın

4. Loss/Metric doğru mu?
   → Regresyon ≠ CE

5. Shape/dtype/device?
   → print(x.shape, x.dtype, x.device)

6. Seed sabitle
   → Reprodüksiyon
```

---

## 1️⃣3️⃣ Deney Disiplini

### 🔬 Bilimsel Yöntem

```
1. GÖZLEM
   "Val loss platoda"

2. HİPOTEZ
   "LR çok büyük olabilir"

3. DENEY PLANI
   "LR'ı 0.01 → 0.001 değiştir"
   "Diğer her şey sabit"

4. ÖLÇÜM
   Val loss, train loss, süre

5. KARAR
   İyileşti mi? Neden?

6. LOG
   Sonuçları yaz

7. İTERE ET
   Sonraki hipotez
```

### 📊 Baseline Stratejisi

```
Level 0: Dummy
  - Mean/mode predictor
  - Random guess
  → Baseline oluştur

Level 1: Simple
  - Linear/Logistic
  → Beat et

Level 2: Standard
  - RF/XGBoost (tabular)
  - Standard CNN (vision)
  → Beat et

Level 3: Custom
  - Domain-specific
  → Gerektiğinde
```

### 📝 Ablation Studies

```
Full model: A + B + C + D → 0.85

Test:
  A + B + C     → 0.82  (D'nin katkısı: +0.03)
  A + B     + D → 0.80  (C'nin katkısı: +0.05)
  A     + C + D → 0.78  (B'nin katkısı: +0.07)
      B + C + D → 0.60  (A'nın katkısı: +0.25)

Sonuç: A en kritik, sonra B, C, D
```

### 🎲 Hyperparameter Search

#### Random Search
```
LR: log-uniform(1e-5, 1e-1)
L2: log-uniform(1e-6, 1e-2)
batch: choice([32, 64, 128, 256])

N=20 rastgele kombinasyon dene

Avantaj: Grid'den daha verimli (kanıtlanmış)
```

#### Bayesian Optimization
```
1. İlk denemeler (N=5)
2. Gaussian Process fit
3. Acquisition function → Sonraki nokta
4. Tekrar

Avantaj: Az deneyle iyi sonuç
```

---

## 1️⃣4️⃣ Sözlük: Formal Tanımlar

```
┌──────────────────────────────────────────────────┐
│ TERİM                   TANIM                    │
├──────────────────────────────────────────────────┤
│ Model (f_θ)             x ↦ ŷ fonksiyonu         │
│ Parametre (θ)           Öğrenilen ağırlıklar     │
│ Hiperparametre          Kullanıcı seçimi (η, λ)  │
│ Loss (L)                Hata fonksiyonu          │
│ Gradient (∇L)           Eğim vektörü             │
│ Learning Rate (η)       Adım büyüklüğü           │
│ Optimizer               GD algoritması           │
│ Epoch                   Tüm veriyi bir görme     │
│ Batch                   Mini grup (32-256)       │
│ Overfit                 Train ↓, Val ↑           │
│ Underfit                Train ↑, Val ↑           │
│ Regularization          Penalty term (λ||θ||²)   │
│ Early Stopping          Val bazlı durdurma       │
│ Validation              Hiperparametre seçimi    │
│ Test                    Final değerlendirme      │
│ i.i.d.                  Bağımsız, özdeş dağılım  │
│ Generalization          Görmediğine genelleme    │
│ MLE                     Max likelihood           │
│ MAP                     Max a posteriori         │
└──────────────────────────────────────────────────┘
```

---

## 1️⃣5️⃣ Neden Lineer Regresyon?

### 🎯 Pedagojik Gerekçeler

#### 1. Konveks Problem
```
L(w) = ||y - Xw||²

→ Tek global minimum
→ Local minimum yok
→ GD davranışı TEMİZ
```

#### 2. Analitik Çözüm Var
```
Normal Equations:
w* = (X^T X)^(-1) X^T y

→ GD ile karşılaştırabilirsin
→ Doğruluğu kontrol edebilirsin
```

#### 3. Tüm Kavramlar Mevcut
```
✓ Loss (MSE)
✓ Gradient
✓ Optimization (GD)
✓ Regularization (L2)
✓ Overfitting
✓ Val/Test split
✓ Metrics (R², RMSE)
✓ Feature scaling etkisi

→ Kamp eğitimi gibi!
```

#### 4. Görsel Anlama
```
2D:
  y
  │  ●
  │    ●  ●
  │  ●   ──── Fit line
  │ ●  ●
  └───────────→ x

Herkes anlar!
```

#### 5. Transfer Edilebilir
```
Linear Regression'da öğrendiğin:

Ölçekleme    → MLP, CNN, Transformer
LR seçimi    → Her model
Overfit      → Her model
Early stop   → Her model
Val split    → Her model

→ TEMEL BURADA ATILIR!
```

---

## 1️⃣6️⃣ Week 0 → Week 1 Köprüsü

### 📚 Bugün Ne Öğrendik?

**Kavramsal:**
```
✓ ML = Veriyle fonksiyon öğrenme
✓ Train/Val/Test neden şart
✓ Loss = Hata ölçüsü
✓ GD = Gradyan takip ederek iniş
✓ Overfit = Ezber (regularization ile önle)
✓ Ölçekleme = Optimizasyonu kolaylaştır
```

**Matematiksel:**
```
✓ θ* = argmin_θ L(θ)
✓ θ ← θ - η∇L
✓ MSE ← Gaussian MLE
✓ CE ← Bernoulli MLE
✓ L2 ← Gaussian prior
```

**Pratik:**
```
✓ AdamW + küçük L2 (başlangıç)
✓ Early stopping (overfit önleme)
✓ Standardization (feature scaling)
✓ Val ile ayarla, Test'e dokunma
✓ Seed sabitle (repro)
```

### 🚀 Week 1'de Ne Yapacağız?

```
1. Sentetik veri oluştur
   y = wx + b + ε

2. Manuel GD
   → Gradient'i KENDİN hesapla
   → Autograd'ı ÇIPLAK gör

3. nn.Module ile GD
   → PyTorch'un gücünü kullan
   → Workflow'u öğren

4. Train/Val split
   → Overfit'i CANLI izle

5. Early stopping
   → Regularization etkisini GÖR

6. Metrikler
   → RMSE, R² hesapla, yorumla
```

### ✅ Hazır mısın? (Self-Check)

```
□ "Model nedir?" → Parametrik fonksiyon
□ "Loss nedir?" → Hata ölçüsü (tek sayı)
□ "Gradient nedir?" → Eğim vektörü
□ "GD nasıl çalışır?" → θ ← θ - η∇L
□ "Overfit nedir?" → Train iyi, Val kötü
□ "Neden ölçekleme?" → Loss yüzeyi yuvarlanır
□ "MSE nereden gelir?" → Gaussian MLE
□ "L2 nereden gelir?" → Gaussian prior
□ "Early stopping nedir?" → Val kötüleşince dur
□ "Test ne zaman?" → Sadece final'de

Hepsi ✓ ise → Week 1'e HAZIRSIN! 🎓
```

---

## 1️⃣7️⃣ Tek Paragraf Özet

> **Makine öğrenmesi**, veriyle uyumlu bir **fonksiyonu bulma** sanatıdır. Bunu, bir **kayıp pusulası** (loss) yardımıyla, parametreleri **gradyan adımlarıyla** (GD) ayarlayarak yaparız. **Doğrulama kümesi** vicdanımızdır: ezberlediğimiz anda bizi durdurur. **Düzenleme** (L2/L1), **ölçekleme**, **doğru metrik** ve **dürüst deney**; güvenilir sonuçların dört ayağıdır. Bu temeller yerindeyse, üstüne kuracağımız her model—lineer, MLP, hatta devasa Transformer—**anlaşılır** ve **kontrol edilebilir** olur.

---

## 🎓 Sonraki Adım

### 📖 Okuma Rotası

```
✅ theory_intro.md (lise)
✅ theory_core_concepts.md (üniversite) ← ŞU AN
⬜ theory_foundations.md (sezgisel detay)
⬜ theory_mathematical.md (matematiksel derinlik)
⬜ theory_advanced.md (pratik saha)
⬜ Setup & Week 1
```

### 💪 Pratik Alıştırma (30 dk)

**3 Problem Analiz Et:**

```
Problem 1: Regresyon
  - Görev: Ev fiyatı
  - Özellikler: m², oda, yaş
  - Loss: MSE (neden? → Gaussian varsayımı)
  - Metric: RMSE (birim anlamlı)
  - Regularization: L2 (neden? → Gaussian prior)
  - Split: Rastgele 70/15/15

Problem 2: Dengesiz Sınıflama
  - Görev: Dolandırıcılık (%1)
  - Loss: BCE + class weight
  - Metric: PR-AUC, F1 (neden? → imbalanced)
  - Regularization: Dropout
  - Split: Stratified (oran koru)

Problem 3: Zaman Serisi
  - Görev: Satış tahmini
  - Loss: MAE (neden? → outlier robust)
  - Metric: MAPE
  - Leakage riski: Temporal!
  - Split: Zamansal (geçmiş→gelecek)
```

### 🧮 Mini Quiz (Self-Test)

```
Q1: MSE neden Gaussian MLE'ye eşit?
A1: Gaussian noise varsayımında log-likelihood
    maksimize etmek = MSE minimize etmek

Q2: L2 regularization ne anlama gelir?
A2: Parametrelere Gaussian prior koyduk
    → Küçük ağırlıkları tercih ediyoruz

Q3: Early stopping neden regularization?
A3: Eğitimi erken durdurarak ağırlıkların
    büyümesini engelliyoruz → implicit L2

Q4: Adam neden SGD'den hızlı?
A4: Her parametreye adaptif LR
    → Dik yönlerde hızlı, yassı yönlerde yavaş
```

---

**🎉 Tebrikler! Week 0 Core Concepts tamamlandı!**

**Fark:**
- theory_intro.md → Günlük dil, sıfır matematik
- theory_core_concepts.md → Formal tanımlar, hafif matematik, probabilistik temeller

**Şimdi yapabilirsin:**
- Week 1 kodunu ANLAYARAK yaz
- "Neden MSE?" "Neden L2?" → Cevapla
- Formüllerin nereden geldiğini BİL

**Hazır ol, Week 1 pratik zamanı!** 🚀
