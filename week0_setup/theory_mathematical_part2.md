# Week 0: Matematiksel Temeller Part 2 - İleri Konular

**NovaDev v1.0 - Probabilistik & Teorik Derinlik**

> "Neden bu loss? Neden bu regularization? → Probabilistik kökenler"

---

## 5️⃣ Sayısal Koşullar & Curvature

### 🎯 Condition Number (Koşul Sayısı)

**Tanım:** Loss yüzeyinin "ne kadar çarpık" olduğunu ölçer.

```
κ(H) = λ_max / λ_min
       ↑         ↑
   En büyük  En küçük
   eigenvalue
   
H = Hessian matrix (2. türevler)
```

#### Görsel Açıklama

**Küçük κ (iyi koşullanmış):**
```
     ●         Dairevi yüzey
   ╱   ╲       
  │  ●  │      ← Minimum
   ╲   ╱
     ●
     
GD doğrudan iner ✓
```

**Büyük κ (kötü koşullanmış):**
```
     ╱╲        Eliptik yüzey
    ╱  ╲
   ╱    ╲      ← Çok uzun
  │  ●   │     ← Çok dar
  │      │
   ╲    ╱
    ╲  ╱
     ╲╱
     
GD zikzak yapar ✗
```

### 📐 Neden Ölçekleme İşe Yarar?

#### Matematiksel Kanıt (Basitleştirilmiş)

**Ölçeksiz:**
```
L(w) = (1/2) Σ (y - w_1×1 - w_2×1000)²

Hessian'ın eigenvalue'ları:
λ_1 ≈ 1
λ_2 ≈ 1,000,000

κ = 1,000,000 / 1 = 10⁶  ← ÇARPIK!
```

**Ölçeklenmiş:**
```
x_1' = x_1 / std(x_1)
x_2' = x_2 / std(x_2)

λ_1 ≈ 1
λ_2 ≈ 1

κ = 1 / 1 = 1  ← YUVARLAK!
```

### 🧮 İkinci Türev Sezgisi

**Birinci türev (gradient):** Yön
**İkinci türev (Hessian):** Eğrilik

```
f''(x) < 0: ∩  (konkav, maksimum civarı)
f''(x) = 0: ─  (inflection point)
f''(x) > 0: ∪  (konveks, minimum civarı)
```

**Newton's Method:**
```
θ_new = θ_old - H^(-1) × ∇L
                  ↑
             Eğriliği kompanse et
```

**Problem:** Hessian hesabı **O(d²)** → Çok pahalı!

**Çözüm:** Adam/Momentum Hessian'ı **yaklaşık** kullanır
- Ucuz
- Pratikte yeterli

---

## 6️⃣ Probabilistik Bakış: Neden MSE? Neden CE?

### 🎲 Maximum Likelihood Estimation (MLE)

**Temel Fikir:**
> "Gözlediğim verinin olasılığını maksimize et"

```
θ* = argmax_θ p(Data | θ)
            ↑
      Likelihood (olabilirlik)
```

**Log trick (rahat çalışmak için):**
```
log p(D|θ) = Σ log p(y_i | x_i, θ)
            ↑
       Log-likelihood
```

### 📐 MSE'nin Probabilistik Kökeni

#### Varsayım: Gaussian Noise

```
y = f_θ(x) + ε
ε ~ N(0, σ²)  ← Normal dağılım

→ p(y|x,θ) = N(f_θ(x), σ²)
```

#### Log-Likelihood Türetimi

```
log p(y|x,θ) = log N(y; f_θ(x), σ²)
             = -(y - f_θ(x))² / (2σ²) + const
             
Maksimize et:
argmax Σ log p(y_i|x_i,θ)
= argmax Σ [-(y_i - f_θ(x_i))²]
= argmin Σ (y_i - f_θ(x_i))²
           ↑
          MSE!
```

**Sonuç:**
```
MSE minimize etmek
    ≡
Gaussian noise varsayımı altında
likelihood maksimize etmek
```

### 🎯 Cross-Entropy'nin Probabilistik Kökeni

#### Varsayım: Bernoulli Distribution (Binary)

```
y ∈ {0, 1}
p(y=1|x,θ) = σ(f_θ(x))  ← Sigmoid
```

#### Log-Likelihood Türetimi

```
p(y|x,θ) = p^y × (1-p)^(1-y)

log p(y|x,θ) = y log p + (1-y) log(1-p)

Maksimize et:
argmax Σ [y_i log p_i + (1-y_i) log(1-p_i)]
= argmin Σ [-y_i log p_i - (1-y_i) log(1-p_i)]
            ↑
     Binary Cross-Entropy!
```

**Sonuç:**
```
BCE minimize etmek
    ≡
Bernoulli varsayımı altında
likelihood maksimize etmek
```

### 🎓 MAP: Bayesian Twist

#### Maximum A Posteriori

**MLE:** p(θ | Data)
**MAP:** p(θ | Data) × p(θ)
                        ↑
                    Prior belief
                    (ön bilgi)

**Bayes Kuralı:**
```
p(θ|D) ∝ p(D|θ) × p(θ)
  ↑        ↑        ↑
Posterior  Like   Prior
```

**Log formunda:**
```
log p(θ|D) = log p(D|θ) + log p(θ) + const

argmax = argmin [-log p(D|θ) - log p(θ)]
                    ↑            ↑
                  Loss      Regularization!
```

### 📊 Regularization'ın Probabilistik Anlamı

#### L2 Regularization = Gaussian Prior

```
Prior: θ ~ N(0, 1/λ)

log p(θ) = -λ/2 × Σ θ_j²

MAP:
argmin [Loss + λ/2 × Σ θ_j²]
                      ↑
                  Ridge (L2)
```

**Yorum:** "Parametreler sıfıra yakın olsun tercihim"

#### L1 Regularization = Laplace Prior

```
Prior: θ ~ Laplace(0, 1/λ)

log p(θ) = -λ × Σ |θ_j|

MAP:
argmin [Loss + λ × Σ |θ_j|]
                    ↑
                Lasso (L1)
```

**Yorum:** "Parametreler TAM SIFIR olsun tercihim" (sparsity)

### 💡 Özet Tablo

```
┌─────────────┬──────────────┬─────────────────┐
│  Varsayım   │     Loss     │   Regularization│
├─────────────┼──────────────┼─────────────────┤
│ Gauss Noise │     MSE      │    L2 (Gauss)   │
│ Bernoulli   │     BCE      │    L1 (Laplace) │
│ Categorical │     CE       │    -            │
│ Poisson     │  Poisson Loss│    -            │
└─────────────┴──────────────┴─────────────────┘
```

**Mesaj:**
> Loss ve regularization **rastgele** değil;
> probabilistik varsayımların matematiksel sonuçları.

---

## 7️⃣ Bias-Variance Trade-off: Matematiksel Ayrıştırma

### 📐 Formal Tanım

**Beklenen test hatası:**
```
E[(y - ŷ)²] = Bias² + Variance + Irreducible Error
               ↑        ↑              ↑
           Systematic Sensitivity  Gürültü
           hata      to data      (ε²)
```

### 🧮 Türetim (Basitleştirilmiş)

```
Gerçek: y = f(x) + ε,  ε ~ N(0, σ²)
Tahmin: ŷ = f̂(x)

E[(y - ŷ)²] 
= E[(y - f + f - ŷ)²]
= E[(f - ŷ)²] + E[ε²]
  ↑             ↑
  Model err   Irreducible

E[(f - ŷ)²]
= E[(f - E[ŷ] + E[ŷ] - ŷ)²]
= (f - E[ŷ])² + E[(E[ŷ] - ŷ)²]
    ↑              ↑
   Bias²        Variance
```

### 📊 Görsel Açıklama

```
        Bias
         ↓
    ┌────┼────┐  ← Model ortalaması
    │    ●    │
    │  ● ● ●  │  ← Farklı train set'lerde
    │    ●    │     model tahminleri
    └─────────┘
         ↕
      Variance
      
      ⊕ = Gerçek hedef
```

**Düşük Bias + Düşük Variance:**
```
    ⊕
  ● ● ●  ← Model tutarlı ve doğru
```

**Yüksek Bias + Düşük Variance:**
```
    ⊕
    
  ● ● ●  ← Model tutarlı ama yanlış
```

**Düşük Bias + Yüksek Variance:**
```
●   ⊕   ●
    ●     ← Model bazen doğru ama
  ●         tutarsız (overfit)
```

### 🎯 Kapasite-Hata İlişkisi

```
Error
  │
  │╲              ← Total Error
  │ ╲
  │  ╲  ╱        
  │   ╲╱         ← Bias
  │    ╲         ← Variance
  │     ╲___
  └──────────────→ Model Capacity
  Basit      Karmaşık
  
  Underfit  Sweet   Overfit
            Spot
```

### 💊 Regularization = Variance Kontrolü

```
Regularization ↑
  → Kapasite ↓
  → Variance ↓
  → Bias ↑ (biraz)
  
Amaç: Toplam hatayı minimize et
```

---

## 8️⃣ Regularization Derinliği

### 🎚️ L2 (Ridge) Matematiksel Analiz

```
L_ridge = MSE + λ/2 × ||w||²

Gradient:
∇L_ridge = ∇MSE + λ × w

Update:
w ← w - η × (∇MSE + λw)
  = w(1 - ηλ) - η∇MSE
      ↑
   Weight decay
   (her step'te biraz küçült)
```

**Etki:** Ağırlıkları **küçük** tutar (sıfıra itmez)

### 🎚️ L1 (Lasso) Matematiksel Analiz

```
L_lasso = MSE + λ × ||w||₁

Gradient (subdifferential):
∂L_lasso = ∇MSE + λ × sign(w)

Update:
w ← w - η × (∇MSE + λ sign(w))
                      ↑
                 Sabit büyüklükte
                 itme (sıfıra)
```

**Etki:** Ağırlıkları **TAM SIFIR** yapar (feature selection)

### 📊 L1 vs L2 Geometri

```
L2 (Ridge):
    w_2
     ↑
   ╱───╲   ← Daire (w₁²+w₂²=c)
  │  ●  │
   ╲───╱
 ─────●─────→ w_1
      │
   Kesişme genelde
   eksen dışında

L1 (Lasso):
    w_2
     ↑
    ╱│╲    ← Elmas (|w₁|+|w₂|=c)
   ╱ │ ╲
  ╱  ●  ╲  ← Kesişme genelde
 ────●────→ w_1  eksen üzerinde
     │            (sparse!)
```

### 🔄 Early Stopping = Implicit Regularization

**Gözlem:**
```
Eğitim devam ettikçe:
  Train loss ↓↓↓
  Val loss ↓ → ↑
           ↑
    Bu noktada dur!
```

**Matematiksel Açıklama:**

```
Gradient descent her step'te:
w_t = w_0 - ηt × ∇̄L
         ↑
    Toplam gradient etkisi

t küçük → w küçük → Implicit L2
t büyük → w büyük → Overfit risk
```

**Kanıt (sketch):**
- Early stop ≈ ağırlıkları büyümeden durdur
- Bu ≈ L2'nin etkisi (küçük ağırlık tercihi)

### 🎲 Dropout (Derin Ağlarda)

```
Training:
  Her neuron p olasılıkla "öl"
  → Ensemble effect
  
Test:
  Tüm neuronlar aktif ama
  çıktıları p ile scale et
```

**Neden İşe Yarar:**
- Co-adaptation'ı kırar
- Ensemble benzeri
- Implicit regularization

---

## 9️⃣ Değerlendirme Metrikleri: Derin Bakış

### 📊 Regression Metrikleri

#### R² (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)

SS_res = Σ(y - ŷ)²     ← Residual sum of squares
SS_tot = Σ(y - ȳ)²     ← Total sum of squares

Yorum:
R² = 0.85 → Model varyansın %85'ini açıklıyor
R² < 0   → Model mean'den kötü!
```

#### RMSE vs MAE
```
RMSE = √(MSE)
       ↑
   Scale geri kazanıldı
   (yorumlanabilir)

MAE  = Mean absolute error
       Robust to outliers
```

### 🎯 Classification Metrikleri

#### Confusion Matrix Ayrıştırma

```
             Predicted
           Pos      Neg
Actual Pos  TP  |   FN    ← Recall = TP/(TP+FN)
       Neg  FP  |   TN
            ↓
    Precision = TP/(TP+FP)
```

**Precision:** "Pozitif dediğim ne kadar doğru?"
**Recall:** "Gerçek pozitiflerin ne kadarını yakaladım?"

#### F1 Score (Harmonic Mean)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Neden harmonic mean?
  → Küçük değerlere ağırlık verir
  → Precision=0.9, Recall=0.1
    Arithmetic: 0.5  (yanıltıcı!)
    Harmonic: 0.18   (gerçekçi)
```

#### ROC-AUC vs PR-AUC

**ROC Curve:**
```
TPR vs FPR (eşikler değişirken)

TPR = TP / (TP + FN)  ← True Positive Rate
FPR = FP / (FP + TN)  ← False Positive Rate
```

**PR Curve:**
```
Precision vs Recall

İmbalanced data'da daha bilgilendirici!
```

**Örnek (Imbalanced):**
```
Pozitif: %1
Negatif: %99

Dummy model: "Hep negatif de"
  Accuracy: %99  ← Yanıltıcı!
  Recall: 0      ← Gerçek
```

#### Calibration (Kalibrasyon)

```
Model: "Bu örnek %80 olasılıkla pozitif"
Gerçek: 100 örneğin 80'i pozitif mi?

Calibration curve:
  y = x doğrusu → Mükemmel
  Sapma → Kalibrasyon gerekli
```

**Neden Önemli:**
- Tıp, finans gibi alanlarda
- Olasılık tahminleri kritik

---

## 🔟 Deney Disiplini: Bilimsel Yöntem

### 🔬 Hipotez Odaklı Deney

```
1. Gözlem:    Val loss platoya ulaştı
2. Hipotez:   LR çok büyük / Overfit
3. Deney:     LR'ı yarıya indir
4. Ölçüm:     Val loss değişimi
5. Sonuç:     Kabul/Ret
6. Öğrenme:   Sonraki hipotez
```

### 📊 Baseline Stratejisi

```
Level 0: Dummy
  - Mean predictor (regression)
  - Most frequent class (classification)
  
Level 1: Simple
  - Linear regression
  - Logistic regression
  
Level 2: Standard
  - RF/XGBoost (tabular)
  - ResNet (vision)
  - BERT (NLP)
  
Level 3: Custom
  - Domain-specific architecture
```

**Kural:** Her level'ı beat et, sonra geç!

### 🔍 Ablation Studies

```
Full model: A + B + C + D → 0.85

Ablation:
  A + B + C     → 0.82  (D katkısı: +0.03)
  A + B     + D → 0.80  (C katkısı: +0.05)
  A     + C + D → 0.78  (B katkısı: +0.07)
      B + C + D → 0.60  (A katkısı: +0.25)
```

**Sonuç:** A en kritik, B ve C önemli, D küçük etki

### 🎲 Hyperparameter Search

#### Grid Search
```
LR: [0.001, 0.01, 0.1]
L2: [0.0001, 0.001, 0.01]

→ 3×3 = 9 deney
```

**Dezavantaj:** Kombinatoryal patlama

#### Random Search
```
LR: Uniform(0.0001, 0.1)
L2: Log-uniform(1e-5, 1e-2)

→ N rastgele nokta
```

**Avantaj:** Daha verimli (kanıtlanmış!)

#### Bayesian Optimization
```
1. İlk denemeler yap
2. Gaussian Process fit et
3. Acquisition function ile sonraki noktayı seç
4. Tekrarla
```

**Avantaj:** Az deneyle iyi sonuç

### 📝 Experiment Logging Template

```markdown
## Experiment 2025-10-06-003

### Hypothesis
LR=0.01'de overfit var, 0.001'e düşürünce düzelir mi?

### Setup
- Model: LinearRegression(input=10)
- Optimizer: Adam(lr=0.001, weight_decay=1e-4)
- Batch: 32
- Epochs: 100
- Seed: 42
- Device: MPS

### Baseline
Exp-002: lr=0.01 → Train=0.05, Val=0.15 (overfit!)

### Results
Train Loss: 0.12 (+0.07 vs baseline)
Val Loss:   0.10 (-0.05 vs baseline) ✓
Time: 45s

### Analysis
- Overfit azaldı ✓
- Training biraz yavaş
- Val iyileşti → Hipotez doğru

### Next Steps
- lr=0.001 ile L2'yi arttır?
- Early stopping ekle?
```

---

## 1️⃣1️⃣ Tensors & Autograd: Derin Sezgi

### 🧊 Tensor = Data + Device + Dtype

```python
x = torch.randn(10, 5, device='mps', dtype=torch.float32)
         ↑       ↑        ↑            ↑
       Yaratıcı Shape  Nerede?    Ne tür sayı?
```

**Neden Önemli:**
- Device: Performans (CPU/GPU/MPS)
- Dtype: Bellek + sayısal stabilite
- Shape: Her işlemde kontrol şart

### 📡 Broadcasting: Otomatik Genişleme

```
(10, 1) + (1, 5) → (10, 5)

Kural:
1. Sağdan hizala
2. Boyutlar eşit VEYA biri 1 olmalı
3. Eksik boyut 1 kabul edilir

Örnekler:
(3, 1, 5) + (4, 5) → (3, 4, 5) ✓
(3, 2) + (3, 1) → (3, 2) ✓
(3, 2) + (2, 3) → ERROR ✗
```

### 🔄 Computational Graph

```
     x              y
      ↓             ↓
    [Linear]      [Square]
       ↓             ↓
       a    →   [Multiply] → b
                     ↓
                  [Sum] → L
```

**Forward:**
```
x, y → a, b → L  (değerleri hesapla)
```

**Backward:**
```
∂L/∂L = 1
∂L/∂b = ∂L/∂L × ∂L/∂b  (chain rule)
∂L/∂a = ∂L/∂b × ∂b/∂a
∂L/∂x = ∂L/∂a × ∂a/∂x
```

**Autograd = Chain Rule Otomasyonu**

### ⚠️ Gradient Accumulation

```python
# ❌ YANLIŞ
for epoch in range(epochs):
    loss = compute_loss()
    loss.backward()  # Gradientler BİRİKİYOR!
    optimizer.step()

# ✅ DOĞRU
for epoch in range(epochs):
    optimizer.zero_grad()  # Önce temizle
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

**Neden:** `backward()` **add** yapar, **set** değil!

---

## 1️⃣2️⃣ Lineer Regresyon: Neden İlk Adım?

### 🎯 Pedagojik Nedenler

#### 1. Kapalı Form Çözümü Var
```
Normal Equations:
w* = (X^T X)^(-1) X^T y

→ Analitik çözüm biliyoruz
→ GD'nin doğruluğunu kontrol edebiliriz
```

#### 2. Konveks Problem
```
L(w) = ||y - Xw||²

→ Tek global minimum
→ Local minimum yok
→ Optimizasyon davranışı temiz
```

#### 3. Görsel Anlama Kolay
```
2D plot:
  y
  │  ●
  │    ●  ●
  │  ●    ─── Fit line
  │ ●  ●
  └─────────→ x
```

#### 4. Tüm Kavramlar Var
- Loss (MSE)
- Gradient
- Optimization (GD)
- Regularization (Ridge/Lasso)
- Overfitting
- Val/Test split
- Metrics (R², RMSE)

### 📐 Geometrik Sezgi

**Linear regression = Projeksiyon**

```
y ∈ R^n  (veri uzayı)
ŷ ∈ span(X)  (özellik uzayı)

ŷ = argmin ||y - v||²
    v ∈ span(X)
    
→ y'nin span(X)'e ortogonal projeksiyonu!
```

---

## 1️⃣3️⃣ Saha Hataları: Gerçek Hayat Deneyimi

### 🐛 En Sık 10 Hata

#### 1. Data Leakage
```
❌ Normalizasyon tüm veride
✅ Train'den öğren, test'e uygula
```

#### 2. Yanlış Loss/Metric
```
❌ Regression'a CrossEntropy
❌ Binary classification'a MSE
```

#### 3. Val = Test Karıştırma
```
❌ Hiperparametre seçimi test'te
✅ Val'de seç, test'e bir kez bak
```

#### 4. LR Felaketi
```
Belirti: Loss → NaN
Çözüm: LR'ı /10 yap
```

#### 5. Shape Uyumsuzluğu
```
Belirti: "RuntimeError: size mismatch"
Çözüm: Her adımda print(x.shape)
```

#### 6. Device Mismatch
```
Belirti: "Expected CPU tensor but got CUDA"
Çözüm: Tüm tensor'leri aynı device'a koy
```

#### 7. Gradient Sıfırlama Unutma
```
Belirti: Loss patlıyor, training unstable
Çözüm: optimizer.zero_grad() her iterasyon
```

#### 8. Seed Yok
```
Belirti: Sonuçlar tekrarlanamıyor
Çözüm: set_seed(42) başta
```

#### 9. Val Loss İzlememe
```
Belirti: Train iyi, test kötü (overfit)
Çözüm: Early stopping kur
```

#### 10. Feature Scale Unutma
```
Belirti: LR hassas, yakınsama yavaş
Çözüm: StandardScaler kullan
```

### 🔍 Debug Protokolü (6 Adım)

```
1. LR Kontrolü
   □ Çok büyük mü? (loss zıplar)
   □ Çok küçük mü? (loss hareket etmez)

2. Data Sızıntısı
   □ Train/val/test doğru ayrıldı mı?
   □ Normalizasyon train'den öğrenildi mi?

3. Shape/Device
   □ Tüm tensor'ler aynı device'da mı?
   □ Shape'ler beklediğin gibi mi?

4. Gradient Akışı
   □ zero_grad() her iterasyonda mı?
   □ backward() çağrılıyor mu?
   □ requires_grad=True mı?

5. Loss/Metric Doğru
   □ Problem tipi ile uyumlu mu?
   □ Reduction (mean/sum) doğru mu?

6. Reproduksiyon
   □ Seed sabitlendi mi?
   □ Versiyonlar loglandı mı?
```

---

## 1️⃣4️⃣ Week 0 → Week 1 Köprü

### 🎓 Week 1'de Yapacaklarının Teorisi

#### 1. Sentetik Veri
```
y = 3x + 2 + ε,  ε ~ N(0, 0.1)

Neden sentetik?
  → Gerçeği biliyoruz
  → Kontrol bizde
  → Debug kolay
```

#### 2. Manuel Gradient Descent
```python
# Autograd YOK, sen hesapla:
∂L/∂w = ?
∂L/∂b = ?

→ Zincir kuralını ÇIPLAK görürsün
```

#### 3. nn.Module ile Eğitim
```python
# Autograd VAR, PyTorch halleder:
loss.backward()

→ Pratik workflow öğrenirsin
```

#### 4. Train/Val Split
```python
train_data, val_data = split(data, 0.8)

→ Overfitting'i CANLI izlersin
```

#### 5. Erken Durdurma
```python
if val_loss > best_val_loss for N epochs:
    break

→ Implicit regularization görürsün
```

### 🎯 Amaç

> "Neden böyle?" sorusuna **akıcı** cevap verebilir hale gel.
> Kod sadece resmileştirme.

---

## 📚 Kapanış: Edebiyat Kristalize

**Öğrenme = Hata yüzeyinde yürüyen gezgin**

**Elinde:**
- Eğim pusulası (gradient)
- Adım ölçer (LR)
- Harita yok! (keşfediyorsun)

**Vicdanın:**
- Validation seti
- "Ezberleme" sinyali

**Stratejin:**
- AdamW ile momentum al
- SGD ile sakin sulara çekil
- Regularization ile dizginle

**Kazandığında:**
- Neden kazandığını anlat
- Kaybettiğinde sebebini bil

**Bugün yerleştirdik:** Bu "neden"leri

---

## ✅ Final Self-Assessment

### Matematiksel Derinlik
- [ ] MLE → MSE türetimini anlıyorum
- [ ] MAP → Regularization bağlantısını görüyorum
- [ ] Bias-variance ayrıştırmasını formüle edebiliyorum
- [ ] Condition number'ın etkisini biliyorum

### Pratik Beceri
- [ ] Loss seçimini probabilistik temelle savunabiliyorum
- [ ] Regularization'ı prior olarak yorumlayabiliyorum
- [ ] Metrik seçimini probleme göre yapabiliyorum
- [ ] Debug protokolünü ezberim

### Bütünsel Anlayış
- [ ] "Neden bu loss?" → Anlatabiliyorum
- [ ] "Neden bu optimizer?" → Sebepliyorum
- [ ] "Neden overfit oldu?" → Teşhis ediyorum
- [ ] Literatür okuyabilecek temele sahibim

**Hepsi ✅ ise:** Week 1 linear regression'a **hazırsın**! 🚀

---

## 🚀 Sıradaki Adım

```bash
cd /Users/onur/code/novadev-protocol
source .venv/bin/activate

# Teori oturdu mu? Test et:
python week1_tensors/linreg_manual.py

# Her satırda "neden?" diye sor kendine
# Cevabı theory_mathematical.md'de var!
```

**Başarı Ölçütü:**
> Week 1'de kod yazarken:
> "Aha! İşte bu yüzden MSE!"
> "İşte bu yüzden L2!"
> diyebilmen.

---

**Hazır mısın? Matematiksel temeller tamam. Artık pratik zamanı!** 💪
