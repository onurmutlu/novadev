# Week 0: Makine Öğrenmesi Temelleri - Ders Notları

**NovaDev v1.0 - Foundation Week**

> "Teori olmadan pratik kördür; pratik olmadan teori anlamsızdır."

---

## 📖 Bu Döküman Hakkında

Bu notlar, **makine öğrenmesinin temel kavramlarını** sıfırdan, sezgisel bir şekilde açıklar. Amaç: kod yazmadan önce **zihinsel modeli** doğru kurmak. Kahvenizi alın, rahat bir yere oturun. ☕

**Hedef:** Week 1'e başlamadan önce bu sayfadaki tüm kavramları **kendi kelimelerinizle** anlatabilmek.

---

## 0️⃣ Neyi Öğreniyoruz? (Bir Cümle)

**Makine Öğrenmesi = Veri → Fonksiyon öğrenmek**

Veri bize hikâyeyi fısıldar, model o fısıltıyı **parametreleri** ayarlayarak fonksiyona çevirir.

**Dikkat:** "Doğru" yok; **en işe yarayan** var.

### 🎯 Zihinsel Çerçeve

```
Veri (Ham Bilgi)  →  Model (Öğrenen)  →  Tahmin (Çıktı)
     ↓                      ↓                  ↓
  Örnek:              Parametreler:        Sonuç:
  Ev metrajı          w, b              Fiyat tahmini
```

**Analoji:** Bir çocuğa bisiklet sürmeyi öğretiyorsun. Düşer, kalkar, hata yapar. Sonunda denge noktasını **kendisi** bulur. Sen sadece ortamı ve dönütü sağlarsın.

---

## 1️⃣ Model Nedir? (Sihir Değil, Fonksiyon)

### Matematiksel Tanım

Model = f_θ(x)

Burada:
- **x**: Girdi (özellikler / features)
- **θ** (theta): Parametreler (ağırlıklar / weights)
- **f**: Fonksiyon (genelde karmaşık, ama matematiksel)
- **Çıktı**: Tahmin (ŷ)

### 🔍 Daha Derin Bakış

**Bizim İşimiz:** İyi bir θ bulmak.

"İyi" ne demek?
- Eğitim verisinde **düşük hata**
- Test verisinde de **genelleme** yapabiliyor
- **Stabil** (küçük veri değişikliğinde çökmüyor)

### 💡 Zihinsel Model

**Radyo Metaforu:**
```
Model = Eski radyo
Parametreler = Frekans düğmeleri
Eğitim = Düğmeleri çevirerek yayını netleştirme
Loss = Parazit miktarı
```

Başta sadece gürültü duyarsın. Düğmeleri (parametreleri) çevirdikçe ses (tahmin) netleşir.

### ⚠️ Yaygın Yanılgılar

1. **"Model düşünür"** ❌
   → Model matematiksel bir fonksiyondur, düşünmez. Sadece verdiğin veriye **uyar**.

2. **"Daha büyük model = daha iyi"** ❌
   → Bazen evet, bazen aşırı öğrenme (overfit). Denge lazım.

3. **"Model her şeyi çözer"** ❌
   → Model sadece **verdiğin verinin deseni**ni öğrenir. Veri kötüyse, model kötü.

---

## 2️⃣ Veri Nedir? (Ve Neden Kutsal)

### Veri Anatomisi

```
┌─────────────────────────────────────┐
│  Özellik 1  │  Özellik 2  │  Hedef  │
├─────────────────────────────────────┤
│     120     │      3      │   450K  │  ← Bir örnek (sample)
│     85      │      2      │   280K  │
│     ...     │     ...     │   ...   │
└─────────────────────────────────────┘
     ↓              ↓             ↓
  Alan (m²)     Oda sayısı    Fiyat
```

- **Satır**: Bir örnek / sample / data point
- **Sütun**: Bir özellik / feature / attribute
- **Hedef (Label)**: "Doğru cevap" (supervised learning'de)

### 🎯 İddia

**Öğrenme = Verideki istikrarlı paternleri bulmak**

"İstikrarlı" derken:
- Tesadüf değil, **tekrarlanan** ilişkiler
- Gürültüden ayırt edilebilen **sinyal**

### ⚖️ Kural 1: i.i.d. Varsayımı

**i.i.d.** = independent and identically distributed (bağımsız ve özdeş dağılımlı)

**Anlamı:** Eğitim ve test verileri **aynı dünyadan** gelmeli.

**Örnek - İhlal:**
```
Eğitim: 2020 verileri (pandemi öncesi)
Test:   2022 verileri (pandemi sonrası)
→ Dağılım kayması! Model şaşırır.
```

**Çözüm:**
- Train/Val/Test ayırımını **rastgele** yap
- Zamansal bağımlılık varsa (time series), kronolojik ayır

### 📊 Train/Val/Test Ayrımı

```
[■■■■■■■■□□□]  = Tüm Veri (100%)
 ↓
[■■■■■■■■] [□] [□]
   Train   Val Test
   70-80%  10-15% 10-15%
```

**Roller:**
- **Train:** Modeli eğit (parametreleri öğren)
- **Val:** Hiperparametre ayarla (LR, regularization, vb.)
- **Test:** Final değerlendirme (bir kez, sonuçları raporla)

### ⚠️ Veri Sızıntısı (Data Leakage)

**En Tehlikeli Hata!**

```
❌ YANLIŞ:
1. Tüm veriyi normalize et
2. Sonra train/test ayır
→ Test bilgisi train'e sızdı!

✅ DOĞRU:
1. Train/test ayır
2. Train'den öğrendiğin parametrelerle normalize et
3. Aynı parametrelerle test'i normalize et
```

---

## 3️⃣ Kayıp (Loss) = Pusulan

### Tanım

**Loss Fonksiyonu:** Modelin hatasını **tek bir sayı**ya indirger.

L(y_gerçek, y_tahmin)

**Amaç:** Bu sayıyı **minimize** et.

### 🎯 Neden Tek Sayı?

Çünkü optimizasyon algoritmaları **vektör alanında tek yön** arar. Çok kriterli hedef, algoritmayı şaşırtır.

### 📐 Yaygın Loss Fonksiyonları

#### Regresyon: MSE (Mean Squared Error)

MSE = (1/N) × Σ(y_i - ŷ_i)²

- **Avantaj:** Türevi kolay, büyük hataları cezalandırır
- **Dezavantaj:** Outlier'lara hassas

**Alternatif:** MAE (L1 loss) - daha robust

#### Sınıflandırma: Cross-Entropy

CE = -Σ y_i × log(ŷ_i)

- **Avantaj:** Olasılık dağılımları için ideal
- **Kullanım:** Softmax çıktısıyla beraber

### 💡 Kural 2: "Ne Ölçersen Ona Dönüşürsün"

```
Accuracy optimize et → Sınıf dengesizliğinde yanılır
F1 optimize et → Precision/Recall dengesini tutar
MSE optimize et → Outlier'lara takılır
```

**Seçim Stratejisi:**
1. İş problemini anla
2. Hangi hata tipi daha maliyetli?
3. O metriği seç

### 🧪 Loss Eğrisi Okuma

```
Loss
  │
  │╲
  │ ╲
  │  ╲___
  │      ╲____
  │           ───────  ← Platoya girdi
  └──────────────────── Epoch
     Öğrenme       İyileşme       Doyma
     başlıyor      hızlı         yavaşladı
```

**Sağlıklı Eğitim:**
- Başta hızlı düşüş
- Sonra yavaşlama
- Platoya yaklaşma

**Problem İşaretleri:**
- Hiç düşmüyor → LR çok küçük / model çok basit
- Zıplıyor → LR çok büyük
- Ani düşüş sonra patlama → Sayısal instabilite

---

## 4️⃣ Optimizasyon: Gradient Descent

### 🎯 Amaç

L(θ) kaybını **azaltmak**

### 🧭 Gradient = Pusıla

**Gradient (∇_θ L):** Loss'un parametrelere göre **türevi**

```
         L(θ)
          │
        ╱   ╲
      ╱       ╲
    ╱           ╲
  ╱               ╲
 ────────────────────  θ
         ↑
    Bu noktadaysan,
    gradient sola işaret eder
    (eğim negatif)
```

**Sezgi:** Dağın zirvesinde sisli havada kaybolmuşsun. Elinde sadece **eğim ölçer** var. En dik iniş yönünde küçük adım atarsın.

### 📐 Güncelleme Kuralı

θ_yeni = θ_eski - η × ∇_θ L

Burada:
- **η** (eta): Öğrenme oranı (Learning Rate / LR)
- **∇_θ L**: Gradient (türev)

### 🎚️ Kural 3: LR Piramidi

```
LR çok büyük:
  Loss ↑↓↑↓  (zıplıyor, NaN olabilir)
  
LR uygun:
  Loss ╲╲╲___ (smooth düşüş)
  
LR çok küçük:
  Loss ╲_ (yavaş öğrenme, zaman kaybı)
```

**Pratik Reçete:**
1. Başta biraz cesur (1e-3, 1e-2)
2. Platoya gelince düşür (schedule)
3. Stabilite yoksa önce **LR'ı** düşür

### 🔄 Gradient Descent Türleri

#### Batch GD
- **Her** veriyi gör, sonra güncelle
- **Avantaj:** Stabil
- **Dezavantaj:** Yavaş (büyük veri setlerinde)

#### Stochastic GD (SGD)
- **Her örnekten** sonra güncelle
- **Avantaj:** Hızlı, lokal minimum'lardan kaçabilir
- **Dezavantaj:** Gürültülü

#### Mini-batch GD
- **Küçük gruplar** (32, 64, 128 örnek) ile güncelle
- **Pratik standart:** İkisinin dengesi

---

## 5️⃣ Tensörler: Numpy Ama Steroidli

### 🧊 Tensör Anatomisi

```
Scalar (0D):     5
Vector (1D):     [1, 2, 3]
Matrix (2D):     [[1, 2],
                  [3, 4]]
Tensor (3D+):    [[[...]]]
```

**PyTorch Tensörü = Numpy Array + GPU Desteği + Autograd**

### 🏷️ Önemli Özellikler

#### Shape (Şekil)
```python
x = torch.randn(32, 3, 224, 224)
# (batch_size, channels, height, width)
#       32        3      224     224
```

#### Dtype (Veri Tipi)
- `torch.float32` (default, çoğunlukla yeterli)
- `torch.float16` (hız için, mixed precision)
- `torch.int64` (index'ler için)

#### Device (Cihaz)
- `cpu`: Herkes için
- `cuda`: NVIDIA GPU
- `mps`: Apple Silicon (M1/M2/M3)

### 🎯 Kural 4: Shape Bilinci

**En çok zaman kaybı:** Shape uyumsuzluğu

```python
# ❌ Hata
x = torch.randn(10, 5)  # (10, 5)
y = torch.randn(3, 5)   # (3, 5)
z = x + y  # Error! Shape mismatch

# ✅ Doğru
x = torch.randn(10, 5)  # (10, 5)
y = torch.randn(1, 5)   # (1, 5) - broadcast olacak
z = x + y  # OK! → (10, 5)
```

**Alışkanlık:** Her işlemde `print(x.shape)` yap.

### 📡 Broadcasting

**Küçük tensör** otomatik olarak **büyük tensör**ün şekline uyar.

```
(10, 1) + (1, 5) → (10, 5)
 ↓         ↓
Satırlara  Sütunlara
kopyala    kopyala
```

**Kural:**
1. Sağdan başla
2. Boyutlar eşit ya da biri 1 olmalı
3. Eksik boyutlar 1 kabul edilir

### 🔄 View vs Reshape vs Contiguous

```python
# view: hafızada yer değiştirme
x.view(2, 8)  # Hızlı ama contiguous olmalı

# reshape: gerekirse kopyalar
x.reshape(2, 8)  # Her zaman çalışır

# contiguous: hafızada düzenle
x = x.transpose(0, 1).contiguous()
```

**İpucu:** View hata verirse `contiguous()` ekle.

---

## 6️⃣ Autograd: Zinciri Tersten Gez

### 🎯 Hesap Grafiği (Computational Graph)

```
        x (input)
         ↓
    [Linear Layer]
         ↓
        a (activation)
         ↓
    [Another Layer]
         ↓
      output
         ↓
       Loss
```

**İleri Geçiş (Forward):** Yukarıdan aşağıya hesapla

**Geri Geçiş (Backward):** Aşağıdan yukarıya türevleri hesapla (chain rule)

### 🔗 Zincir Kuralı Sezgisi

**Analoji:** LEGO'lardan kule yaptın.

- **Forward:** Parçaları üst üste koy
- **Backward:** Yıkarken her parçanın üstüne binen yükü hesapla

```
dL/dx = dL/dy × dy/dx
        ↑       ↑
    Üstten   Lokal
    gelen    türev
```

### 📝 Autograd Kullanımı

```python
# 1. Parametre tanımla (gradient takibi açık)
w = torch.randn(5, 3, requires_grad=True)

# 2. Forward pass
y = model(x)
loss = criterion(y, target)

# 3. Backward (gradientleri hesapla)
loss.backward()

# 4. Gradientleri oku
print(w.grad)  # dL/dw

# 5. Güncelle (gradient birikmesin)
with torch.no_grad():
    w -= lr * w.grad
    
# 6. Gradientleri sıfırla (sonraki iterasyon için)
w.grad.zero_()
```

### ⚠️ Yaygın Hatalar

**1. Gradient birikmesi**
```python
# ❌ Her iterasyonda zero_grad() yok
for epoch in epochs:
    loss.backward()  # Gradientler birikir!
    
# ✅ Doğru
for epoch in epochs:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**2. no_grad() unutmak**
```python
# ❌ Güncelleme sırasında graf kirlenebilir
w = w - lr * w.grad

# ✅ Doğru
with torch.no_grad():
    w = w - lr * w.grad
```

---

## 7️⃣ Optimizatörler: Adam vs SGD

### 🔧 Optimizer Seçimi

#### SGD (Stochastic Gradient Descent)
```
θ = θ - lr × gradient
```

**Avantaj:**
- Basit, anlaşılır
- Genelleme iyi olabilir
- Az bellek

**Dezavantaj:**
- Yavaş yakınsama
- LR seçimi kritik

#### Momentum
```
v = β × v + gradient
θ = θ - lr × v
```

**Analoji:** Topu dağdan aşağı yuvarla. Momentum birikir, vadileri aşar.

**Avantaj:**
- Lokal minimum'lardan kaçabilir
- Daha hızlı yakınsama

#### Adam (Adaptive Moment Estimation)
```
m = β1 × m + (1-β1) × gradient        (momentum)
v = β2 × v + (1-β2) × gradient²       (variance)
θ = θ - lr × m / (√v + ε)
```

**Avantaj:**
- Her parametreye **adaptif LR**
- Pratikte çok işe yarar
- Hiperparametre hassasiyeti az

**Dezavantaj:**
- Bazen genelleme SGD'den düşük olabilir
- Daha fazla bellek

### 💡 Kural 5: Pratik Reçete

```
Başlangıç:
  Adam + küçük weight_decay (1e-4)
  LR: 1e-3 veya 1e-4
  
Stabil olunca:
  İsteğe bağlı SGD+Momentum'a geç
  (daha iyi genelleme için)
```

### ⚖️ Weight Decay (L2 Regularization)

**Amaç:** Ağırlıkları **küçük** tut → Overfit'i azalt

```
Loss = Loss_data + λ × Σ(w²)
                      ↑
                 Regularization
                 terimi
```

**PyTorch'ta:**
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4  # L2 reg
)
```

---

## 8️⃣ Bias-Variance Trade-off

### 🎯 İkili Sorun

```
Toplam Hata = Bias² + Variance + Noise
               ↓         ↓         ↓
           Underfitting Overfitting Kaçınılmaz
```

### 📊 Underfit vs Overfit

```
Performance
    │         ╱────╲  ← Sweet Spot
    │       ╱        ╲
    │     ╱  Overfit  ╲
    │   ╱              ╲
    │ ╱ Underfit         ╲
    └────────────────────────  Model Complexity
      Basit              Karmaşık
```

### 🔍 Teşhis

#### Underfit (Az Öğrenme)
```
Train Loss: Yüksek (≥ 1.5)
Val Loss:   Yüksek (≥ 1.5)
```

**Belirtiler:**
- Model yeterince öğrenemiyor
- Train loss bile yüksek

**Çözümler:**
- Daha karmaşık model
- Daha uzun eğitim
- Daha iyi özellik mühendisliği
- LR arttır

#### Overfit (Aşırı Öğrenme)
```
Train Loss: Düşük (< 0.1)
Val Loss:   Yüksek (> 1.0)
```

**Belirtiler:**
- Train mükemmel, val/test kötü
- Model "ezberleme" yapıyor

**Çözümler:**
- Daha fazla veri
- Regularization (L2, dropout)
- Early stopping
- Data augmentation
- Daha basit model

### 📈 Kural 6: Val Eğrisi İzleme

```
Loss
  │
  │╲         Train
  │ ╲╲╲╲___________
  │     ╲
  │       ╲╱╱╱╱ Val
  │           ╱
  │         ╱  ← Bu noktada DUR!
  └──────────────────  Epoch
            ↑
        Overfit
        başlıyor
```

**Early Stopping:**
Val loss N epoch boyunca iyileşmezse **dur**.

---

**(Devamı theory_advanced.md'de...)**

---

## 📚 Week 0 Özet

### Temel Kavramlar Cheat Sheet

| Kavram | Özet | Dikkat |
|--------|------|--------|
| **Model** | f_θ(x) fonksiyonu | Düşünmez, uyar |
| **Loss** | Hata ölçütü | Ne ölçersen ona dönüşürsün |
| **Gradient** | Türev / yön | Pusılan |
| **LR** | Adım büyüklüğü | En kritik hiperparametre |
| **Tensor** | Çok boyutlu array | Shape bilinci! |
| **Autograd** | Otomatik türev | Chain rule otomatik |
| **Optimizer** | Güncelleme kuralı | Adam pratik, SGD teorik |
| **Overfit** | Ezber yapmak | Val loss izle |

### 🎯 Mastery Checklist

Kendinize sorun:

- [ ] Tensör shape'ini gözümde canlandırabiliyorum
- [ ] Forward-backward akışını anlatabiliyorum
- [ ] LR'ın etkisini tahmin edebiliyorum
- [ ] Overfit/underfit'i teşhis edebiliyorum
- [ ] Loss eğrisinden sorun çıkarabilirim
- [ ] Gradient nedir, neden önemlidir biliyorum

**Eğer hepsi ✅ ise:** Week 1'e hazırsın! 🚀

---

## 🧪 Mini Alıştırmalar (10-15 dk)

### Alıştırma 1: LR Deneyi

**Senaryo:** LR = 0.01'de loss düzgün düşüyor.

**Sorular:**
1. LR'ı 0.1 yaparsan ne olur?
2. LR'ı 0.001 yaparsan ne olur?
3. Hangi belirtiden anlarsın?

**Cevaplar (aşağıda)**

### Alıştırma 2: Overfit Tespiti

**Gözlem:**
```
Epoch 50:  Train Loss = 0.05, Val Loss = 0.08
Epoch 100: Train Loss = 0.01, Val Loss = 0.15
```

**Sorular:**
1. Bu overfit mi?
2. Ne yapmalısın?
3. Hangi metriğe bakarak karar verdin?

### Alıştırma 3: Broadcasting

**Kod:**
```python
x = torch.randn(10, 1)  # (10, 1)
y = torch.randn(1, 5)   # (1, 5)
z = x + y               # Shape?
```

**Sonuç shape'i nedir? Neden?**

---

### 📝 Cevaplar

**Alıştırma 1:**
1. LR = 0.1: Loss zıplar, NaN olabilir (çok büyük adım)
2. LR = 0.001: Yavaş öğrenir (çok küçük adım)
3. Loss grafiğinden: smooth ise iyi, zıplıyorsa büyük

**Alıştırma 2:**
1. Evet, overfit! Train düşüyor, val artıyor
2. Early stopping (Epoch 50'de dur), regularization ekle
3. Val loss'un artışından

**Alıştırma 3:**
```python
z.shape  # torch.Size([10, 5])
# (10, 1) → (10, 5) sütunlara kopyala
# (1, 5)  → (10, 5) satırlara kopyala
```

---

## 🚀 Sıradaki Adım

Bu temeller üzerine **Week 1** inşa edilecek:
- Bu teorileri **linear regression**'da pratiğe dökeceğiz
- Manuel gradient descent yazacağız
- PyTorch API'sini kullanacağız
- Train/val split'i uygulayacağız

**Hazır mısın?** `python week1_tensors/linreg_manual.py` 🎯
