# Week 0: Makine Öğrenmesine Giriş - Sıfırdan Başlangıç

**NovaDev v1.0 - Lise Seviyesi Anlatım**

> "Kod yok, formül minimum, sabırlı hoca modunda. Bittiğinde 'ne yapıyoruz, neden, ne zaman neye bakıyoruz' net olsun."

---

## 🎯 Bu Döküman Kimler İçin?

- ✅ Hiç ML deneyimi yok
- ✅ "Makine öğrenmesi" kelimesi soyut geliyor
- ✅ Önce **sezgiyle** anlamak istiyorum
- ✅ Matematiği sonra öğrenirim, önce mantığı kavramak istiyorum

**Seviye:** Tam Başlangıç (Lise)
**Süre:** 45-60 dakika
**Format:** Günlük dil, bol benzetme, mini quiz
**Hedef:** "Ah, ha! Demek bu kadar basit!" demek

---

## 0️⃣ Makine Öğrenmesi Nedir? (Tek Cümle)

### 🎯 Tanım

**Makine öğrenmesi = Örneklerden (veri) bir kural (fonksiyon) öğrenip, yeni durumları tahmin etme işi.**

### 🌟 Günlük Hayattan Örnekler

**1. Telefonun Klavyesi**
```
Sen: "Merhaba, nas..."
Klavye: "→ nasılsın?" (otomatik tamamlama)

Nasıl yapıyor?
→ Milyonlarca mesajdan "merhaba"dan sonra
  "nasılsın" gelme PATERNINI öğrenmiş
```

**2. Netflix Önerileri**
```
İzlediklerin: Aksiyon filmleri, bilim kurgu
Netflix: "Şunu da beğenebilirsin: Inception"

Nasıl yapıyor?
→ Senin gibi izleyenlerin sonra ne izlediğine bakıyor
  (benzer profil → benzer tercih)
```

**3. Spam Filtresi**
```
Email: "Tıkla kazaaaan!!! 1000000$$$"
Gmail: 🚫 SPAM

Nasıl yapıyor?
→ Milyonlarca spam örneğinden ORTAK özellikleri öğrenmiş:
  - Çok ünlem (!!!)
  - Para sembolü ($$$)
  - Aşırı büyük rakamlar
```

### 🧩 Temel Bileşenler

```
┌────────────────────────────────────────┐
│  Girdi (x)                             │
│  → Modele verdiğin bilgi               │
│    Örnek: Ev özellikleri (m², oda)     │
└────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│  Model (f_θ)                           │
│  → Kutu içindeki fonksiyon             │
│  → İçinde "ayar düğmeleri" var (θ)     │
└────────────────────────────────────────┘
              ↓
┌────────────────────────────────────────┐
│  Çıktı (ŷ)                             │
│  → Modelin tahmini                     │
│    Örnek: Ev fiyatı (500K TL)          │
└────────────────────────────────────────┘
```

**Parametre (θ):** Modelin iç ayarları (öğrenilen sayılar)
**Amaç:** Bu ayarları öyle ayarla ki tahminler **doğru** ve **tutarlı** olsun

### 💡 Altın Prensip

> **"Benzer şeyler benzer sonuçlar verir"**
> 
> Bu prensibi öğrenilebilir bir kurala dönüştürüyoruz!

---

## 1️⃣ Veri: Neden Kutsal?

### 📊 Veri Nedir?

**Tablo olarak düşün:**

```
┌─────────┬──────┬─────┬──────┬────────────┐
│  Ev No  │  m²  │ Oda │ Yaş  │  Fiyat(TL) │
├─────────┼──────┼─────┼──────┼────────────┤
│    1    │ 100  │  3  │  5   │   400K     │ ← Örnek 1
│    2    │  75  │  2  │ 10   │   300K     │ ← Örnek 2
│    3    │ 150  │  4  │  2   │   700K     │ ← Örnek 3
│   ...   │ ...  │ ... │ ...  │   ...      │
└─────────┴──────┴─────┴──────┴────────────┘
          ↑      ↑     ↑        ↑
       Özellik Özellik Özellik  Etiket
       (Feature)                (Label)
```

**Terminoloji:**
- **Satır = Örnek** (bir ev, bir öğrenci, bir fotoğraf)
- **Sütun = Özellik** (metrekare, yaş, piksel değeri)
- **Hedef = Etiket** (fiyat, sınav notu, "kedi"/"köpek")

### 🎓 Öğrenme Türleri

#### 1. Gözetimli (Supervised)
```
Veri: (Girdi, Doğru Cevap) çiftleri

Örnek: (Ev özellikleri, Fiyat)
       (Fotoğraf, "Kedi"/"Köpek")
       
Görev: Yeni girdide doğru cevabı tahmin et
```

#### 2. Gözetimsiz (Unsupervised)
```
Veri: Sadece girdiler (cevap yok)

Örnek: Müşteri alışverişleri
       
Görev: Benzer grupları bul (kümeleme)
       Anormallikleri tespit et
```

#### 3. Pekiştirmeli (Reinforcement)
```
Durum: Oyun tahtası / robot sensörleri
Aksiyon: Hamle yap
Ödül: Kazandın mı? Kaybettin mi?

Görev: En çok ödülü topla (dene yanıl)
```

**Bu programda:** Çoğunlukla **gözetimli öğrenme** (supervised)

### 🔄 Neden Train/Val/Test Böl?

#### 📖 Öğrenci Benzetmesi

```
Bir öğrenciyi düşün:

❌ YANLIŞ YOL:
  Çalışma: 10 soru → Ezberle
  Sınav: Aynı 10 soru
  Sonuç: "100 aldım!" (ama gerçekte hiçbir şey bilmiyor)

✅ DOĞRU YOL:
  Çalışma: 100 soru (TRAIN)
  Ara sınav: 20 yeni soru (VALIDATION) → Çalışma stratejisini ayarla
  Final: 30 tamamen yeni soru (TEST) → Gerçek performans
```

#### 📦 Veri Bölme

```
Tüm Veri (1000 örnek)
         ↓
    ┌────────────┐
    │            │
  ┌─┴──────────┐ │
  │   TRAIN    │ │  70% (700 örnek)
  │ "Çalışma"  │ │  → Model burada öğrenir
  └────────────┘ │
    ┌──────────┐ │
    │   VAL    │ │  15% (150 örnek)
    │"Ara sınav"│ │  → Ayarları seçeriz (LR, L2, vs.)
    └──────────┘ │
      ┌────────┐ │
      │  TEST  │─┘  15% (150 örnek)
      │"Final" │    → Tek seferlik gerçek performans
      └────────┘
```

### ⚠️ Altın Kural

```
❌ Test'e bakıp ayar yapma!

Neden?
→ Test'i "ezberlersin"
→ Gerçek dünya performansı bilinmez kalır

✅ Val'e bakıp ayarla, Test'e sadece BİR KEZ bak
```

### 🌊 Dağılım Kayması (Distribution Shift)

#### İdeal Dünya
```
Train verisi: İstanbul'da çekilmiş fotoğraflar
Test verisi:  İstanbul'da çekilmiş fotoğraflar
               ↑
         Aynı koşullar → Model iyi çalışır ✓
```

#### Gerçek Dünya
```
Train: İstanbul (güneşli, gündüz)
Test:  Ankara (karlı, akşam)
        ↑
    Farklı koşullar → Model şaşırır ✗
```

**Örnekler:**
- **Kovaryat kayması:** Kamera modeli değişti
- **Konsept kayması:** "Spam" tanımı zamanla evrildi
- **Sınıf oranı:** Training'de %50 spam, gerçekte %5 spam

### 🕐 Zaman Serisi Özel Durum

```
❌ YANLIŞ:
  Train: Rastgele günler
  Test:  Rastgele günler
  
  → Geleceği bilip geçmişi tahmin edebilir (HILE!)

✅ DOĞRU:
  Train: [───────Geçmiş───────]
  Val:               [─Yakın gelecek─]
  Test:                          [─Gelecek─]
  
  → Zaman sırasını MUTLAKA koru
```

---

## 2️⃣ Model: Kutudaki Fonksiyon

### 🎛️ Model = Ayarlı Radyo

```
┌─────────────────────────────┐
│    📻 RADYO                 │
│                              │
│  Frekans düğmesi: [⚫────]  │ ← Parametre 1
│  Ses düğmesi:     [───⚫─]  │ ← Parametre 2
│  Bas düğmesi:     [──⚫──]  │ ← Parametre 3
│                              │
│  Çıkan ses: 📢 "...kırrr..." │
└─────────────────────────────┘

Hedef: Düğmeleri öyle ayarla ki yayın NETLEŞSİN

Model = Radyo
Parametreler = Düğmeler (θ)
Eğitim = Düğmeleri ayarlama süreci
```

### 🧮 Matematiksel (Basit)

```
ŷ = f_θ(x)

ŷ: Tahmin (radyodan çıkan ses)
x: Girdi (radyo dalgası)
θ: Parametreler (düğmelerin konumu)
f: Fonksiyon (radyo devresi)
```

### 🎚️ Parametre vs Hiperparametre

```
┌───────────────────────────────────────┐
│  PARAMETRE (θ)                        │
├───────────────────────────────────────┤
│  → Model ÖĞRENİR                      │
│  → Veriyle ayarlanır                  │
│  → Örnek: Ağırlıklar, bias'lar       │
│  → Sayısı: Binler, milyonlar         │
└───────────────────────────────────────┘

┌───────────────────────────────────────┐
│  HİPERPARAMETRE                       │
├───────────────────────────────────────┤
│  → SEN seçersin                       │
│  → Eğitim başlamadan önce kararlaştır│
│  → Örnek: Öğrenme hızı, katman sayısı│
│  → Sayısı: 5-20 arası                │
└───────────────────────────────────────┘
```

**Benzetme:**
- **Parametre:** Öğrencinin ezberledikleri
- **Hiperparametre:** Çalışma stratejisi (kaç saat, hangi kitap)

### 📈 Model Kapasitesi

```
Kapasite = Modelin esnekliği

Çok Düşük:              Çok Yüksek:
  Basit çizgi             Karmaşık eğri
  
     ●                       ●
   ●   ●                   ●   ●
  ●─────●●               ● ╱╲ ╱╲●
 ●       ●              ●╱  ╲╱  ╲●
         ●                       ●
         
  UNDERFIT               OVERFIT
  (Yetersiz)             (Ezber)
```

**İpucu:** Küçükten başla → Gerekirse büyüt

---

## 3️⃣ Hata Nasıl Ölçülür? (Loss = Kayıp)

### 🎯 Neden Tek Sayı?

```
Model 1: Bazı tahminler iyi, bazı kötü
Model 2: Bazı tahminler iyi, bazı kötü

Hangisi daha iyi? 🤔

→ Tek bir sayıya çevir → Karşılaştır! ✓
```

**Loss (Kayıp) = Hedefe uzaklık (tek sayı)**

### 📏 Regresyon Loss'ları (Sayı Tahmini)

#### MSE (Mean Squared Error)
```
Gerçek: 100
Tahmin: 90
Hata: 10

MSE = (10)² = 100  ← Kare alıyor!

Gerçek: 100
Tahmin: 50
Hata: 50

MSE = (50)² = 2500  ← Büyük hata → ÇOK AĞIR CEZA
```

**Özellik:** Büyük hataları **sert** cezalandırır

**Ne Zaman Kullan:**
- Standard regresyon
- Aykırı değer az
- "Büyük hatalar çok kötü" diyorsan

#### MAE (Mean Absolute Error)
```
Gerçek: 100
Tahmin: 90
Hata: 10

MAE = |10| = 10

Gerçek: 100
Tahmin: 50
Hata: 50

MAE = |50| = 50  ← Linear ceza (adil)
```

**Özellik:** Aykırı değerlere **toleranslı**

**Ne Zaman Kullan:**
- Aykırı değer çok
- "Her hata eşit önemde"

### 🎲 Classification Loss (Sınıflama)

#### Cross-Entropy (Log Loss)
```
Gerçek: "Kedi" (1)
Model güveni: %10 Kedi
→ AĞIR CEZA! (Yanlış ve çok emin!)

Gerçek: "Kedi" (1)
Model güveni: %90 Kedi
→ Hafif ceza (Doğru ve emin)

Gerçek: "Kedi" (1)
Model güveni: %99 Kedi
→ Çok hafif ceza (Mükemmel!)
```

**Analoji - Sınav:**
```
Soru: "2+2 = ?"
Senin cevabın: "5" + "Kesinlikle eminim!"
Hoca: 😡 Ağır ceza! (Yanlış + kendinden emin)

Senin cevabın: "5" + "Emin değilim..."
Hoca: 😐 Orta ceza (Yanlış ama şüphelisin)
```

### ⚖️ Metric vs Loss

```
┌─────────────────────────────────────┐
│  LOSS                               │
├─────────────────────────────────────┤
│  → Eğitim sırasında OPTİMİZE edilir │
│  → Matematiksel olarak türevlenebilir│
│  → Örnek: MSE, Cross-Entropy        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  METRIC                             │
├─────────────────────────────────────┤
│  → İnsan için anlamlı               │
│  → Raporlamada kullanılır           │
│  → Örnek: Accuracy, F1, RMSE        │
└─────────────────────────────────────┘
```

---

## 4️⃣ Kayıp Nasıl Azaltılır? (Gradient Descent)

### 🗻 Dağ Analojisi

```
    ╱╲    ╱╲
   ╱  ╲  ╱  ╲
  ╱    ╲╱    ╲
 ╱      *     ╲  ← Sen buradasın (sisli dağ)
╱──────────────╲

Hedef: Aşağı in (minimum)
Elinde: Eğim ölçer (gradient)
         "Bu yönde eğim aşağı iniyor"
```

### 📐 Gradient Descent (Gradyan İnişi)

```
1. Bulunduğun noktada EĞIMI ölç
   ↓
2. Eğimin gösterdiği yönde ADIM at
   ↓
3. Yeni noktada tekrar EĞIMI ölç
   ↓
4. Tekrarla (kayıp artık değişmiyorsa → DUR)
```

**Formül (Basit):**
```
Yeni_Parametre = Eski_Parametre - (Öğrenme_Hızı × Eğim)
                                      ↑              ↑
                                   Ne kadar          Hangi
                                   atlayacak?        yöne?
```

### 🎚️ Learning Rate (Öğrenme Hızı)

#### Çok Büyük LR (0.9)
```
    ╱╲
   ╱  ╲
  ●────┼────●  ← ZIK ZAK! Minimum'u aşıyor
      ╲╱
       ↑
   Hedef (ama geçiyor)
```

**Sonuç:** Patlar, diverge olur, NaN

#### Uygun LR (0.1)
```
    ╱╲
   ╱  ╲
  ●──→●─→●─→●  ← Smooth iniş
         ╲╱
          ↑
        Minimum'a ulaştı ✓
```

**Sonuç:** Başarılı!

#### Çok Küçük LR (0.001)
```
    ╱╲
   ╱  ╲
  ●→●→●→●→●→...  ← Çok yavaş
         ╲╱
```

**Sonuç:** Saatler sürer, sıkıcı

### 🎒 Mini-Batch Gradient Descent

```
Batch GD:
  → Tüm veriyi gör → Hesapla → Güncelle
  → Stabil ama YAVAŞ (1 milyon örnek!)

Stochastic GD:
  → Her örneği gör → Güncelle
  → Hızlı ama GÜRÜLTÜLÜ (zikzak)

Mini-batch GD:  ⭐ PRAKTİK!
  → 32-256 örnek → Hesapla → Güncelle
  → Hız + Stabilite dengesi
  → GPU parallelizasyonu
```

**Gürültü = Bazen İyi:**
```
Gürültülü adımlar:
  → Dar çukurlardan kaçar
  → Daha geniş çukura oturur
  → Daha iyi genelleme!
```

### 🚀 Gelişmiş Optimizatörler

#### Momentum
```
Top vadiden yuvarlanıyor 🏀

  ╱╲   ╱╲
 ╱  ╲ ╱  ╲
╱    ●    ╲  ← Küçük tepeyi aşar (momentum)
     ↓
    ╱╲
   ╱  ●  ← Daha derin çukura ulaştı!
   ╲──╱
```

**Mantık:** Geçmiş adımları da hesaba kat

#### Adam/AdamW ⭐
```
Her parametre için AYRI hız ayarı:

Parametre 1: Hızlı değişsin → Yüksek LR
Parametre 2: Yavaş değişsin → Düşük LR

→ Adaptif (akıllı)
→ Pratikte en çok kullanılır
```

**Pratik Reçete:**
```
Başlangıç: AdamW + küçük L2
          ↑
    %90 durumda işe yarar!
```

---

## 5️⃣ Tensors: Yapı Taşı

### 📦 Tensor = Gelişmiş Sayı Tablosu

```
Skalar (0D):     5
Vektör (1D):    [1, 2, 3]
Matrix (2D):    [[1, 2],
                 [3, 4]]
Tensor (3D+):   [[[1,2],[3,4]],
                 [[5,6],[7,8]]]
```

### 🏷️ Tensor Özellikleri

```python
x = torch.randn(64, 3, 224, 224)
                 ↑   ↑   ↑    ↑
             Batch  RGB  Height Width

Shape:  (64, 3, 224, 224)
Dtype:  torch.float32
Device: mps (Apple GPU)
```

**Shape = En Önemli:**
```
❌ En sık hata: Shape uyuşmazlığı!

"Expected (10, 5) but got (10, 3)"
         ↑              ↑
      Beklenen      Gelen
      
Çözüm: Her adımda print(x.shape)
```

### 📡 Broadcasting (Otomatik Genişleme)

```
(10, 1) + (1, 5) → (10, 5)

Görsel:
  [10]     [1 2 3 4 5]
  [10]  +  [1 2 3 4 5]
  [10]     [1 2 3 4 5]
  ...

  =
  
  [11 12 13 14 15]
  [11 12 13 14 15]
  [11 12 13 14 15]
  ...
```

**Kural (Basit):**
- Boyutlar eşit VEYA biri 1 olmalı
- Otomatik kopyalar

---

## 6️⃣ Autograd: Otomatik Türev

### 📝 Fatura Analojisi

```
Restoran:
  Ana yemek: 50 TL
  İçecek:    20 TL
  Tatlı:     30 TL
  ──────────────
  KDV (%18): ???  ← Her kalem için hesapla
  ──────────────
  Toplam:    ???
```

**Autograd = Otomatik muhasebe:**
```
Forward (İleri):
  x → işlem1 → işlem2 → işlem3 → Loss
  
  (Fişleri keserken her işlem KAYDEDİLİYOR)

Backward (Geri):
  Loss ← "Senin payın ne?" ← işlem3
       ← "Senin payın ne?" ← işlem2
       ← "Senin payın ne?" ← işlem1
  
  (Fişleri tersten topla, herkesin borcunu hesapla)
```

### 🔧 Temel Komutlar

```python
# 1. "Bu parametrenin türevini tut"
x = torch.tensor([1.0], requires_grad=True)

# 2. İleri hesap (fiş kes)
y = x ** 2

# 3. Geri hesap (borç öde)
y.backward()

# 4. Gradient'i oku
print(x.grad)  # dy/dx = 2x = 2

# 5. Sıfırla (yeni tur)
x.grad.zero_()
```

### ⚠️ Önemli: Gradient Birikimi

```python
# ❌ YANLIŞ
for epoch in range(10):
    loss = compute_loss()
    loss.backward()  # Gradient BİRİKİYOR!
    optimizer.step()

# ✅ DOĞRU
for epoch in range(10):
    optimizer.zero_grad()  # Önce temizle!
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

**Neden?**
```
backward() → ADD yapar (ekler)
           → SET yapmaz (ayarlamaz)
           
Temizlemezsen → Sürekli büyür → Patlama!
```

---

## 7️⃣ Overfit/Underfit: Altın Denge

### 📊 Üç Senaryo

#### 1. Underfit (Yetersiz Öğrenme)
```
Train Loss: Yüksek 📈
Val Loss:   Yüksek 📈

Neden?
  → Model çok basit
  → Yeterince öğrenmemiş
  
Çözüm:
  ✓ Daha karmaşık model
  ✓ Daha uzun eğitim
  ✓ Daha iyi özellikler
```

#### 2. Overfit (Ezber)
```
Train Loss: Çok düşük 📉
Val Loss:   Yüksek 📈

Neden?
  → Model ezberledi
  → Yeni veriyi genelleyemiyor
  
Çözüm:
  ✓ Regularization (L2, dropout)
  ✓ Early stopping
  ✓ Daha fazla veri
  ✓ Data augmentation
```

#### 3. Good Fit (İDEAL) ⭐
```
Train Loss: Düşük 📉
Val Loss:   Düşük 📉

Durum: Mükemmel denge! ✓
```

### 📈 Loss Eğrileri

```
Loss
  │
  │ Train ──────╲___________
  │              ╲
  │               ╲  
  │ Val   ─────────╲____╱───  ← Overfit başladı!
  │                 ↑
  │            Bu noktada DUR
  └─────────────────────────→ Epoch
                    ↑
              Early stopping
```

### 🛡️ Regularization Cephanesi

#### 1. L2 (Weight Decay)
```
"Ağırlıkları KÜÇÜK tut"

Büyük ağırlık = Aşırı hassas model = Overfit
Küçük ağırlık = Sade model = Genelleme ✓
```

#### 2. L1 (Lasso)
```
"Bazı ağırlıkları TAM SIFIR yap"

→ Özellik seçimi (gereksizleri atar)
```

#### 3. Dropout (Derin Ağlarda)
```
Eğitim sırasında rastgele nöronları "öldür"

→ Model tek nörona bağımlı olamaz
→ Daha robust (dayanıklı)
```

#### 4. Early Stopping ⭐
```
Val loss kötüleşmeye başladı mı? → DUR!

→ En basit regularization
→ Pratikte çok etkili
```

---

## 8️⃣ Ölçekleme: Neden Önemli?

### 🎢 Çarpık Yüzey Problemi

```
Özellik 1: 0-1 arası
Özellik 2: 0-10,000 arası

Loss Yüzeyi:

        Öz2
         ↑
    ╱╲  │  ╱╲     ← Çok dik (Öz2 hassas)
   ╱  ╲ │ ╱  ╲
  ╱    ╲│╱    ╲
 ────────────────→ Öz1
        ← Çok yassı (Öz1 etkisiz)

Gradient Descent bu yüzeyde ZİKZAK yapar!
```

### ✨ Ölçekleme Sonrası

```
Özellik 1: Ortalama=0, Std=1
Özellik 2: Ortalama=0, Std=1

Loss Yüzeyi:

        Öz2
         ↑
       ╱─╲        ← Dairevi (dengeli)
      │   │
      ╲───╱
 ─────────────→ Öz1

Gradient Descent DÜMDÜZ iner! ✓
```

### 📏 Standardization (Z-Score)

```
x' = (x - mean) / std

Örnek:
  Yaşlar: [20, 30, 40, 50, 60]
  Mean: 40
  Std: 14.14
  
  Ölçeklenmiş: [-1.41, -0.71, 0, 0.71, 1.41]
               ↑
           Ortalama=0, Std=1
```

**Ne Zaman:** Çoğu durumda ilk tercih!

---

## 9️⃣ Değerlendirme Metrikleri

### 📐 Regresyon Metrikleri

#### RMSE (Root Mean Squared Error)
```
Tahmin: [100, 200, 300]
Gerçek: [110, 190, 310]
Hata:   [10,  10,  10]

MSE = (10² + 10² + 10²) / 3 = 33.33
RMSE = √33.33 = 5.77

Birim: Aynı (TL, m², vs.) → Yorumlanabilir!
```

#### MAE (Mean Absolute Error)
```
Tahmin: [100, 200, 300]
Gerçek: [110, 190, 310]
Hata:   [10,  10,  10]

MAE = (|10| + |10| + |10|) / 3 = 10

Daha robust (aykırı değerlere toleranslı)
```

### 🎯 Classification Metrikleri

#### Confusion Matrix
```
                 Tahmin
              Pos      Neg
Gerçek Pos    90   |   10    ← FN (False Negative) "Kaçırdık"
       Neg    20   |  880    ← TN (True Negative) "Doğru red"
              ↑
            FP (False Positive) "Yanlış alarm"
```

#### Precision (İsabet)
```
Precision = TP / (TP + FP)
          = 90 / (90 + 20)
          = 0.82

"Pozitif dediğimizin %82'si gerçekten pozitif"
```

#### Recall (Yakalama)
```
Recall = TP / (TP + FN)
       = 90 / (90 + 10)
       = 0.90

"Gerçek pozitiflerin %90'ını yakaladık"
```

#### F1 Score (Denge)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.82 × 0.90) / (0.82 + 0.90)
   = 0.86

İkisinin harmonik ortalaması (dengeli)
```

### ⚠️ Accuracy Tuzağı

```
Veri: %99 negatif, %1 pozitif

Aptal Model: "Hep negatif de"
Accuracy: %99  ← YANILTI!
Recall: 0      ← Hiç pozitif yakalayamıyor!

Çözüm: Dengesiz veride Precision/Recall/F1 kullan
```

---

## 🔟 Data Leakage (Sızıntı)

### 🚨 En Sinsi Hata

**Tanım:** Eğitimde olmaması gereken bilgi **sızdı**

### 📅 Temporal Leakage (Zaman Sızıntısı)

```
❌ YANLIŞ:
  2020 verileri → Train
  2019 verileri → Test
  
  → GELECEĞİ bilip GEÇMİŞİ tahmin ediyorsun! (HILE)

✅ DOĞRU:
  2019 verileri → Train
  2020 verileri → Test
  
  → Geçmişten geleceği tahmin (gerçekçi)
```

### 🔗 Target Leakage (Hedef Sızıntısı)

```
Hedef: Kredi geri ödenecek mi? (Evet/Hayır)

Özellikler:
  - Yaş ✓
  - Gelir ✓
  - Kredi skoru ✓
  - Geri_ödeme_planı ✗ ← SIKINTI!
                        (Bu hedefe çok yakın!)

Neden?
→ Geri ödeme planı ancak ödeme BAŞLADIĞINDA bilinir
→ Tahmin anında bu bilgi YOK!
```

### 🧮 Normalization Leakage

```python
# ❌ YANLIŞ: Tüm veriyi kullanarak normalize et
mean = all_data.mean()  # Train + Test!
std = all_data.std()
normalized = (all_data - mean) / std

# ✅ DOĞRU: Sadece train'den öğren
mean = train_data.mean()  # Sadece train
std = train_data.std()
train_norm = (train_data - mean) / std
test_norm = (test_data - mean) / std  # Aynı mean/std kullan
```

### 🔍 Leak Tespiti

```
Sorular:
1. Bu özellik tahmin ANI'nda bilinir mi?
2. Bu özellik hedefle "fazla" ilişkili mi? (correlation > 0.95)
3. Zaman sırasını bozdum mu?
4. Test verisine hiç dokundum mu?

Eğer herhangi biri "Hayır" → Leakage olabilir!
```

---

## 1️⃣1️⃣ Deney Disiplini

### 🔬 Bilimsel Yöntem

```
1. GÖZLEM
   "Val loss platoya ulaştı"
   
2. HİPOTEZ
   "LR çok büyük olabilir"
   
3. DENEY
   "LR'ı yarıya indireyim"
   
4. ÖLÇÜM
   "Val loss değişti mi?"
   
5. SONUÇ
   "Evet, düzeldi" → Hipotez doğru ✓
   "Hayır" → Başka hipotez dene
   
6. ÖĞRENİM
   "LR hassasiyeti yüksekmiş"
```

### 📊 Baseline Stratejisi

```
Level 0: En Basit (Dummy)
  Regresyon: Ortalama tahmin et
  Sınıflama: En sık sınıfı seç
  
  Sonuç: Accuracy %65
  ↓

Level 1: Basit Model
  Linear Regression / Logistic Regression
  
  Sonuç: Accuracy %72 (+7%)
  ↓

Level 2: Standard
  Random Forest / XGBoost
  
  Sonuç: Accuracy %85 (+13%)
  ↓

Level 3: Karmaşık (Gerekiyorsa)
  Neural Network / Custom
  
  Sonuç: Accuracy %87 (+2%, pahalı!)
```

**Kural:** Her seviyeyi **beat et**, sonra geç!

### 📝 Experiment Log Şablonu

```markdown
## Deney #005 - 2025-10-06

### Hipotez
"L2=0.001 ekleyince overfit azalacak"

### Setup
- Model: LinearRegression
- LR: 0.01
- L2: 0.001  ← DEĞİŞTİRİLEN
- Batch: 32
- Seed: 42

### Baseline (Deney #004)
Train: 0.05, Val: 0.15 (overfit!)

### Sonuç
Train: 0.08 (+0.03, beklenen)
Val: 0.10 (-0.05, ✓ düzeldi!)

### Karar
✓ L2 etkili
→ Sırada: Early stopping ekle
```

---

## 1️⃣2️⃣ En Sık 10 Hata & Çözüm

### 🐛 Hata Kataloğu

#### 1. Learning Rate Felaketi
```
Belirti: Loss → NaN veya sonsuz
Sebep: LR çok büyük
Çözüm: LR'ı 10x düşür (0.01 → 0.001)
```

#### 2. Shape Uyumsuzluğu
```
Belirti: "RuntimeError: size mismatch (10,5) vs (10,3)"
Sebep: Katmanlar uyuşmuyor
Çözüm: Her katmanda print(x.shape)
```

#### 3. Device Mismatch
```
Belirti: "Expected MPS tensor but got CPU"
Sebep: Bazı tensor'ler farklı cihazda
Çözüm: Hepsini aynı device'a taşı (.to(device))
```

#### 4. Gradient Unutma
```
Belirti: Training unstable, loss zıplıyor
Sebep: optimizer.zero_grad() unutulmuş
Çözüm: Her iterasyon başında zero_grad()
```

#### 5. Val = Test Karıştırma
```
Belirti: Test'te beklenmedik düşük performans
Sebep: Val'e bakıp ayar yaptın (ezberledin)
Çözüm: Test'e sadece BİR KEZ bak
```

#### 6. Ölçekleme Unutma
```
Belirti: Yakınsama çok yavaş, LR hassas
Sebep: Özellikler normalize edilmemiş
Çözüm: StandardScaler kullan
```

#### 7. Data Leakage
```
Belirti: Train mükemmel, test felaket
Sebep: Test bilgisi train'e sızmış
Çözüm: Temporal split, normalization dikkat
```

#### 8. Dengesiz Sınıfta Accuracy
```
Belirti: %99 accuracy ama işe yaramaz
Sebep: Sınıf oranı %1/%99
Çözüm: Precision/Recall/F1 kullan
```

#### 9. Seed Yok
```
Belirti: Her çalıştırmada farklı sonuç
Sebep: Seed sabitlenmemiş
Çözüm: set_seed(42) en başta
```

#### 10. Overfit İşareti Kaçırma
```
Belirti: Train ↓↓, Val ↑↑
Sebep: Erken durdurma yok
Çözüm: Early stopping ekle
```

### 🚑 Acil Durum Kiti (İlk Yardım)

```
Problem: Kod çalışmıyor!

Checklist:
1. ☐ LR çok büyük mü? → Yarıya indir
2. ☐ Özellikler ölçekli mi? → Standardize et
3. ☐ zero_grad() var mı? → Ekle
4. ☐ Shape'ler uyumlu mu? → Yazdır
5. ☐ Device aynı mı? → Kontrol et
6. ☐ Loss doğru mu? → Regresyon ≠ CE
```

---

## 1️⃣3️⃣ Mini Quiz (Kendini Test Et!)

### 🎯 Soru 1: Overfit Tespiti
```
Train Loss: 0.01
Val Loss:   0.50

Durum: ?
Çözüm: ?

Cevap: Overfit! (Ezber)
Çözüm: L2 ekle, early stopping, daha fazla veri
```

### 🎯 Soru 2: Accuracy Tuzağı
```
Veri: %95 negatif, %5 pozitif
Model: "Hep negatif"
Accuracy: %95

Sorun: ?

Cevap: Model hiç pozitif yakalayamıyor! (Recall=0)
Çözüm: F1, Precision/Recall kullan
```

### 🎯 Soru 3: Loss Patlaması
```
Epoch 1: Loss = 2.5
Epoch 2: Loss = 5.8
Epoch 3: Loss = NaN

Sebep: ?

Cevap: LR çok büyük
Çözüm: LR'ı 10x düşür
```

### 🎯 Soru 4: Yavaş Yakınsama
```
1000 epoch sonra hala düzelmiyor
LR hassas (biraz artırınca patlar)

Sebep: ?

Cevap: Ölçekleme yapılmamış, condition number yüksek
Çözüm: StandardScaler kullan
```

---

## 1️⃣4️⃣ Sözlük: Cep Kartı

```
┌──────────────────────────────────────────────┐
│  TERİM              AÇIKLAMA                 │
├──────────────────────────────────────────────┤
│  Model              Girdi→Çıktı fonksiyonu   │
│  Parametre (θ)      Model'in öğrenen sayılar │
│  Hiperparametre     Senin seçtiklerin (LR)   │
│  Loss               Hata miktarı (tek sayı)  │
│  Gradient           Eğim (hangi yöne?)       │
│  Learning Rate      Adım büyüklüğü           │
│  Optimizer          Gradient descent algoritm│
│  Epoch              Tüm veriyi bir kez görme │
│  Batch              Mini grup (32-256)       │
│  Overfit            Ezber (train iyi, val kötü)
│  Underfit           Yetersiz öğrenme         │
│  Regularization     Ezberi frenle (L2, dropout)
│  Early Stopping     Val bozulunca dur        │
│  Validation         Ara sınav (ayar seç)     │
│  Test               Final (tek seferlik)     │
│  Precision          İsabet oranı             │
│  Recall             Yakalama oranı           │
│  F1                 Precision+Recall dengesi │
│  Data Leakage       Test bilgisi sızdı       │
│  Seed               Rastgelelik sabitleyici  │
└──────────────────────────────────────────────┘
```

---

## 1️⃣5️⃣ Neden Lineer Regresyon ile Başlıyoruz?

### 🎯 Pedagojik Avantajlar

#### 1. Basit ve Temiz
```
y = w₀ + w₁×x₁ + w₂×x₂

→ Tek global minimum (convex)
→ Optimizasyon davranışı NET görünür
→ Debug kolay
```

#### 2. Tüm Kavramlar Var
```
✓ Loss (MSE)
✓ Gradient
✓ Optimization (GD)
✓ Regularization (L2)
✓ Overfitting
✓ Val/Test split
✓ Metrics (R², RMSE)

→ Kamp eğitimi gibi! Her beceri burada öğrenilir
```

#### 3. Görsel Anlama Kolay
```
2D grafikte:
  
  Fiyat
    │  ●
    │    ●  ●
    │  ●   ──── Fit line
    │ ●  ●
    └───────────→ m²
    
"Çizgiyi noktalara uydur" → Herkes anlıyor!
```

#### 4. Kavramlar Transfer Eder
```
Linear Regression'da öğrendiğin:

Ölçekleme    → MLP'de de şart
LR seçimi    → CNN'de de kritik
Overfit      → Transformer'da da sorun
Early stop   → Her yerde kullanılır

→ Temel burada SAĞLAM atılır!
```

---

## 1️⃣6️⃣ Week 0 Bugün: Ne Yapmalı?

### 📝 Pratik Alıştırma (30 dk)

**3 Problem Yaz, Analiz Et:**

#### Problem 1: Regresyon
```
Görev: Ev fiyatı tahmini
Özellikler: m², oda sayısı, yaş
Hedef: Fiyat (TL)

Analiz:
  - Loss: MSE (sayısal tahmin)
  - Metric: RMSE (yorumlanabilir)
  - Risk: Aykırı değer (lüks villa)
  - Split: Rastgele 70/15/15
```

#### Problem 2: Dengesiz Sınıflama
```
Görev: Kredi kartı dolandırıcılığı
Özellikler: İşlem tutarı, zaman, lokasyon
Hedef: Dolandırıcılık mı? (Evet/Hayır)
Denge: %99 normal, %1 dolandırıcılık

Analiz:
  - Loss: Cross-Entropy (+ class weight)
  - Metric: F1, Recall (yakalama önemli!)
  - Risk: Accuracy yanıltır (%99 aptal model)
  - Split: Stratified (oranı koru)
```

#### Problem 3: Zaman Serisi
```
Görev: Satış tahmini
Özellikler: Geçmiş satış, mevsim, promosyon
Hedef: Yarınki satış

Analiz:
  - Loss: MAE (outlier'a robust)
  - Metric: MAPE (yüzde hata)
  - Risk: Temporal leakage (gelecek sızmasın!)
  - Split: Zamansal (geçmiş→gelecek)
```

### ✅ Self-Check (Kendini Kontrol Et)

```
Week 0 bittiğinde cevap verebiliyor musun?

□ "Model nedir?" → Ayarlı fonksiyon
□ "Loss nedir?" → Hata miktarı (tek sayı)
□ "Gradient nedir?" → Eğim (hangi yöne)
□ "Overfit nedir?" → Ezber (train iyi, val kötü)
□ "Neden train/val/test?" → Ezber önlemek
□ "Neden ölçekleme?" → Loss yüzeyini yuvarlatmak
□ "Leakage nedir?" → Test bilgisi sızdı
□ "Precision vs Recall?" → İsabet vs Yakalama
□ "Neden LR küçük olmalı?" → Patlamayı önlemek

Hepsi ✓ ise → Week 1'e hazırsın! 🚀
```

---

## 1️⃣7️⃣ Tek Paragraf Özet (Hafızana Kazınsın)

> **Makine öğrenmesi**, veriye bakıp bir **kural** bulma işidir. Bu kuralı, hatamızı tek sayıya indirgeyen **kayıp** ile ölçer, **gradyan adımları** ile parametreleri düzeltiriz. **Doğrulama kümesi** bize "ezberledin mi?" diye sorar; gerekirse **erken durdurur** ve **düzenleriz**. **Ölçekleme**, **doğru metrik**, **dürüst veri ayrımı** ve **küçük ama sürekli deneyler** başarıyı garanti eder. Temel bu; üstüne her şeyi inşa edebiliriz.

---

## 🎓 Sonraki Adım

### 📚 Okuma Sırası

```
✅ Bu döküman (theory_intro.md)
   └─ Sezgi oturdu ✓

⬜ theory_foundations.md
   └─ Biraz daha detay + görsel

⬜ theory_mathematical.md
   └─ Matematiksel derinlik (opsiyonel)

⬜ Kurulum & Setup
   └─ PyTorch MPS test

⬜ Week 1: Linear Regression
   └─ PRATIK!
```

### 🚀 Hazır mısın?

```bash
cd /Users/onur/code/novadev-protocol
source .venv/bin/activate

# Week 1'e başla:
python week1_tensors/linreg_manual.py
```

**Başarı Kriteri:**
> Week 1'de kod yazarken:
> "Aha! Gradient burası!"
> "İşte overfit belirtisi!"
> diyebilmen.

---

**🎉 Tebrikler! Week 0 Introduction tamamlandı!**

**Şimdi:** Kod yazmadan önce **zihnindeKİ model** oluştu.
**Sonra:** Kodu yazdığında "neden böyle?" bileceksin.
**Sonuç:** Daha az hata, daha hızlı öğrenme, daha derin anlayış!

**Hazır ol, Week 1 geliyor!** 💪

