# Hafta 0: Kurulum & Kas Isıtma

**Hedef:** Tüm araçları kur, PyTorch MPS'yi doğrula, Ollama ile ilk deneme yap.

---

## 📚 Teori Notları (ÖNEMLİ!)

Week 1'e başlamadan önce bu dökümanları oku. Kod yazmadan önce **zihinsel modeli** doğru kurmak kritik.

### 📖 Ders Dökümanları (5 Seviye - Kademe Kademe)

#### Seviye 0: Sıfırdan Başlangıç (Lise) 🌱
**[Makine Öğrenmesine Giriş - İlk Adım](theory_intro.md)** 
- "ML nedir?" tek cümle + günlük örnekler (Netflix, Spam, Klavye)
- Kod yok, formül yok, bol benzetme (Radyo, Dağ, Fatura)
- Mini quiz (4 soru) + pratik alıştırma (3 problem)
- En sık 10 hata + çözüm
- Sözlük (cep kartı)
- **Süre:** 45-60 dakika
- **Hedef:** "Ah ha! Demek bu kadar basit!" anı

#### Seviye 1: Temel Kavramlar (Üniversite Giriş) 📚
**[Core Concepts - Akademik Yaklaşım](theory_core_concepts.md)** ⭐ YENİ!
- Formal tanımlar (f_θ, parametreler, loss, gradient)
- Matematiksel çerçeve (hafif matematik, korkutmadan)
- Probabilistik kökenler (MSE ← Gaussian MLE, CE ← Bernoulli)
- L2/L1'in MAP bağlantısı (Gaussian/Laplace prior)
- Bias-variance trade-off
- **Süre:** 90-120 dakika
- **Hedef:** Week 1'de "bu formül nereden geldi?" bilmek

#### Seviye 2: Sezgisel Derinlik (Görselleştirme) 🎨
**[Foundations - Sezgisel Bakış](theory_foundations.md)**
- Model, veri, loss, gradient DETAYLIdır
- Tensör operasyonları, autograd akışı
- Overfit/underfit, optimizer seçimi
- Bol görsel açıklama
- **Süre:** 60-90 dakika
- **Hedef:** Kavramları **görselleştir**ebilme

#### Seviye 3: Matematiksel Temeller (Hocanın Tahtası) 📐
**[Mathematical Foundations Part 1](theory_mathematical.md)**
- i.i.d. varsayımı ve ihlalleri (covariate/concept/prior shift)
- Data leakage detayları (temporal, target, preprocessing)
- Condition number, curvature
- Feature engineering matematiği
- Loss fonksiyonları (MSE/MAE/Huber/CE/Focal)
- **Süre:** 90 dakika
- **Hedef:** Matematiksel **derinlik**

**[Mathematical Foundations Part 2](theory_mathematical_part2.md)**
- Sayısal koşullar, Hessian
- MLE/MAP türetim detayları
- Bias-variance matematiksel ayrıştırma
- Regularization teorisi derinliği
- Metrik matematiği (ROC, PR, calibration)
- Deney disiplini, hyperparameter search
- **Süre:** 60-90 dakika
- **Hedef:** "Neden?" sorularına **tam cevap**

#### Seviye 4: İleri Konular & Saha (Uzman) 🎯
**[Advanced Topics - Pratik Deneyim](theory_advanced.md)**
- Reproduksiyon stratejileri
- Donanım optimizasyonu (MPS/CUDA/CPU)
- Sayısal stabilite (NaN, gradient explosion, clipping)
- Debug protokolü
- Gerçek hayat hikayeleri (saha deneyimi)
- **Süre:** 30-45 dakika
- **Hedef:** Profesyonel **pratikler**

#### 🎓 Final: Kapanış & Self-Assessment
**[Week 0 Kapanış Dökümanı](theory_closure.md)** ⭐ ZORUNLU!
- Son kontrol listesi (soru-cevap formatında)
- Mini-ödev çözümleri (3 problem analizi)
- LR semptomları tablosu
- Overfit/Underfit first-aid
- Tensor checklist
- **Süre:** 30-45 dakika
- **Hedef:** "Week 0'ı gerçekten özümsedim mi?" kendini test et

### 🎯 Önerilen Okuma Sırası

```
┌──────────────────────────────────────────────────┐
│ Day 0 - Sabah (2.5-3 saat)                       │
├──────────────────────────────────────────────────┤
│ 1. theory_intro.md (45-60 dk) 🌱                 │
│    └─ Sıfırdan başlangıç, lise seviyesi          │
│    └─ Günlük örnekler, benzetmeler               │
│    └─ Hedef: "Ah ha!" anı                        │
│                                                   │
│ 2. theory_core_concepts.md (90-120 dk) 📚 ⭐ YENİ│
│    └─ Üniversite seviyesi, formal tanımlar       │
│    └─ Hafif matematik (θ, ∇L, MLE, MAP)         │
│    └─ Hedef: "Formül nereden geldi?" bilmek      │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│ Day 0 - Öğleden Sonra (2.5-3 saat)               │
├──────────────────────────────────────────────────┤
│ 3. theory_foundations.md (60-90 dk) 🎨           │
│    └─ Sezgisel derinlik, görsel açıklamalar      │
│    └─ Tensör, autograd, optimizer detayları      │
│                                                   │
│ 4. theory_mathematical.md (90 dk) 📐             │
│    └─ Matematiksel temeller (Part 1)             │
│    └─ i.i.d., leakage, condition number          │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│ Day 0 - Akşam (2-2.5 saat)                       │
├──────────────────────────────────────────────────┤
│ 5. theory_mathematical_part2.md (60-90 dk)       │
│    └─ MLE/MAP türetimler, bias-variance          │
│    └─ Metrik matematiği, deney disiplini         │
│                                                   │
│ 6. theory_advanced.md (30-45 dk) 🎯              │
│    └─ Pratik ipuçları, donanım, debug            │
│    └─ Saha deneyimleri                           │
│                                                   │
│ 7. theory_closure.md (30-45 dk) ⭐ ZORUNLU       │
│    └─ Son kontrol listesi, self-assessment       │
│    └─ Mini-ödev çözümleri                        │
│    └─ "Week 0'ı özümsedim mi?" test              │
│                                                   │
│ 8. Kurulum ve testler (30 dk)                    │
│    └─ PyTorch MPS, Ollama, setup verify          │
│    └─ Bu dökümanın devamı ↓                      │
└──────────────────────────────────────────────────┘

TOPLAM: 7-9 saat (yoğun ama TAM kapsamlı!)
```

### 💡 Hangi Seviyeden Başlamalıyım?

#### 🌱 Tam Yeni Başlayan (ML hiç bilmiyorum)
```
Yol:
  theory_intro.md
  → theory_closure.md (kendi cevaplarını yaz!) ⭐
  → Setup → Week 1
  (Diğerlerini Week 1 sırasında/sonra oku)

Süre: 1.5 saat teori + 30 dk setup

Neden?
  ✓ Önce sezgi kazan (radyo, dağ analojileri)
  ✓ Korkutmadan başla (formül yok)
  ✓ Self-check ile pekiştir (closure)
  ✓ Kod yazarak pekiştir
  ✓ Sonra derinleş (geri dön theory_core_concepts'e)

Sonuç:
  Week 1'de "ne yaptığımı biliyorum" rahatı
```

#### 📚 Orta Seviye (Python biliyorum, ML az var)
```
Yol:
  theory_intro.md (45 dk, gözden geçir)
  → theory_core_concepts.md (90 dk, ÖNEMLİ!) ⭐
  → theory_foundations.md (60 dk)
  → theory_closure.md (30 dk, self-check!) ⭐
  → Setup → Week 1
  (Matematiksel kısmı Week 1'den sonra oku)

Süre: 3.5-4.5 saat teori + 30 dk setup

Neden?
  ✓ Boşlukları doldur (formal tanımlar)
  ✓ "Neden MSE, neden L2?" öğren
  ✓ Terminolojiyi yerleştir (θ, ∇L, MLE)
  ✓ Self-check ile teyit et (closure)
  ✓ Week 1 kodunu ANLAYARAK yaz

Sonuç:
  Week 1'de "bu formül nereden geldi" bileceksin
```

#### 📐 İleri Seviye (ML background var, derinlemek istiyorum)
```
Yol:
  Hepsini sırayla oku (theory_intro → core_concepts
  → foundations → mathematical 1&2 → advanced
  → closure!) ⭐
  → Setup → Week 1

Süre: 7-9 saat teori + 30 dk setup

Neden?
  ✓ Matematiksel temelleri TAM otur
  ✓ "Neden?" sorularına DERİN cevaplar
  ✓ MLE/MAP bağlantılarını GÖR
  ✓ Bias-variance matematiğini ANLA
  ✓ Saha deneyimlerini AL
  ✓ Self-assessment ile teyit et

Sonuç:
  Week 1'i akademik derinlikle yaz, literatür okumaya hazır ol
```

#### 🚀 Hızlı Track (Zaman dar, tecrübe var)
```
Yol:
  theory_core_concepts.md (ÖNEMLİ!)
  + theory_advanced.md (debug, pratik)
  + theory_closure.md (self-check!) ⭐
  → Setup → Week 1

Süre: 2.5-3 saat

Neden?
  ✓ Temel formülleri kapat (MLE, MAP)
  ✓ Pratik ipuçlarını al (debug protokolü)
  ✓ Self-check ile boşluk bul
  ✓ Direkt Week 1'e geç

Sonuç:
  Hızlı başla ama sağlam temel + özgüven
```

### ⚠️ Önemli Not

**Kod yazmadan teoriyi okumak zaman kaybı DEĞİL!**

Sebebi:
- Week 1'de **çok daha hızlı** ilerlersin
- "Neden?" sorularına **anında** cevap verebilirsin
- Debug yaparken **sistematik** düşünürsün
- **Literatür** okumaya hazır olursun

> "Teori olmadan pratik kördür, pratik olmadan teori anlamsızdır."

---

## ✅ Kurulum Adımları

### 1. Python & Virtual Environment

```bash
# Python versiyonu kontrol (3.11+)
python3 --version

# Repo ana dizininde venv oluştur
python3 -m venv .venv
source .venv/bin/activate

# Temel paketleri yükle
pip install --upgrade pip
pip install -e .
pip install -e ".[dev]"
```

### 2. PyTorch MPS (Metal) Doğrulama

```bash
python week0_setup/hello_tensor.py
```

**Beklenen çıktı:**
```
✅ Using MPS (Metal Performance Shaders)
Tensor device: mps
Random tensor shape: torch.Size([3, 4])
Matrix multiplication result shape: torch.Size([3, 4])
```

### 3. Ollama Kurulumu (Opsiyonel ama önerilen)

```bash
# Homebrew ile kur
brew install ollama

# Servis başlat (arka planda)
ollama serve &

# 7B model indir (birini seç)
ollama pull qwen2.5:7b      # Çin menşeli, çok iyi
ollama pull llama3.2:7b     # Meta'nın son modeli

# Test et
ollama run qwen2.5:7b "Explain tensors in one sentence"
```

### 4. Ruff & Pytest Doğrulama

```bash
# Code formatting kontrol
ruff check .

# Testleri çalıştır (henüz boş)
pytest tests/
```

---

## 📁 Dosyalar

- `hello_tensor.py`: MPS device doğrulama
- `ollama_test.py`: Ollama API ile basit prompt denemesi
- `nova-setup.md`: Kurulum tamamlandı işareti (sen oluşturacaksın)

---

## 🎯 Teslim (Hafta 0 Sonu)

1. **`nova-setup.md`** dosyası oluştur (aşağıdaki template):

```markdown
# NovaDev Kurulum Tamamlandı

**Tarih:** [YYYY-MM-DD]

## ✅ Tamamlanan Kurulumlar

- [ ] Python 3.11+ kurulu
- [ ] Virtual environment aktif
- [ ] PyTorch kurulu ve MPS çalışıyor
- [ ] Ollama kurulu ve 7B model indirildi
- [ ] Ruff & Pytest çalışıyor

## 🖥️ Sistem Bilgileri

- **OS:** macOS (M3)
- **Python:** [version]
- **PyTorch:** [version]
- **Device:** MPS

## 📊 hello_tensor.py Çıktısı

[Buraya hello_tensor.py çıktısını yapıştır]

## 🧠 Ollama Test Sonucu

[Buraya ollama test promptunu ve yanıtını yapıştır]

## 💭 Notlar

- MPS ile karşılaşılan sorunlar (varsa)
- Kurulum sırasında öğrenilenler
- Hafta 1 için hazırlık notları
```

2. **İlk commit:**

```bash
git add .
git commit -m "day0: Setup complete - PyTorch MPS verified, Ollama ready"
git push
```

---

## 🔧 Troubleshooting

### MPS Bulunamıyor Hatası

```bash
# PyTorch yeniden kur (nightly bazen daha stabil)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Ollama Bağlantı Hatası

```bash
# Servis çalışıyor mu?
ps aux | grep ollama

# Yeniden başlat
pkill ollama
ollama serve
```

---

**Sonraki Adım:** Hafta 1 → Tensör matematiği ve linear regression
