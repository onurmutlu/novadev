# Hafta 0: Kurulum & Kas Isıtma

**Hedef:** Tüm araçları kur, PyTorch MPS'yi doğrula, Ollama ile ilk deneme yap.

---

## 📚 Teori Notları (ÖNEMLİ!)

Week 1'e başlamadan önce bu dökümanları oku. Kod yazmadan önce **zihinsel modeli** doğru kurmak kritik.

### 📖 Ders Dökümanları (3 Seviye)

#### Seviye 1: Temel Kavramlar (Başlangıç)
**[Temel Kavramlar - Sezgisel Bakış](theory_foundations.md)** ⭐
- Model, veri, loss, gradient nedir?
- Tensör operasyonları, autograd sezgisi
- Overfit/underfit, optimizer seçimi
- **Süre:** 60-90 dakika
- **Hedef:** ML kavramlarını **görselleştir**ebilme

#### Seviye 2: Matematiksel Temeller (Orta-İleri)
**[Matematiksel Temeller - Hocanın Tahtası](theory_mathematical.md)** 📐
- Loss fonksiyonlarının probabilistik kökenleri (MLE, MAP)
- Optimizasyon matematiği (condition number, curvature)
- Feature engineering derinliği
- **Süre:** 90-120 dakika (2 bölüm)
- **Hedef:** **"Neden?"** sorularına cevap

**[Matematiksel Temeller Part 2](theory_mathematical_part2.md)**
- Bias-variance matematiği
- Regularization teorisi (L1/L2 probabilistik köken)
- Metrik seçimi, deney disiplini
- Debug protokolü

#### Seviye 3: İleri Konular & Pratik (Uzman)
**[İleri Konular & Saha Deneyimi](theory_advanced.md)** 🎯
- Reproduksiyon stratejileri
- Donanım optimizasyonu (MPS/CUDA)
- Sayısal stabilite (NaN, gradient explosion)
- Gerçek hayat hikayeleri

### 🎯 Önerilen Okuma Sırası

```
┌─────────────────────────────────────────────┐
│ Day 0 - Sabah (90-120 dk)                   │
├─────────────────────────────────────────────┤
│ 1. theory_foundations.md                    │
│    └─ Sezgisel kavramlar, görsel açıklamalar│
│                                              │
│ 2. theory_mathematical.md (Part 1)          │
│    └─ Matematiksel derinlik başlangıç       │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Day 0 - Öğleden Sonra (90-120 dk)           │
├─────────────────────────────────────────────┤
│ 3. theory_mathematical_part2.md             │
│    └─ Probabilistik bakış, regularization   │
│                                              │
│ 4. theory_advanced.md                       │
│    └─ Pratik ipuçları, debug                │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Day 0 - Akşam (30-45 dk)                    │
├─────────────────────────────────────────────┤
│ 5. Kurulum ve testler (bu döküman)          │
│    └─ PyTorch MPS, Ollama, setup verify     │
└─────────────────────────────────────────────┘
```

### 💡 Hangi Seviyeden Başlamalıyım?

**Yeni Başlayan:**
```
theory_foundations.md → Setup → Week 1
(Matematiksel kısmı Week 1'den sonra oku)
```

**Orta Seviye (Python + biraz matematik):**
```
theory_foundations.md → theory_mathematical.md → Setup → Week 1
```

**İleri Seviye:**
```
Hepsini sırayla oku, Week 1'de derin anlayışla başla
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
