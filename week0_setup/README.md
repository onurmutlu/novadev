# Hafta 0: Kurulum & Kas Isıtma

**Hedef:** Tüm araçları kur, PyTorch MPS'yi doğrula, Ollama ile ilk deneme yap.

---

## 📚 Teori Notları (ÖNEMLİ!)

Week 1'e başlamadan önce bu dökümanları oku. Kod yazmadan önce **zihinsel modeli** doğru kurmak kritik.

### 📖 Ders Dökümanları

1. **[Temel Kavramlar](theory_foundations.md)** ⭐
   - Model, veri, loss, gradient, tensör, autograd
   - 45-60 dakika okuma
   - **Hedef:** ML'in temellerini kavramak

2. **[İleri Konular & Pratik](theory_advanced.md)**
   - Reproduksiyon, donanım, debug, gerçek hayat
   - 30-45 dakika okuma
   - **Hedef:** Profesyonel pratikler

### 🎯 Okuma Sırası

```
Day 0 Sabah:    theory_foundations.md (Temel kavramlar)
Day 0 Öğleden:  theory_advanced.md (İleri konular)
Day 0 Akşam:    Kurulum ve testler (bu döküman)
```

**Not:** Kod yazmadan teoriyi okumak **zaman kaybı DEĞİL**. Week 1'de çok daha hızlı ilerlersin.

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
