# 🚀 NovaDev v1.1 — AI + Crypto Paralel Program

**"Öğrenirken İki Gemi Yap: 8 Hafta, 80-100 Saat, İki Çalışan Sistem"**

> Bu bir AI kursu veya trading kursu DEĞİL; teori + pratik + ürün birlikte ilerleyen **yaparak öğrenme protokolü**. **İki paralel hat:** AI (ML temelleri) + Crypto (on-chain istihbarat).

**⚠️ Yasal Uyarı:** Crypto hattı bilgilendirme amaçlıdır, finansal tavsiye değildir. Read-only, testnet-first.

---

## 🎯 Amaç

**8 Haftada İki Sistem:**

### 1️⃣ AI Hattı
- ✅ ML temelleri (Linear regression → MLP → NLP)
- ✅ RAG (dokümandan kaynaklı yanıt)
- ✅ Tool-Agent (araç çağırma)
- ✅ FastAPI servis

### 2️⃣ Crypto Hattı
- ✅ On-Chain Intel Copilot (read-only)
- ✅ Event collector + DuckDB
- ✅ Telegram uyarı botu
- ✅ Protokol RAG + simülasyon

### Entegre Çıktı
- ✅ FastAPI (AI + Crypto endpoints)
- ✅ Docker Compose deployment
- ✅ 5 dakikalık capstone demo

**Başarı Kriteri:** _"Aynı iki sistemi yarın tek başına kurabilir misin?" → **EVET**_

---

## 📚 Dökümantasyon

### 🗺️ Genel Bakış
**[docs/program_overview.md](docs/program_overview.md)** ⭐⭐⭐ ÖNCE OKU!
- **AI + Crypto Paralel Program** (tam syllabus)
- T→P→X ritmi (Teori/Pratik/Ürün)
- 8 haftalık detaylı plan (her hafta DoD/KPI)
- Güvenlik ilkeleri & rapor disiplini
- Haftalık komut planları

**Ek Detaylar:**
- [docs/overview.md](docs/overview.md) - AI hattı derinlemesine
- [docs/crypto_overview.md](docs/crypto_overview.md) - Crypto hattı derinlemesine

### 📖 Week 0: Teori Temelleri (TAMAMLANDI ✓)
**[week0_setup/README.md](week0_setup/README.md)**
- 7061 satır teori notları (7 döküman, 5 seviye)
- Self-assessment (theory_closure.md)
- Setup & verification

### 💻 Week 1: Linear Regression (ŞİMDİ BAŞLA!)
**[week1_tensors/README.md](week1_tensors/README.md)**
- 45 dakikalık hızlı sprint
- 5 günlük detaylı plan
- Hedef: Val MSE < 0.5

---

## 🧱 Program Mimarisi (3 Hat Birlikte)

```
┌─────────────────────────────────────┐
│ TEMEL (T) - Kavram + Sezgi          │
│ "Neden MSE?" → Gaussian MLE         │
│ Süre: 45-60 dk/hafta                │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ PRATİK (P) - Kod & Deney            │
│ MSE kodla → Val MSE < 0.5           │
│ Süre: 60-90 dk/gün                  │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ ÜRÜN (X) - Servis & İzleme          │
│ /predict endpoint → Docker          │
│ Süre: 60-90 dk/hafta                │
└─────────────────────────────────────┘
```

**T → P → X Döngüsü:** Önce anla, sonra dene, sonra gönder.

---

## 🗺️ 8 Haftalık Roadmap (AI + Crypto Paralel)

| Hafta | AI Hattı | Crypto Hattı | DoD/KPI |
|-------|----------|--------------|---------|
| **0** ✅ | Temel Zihin (7061 satır teori) | RPC + Event Ingest | Setup ✓ |
| **1** 👉 | Linear Regression | Veri Katmanı | AI: MSE<0.5 / Crypto: JSON report |
| **2** | MLP + MNIST | Telegram Bot v0 | AI: acc≥0.97 / Crypto: 2+ alerts |
| **3** | NLP (Türkçe BERT) | Event Classifier | AI: F1≥0.85 / Crypto: F1≥0.80 |
| **4** | RAG Pipeline | Protokol RAG | Recall@k≥60% / Sourced≥95% |
| **5** | Tool-Agent | Simülasyon | 2-step chain / Quote<2s |
| **6** | LoRA Fine-tune | Üslup Uyarlama | A/B≥60% / Citation≥95% |
| **7** | FastAPI Entegre | Servis + İzleme | p95<2.5s, error<1% |
| **8** | Capstone E2E | 3 Scenario Demo | 5dk video + setup<10dk |

**Detay:** [docs/program_overview.md](docs/program_overview.md) ⭐ TAM SYLLABUS

---

## 🚀 Hızlı Başlangıç

### Kurulum (15 dk)

```bash
# 1. Repo klonla
git clone <repo-url>
cd novadev-protocol

# 2. Python 3.11+ kontrol
python3 --version

# 3. Virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 4. Dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# 5. MPS test (Apple Silicon)
python week0_setup/hello_tensor.py
# Output: "MPS is available! ✓"

# 6. Pytest & Lint
pytest -q
ruff check .
```

### Sonraki Adım

```bash
# Week 0 teoriyi tamamladın mı?
cat week0_setup/theory_closure.md

# Self-check:
# □ Train/Val/Test farkını biliyorum
# □ MSE/MAE ne zaman kullanılır biliyorum
# □ LR semptomlarını tanıyorum
# □ Overfit/Underfit teşhis edebiliyorum

# Hepsi ✓ ise Week 1'e başla:
cd week1_tensors
cat README.md  # 45 dk hızlı sprint planı
```

---

## 📋 Günlük / Haftalık Ritim

### Günlük (2-3 saat)

```
Sabah (30 dk):
  1. Hedef belirle (1 cümle)
  2. Plan yap (3 madde)
  3. Teori notunu oku (15-30 dk)

Öğlen (90 dk):
  4. Kod/Deney yap
  5. Metrikleri kaydet

Akşam (15 dk):
  6. Log + Özet (exp_log.csv)
  7. Git commit
```

**Kural:** _"Bugün 1 deney koşmadıysan, öğrenmedin."_

### Haftalık

```
Pazartesi:      Hedef & Plan
Salı-Perşembe:  Deneyler & Ablation
Cuma:           Rapor & Demo
```

---

## 🗂️ Repo Yapısı

```
novadev-protocol/
├── docs/
│   ├── overview.md           ⭐ Genel bakış (önce oku!)
│   └── week0_kapanis.md      Self-assessment
│
├── week0_setup/              ✅ Teori (7061 satır)
│   ├── README.md
│   ├── theory_intro.md       (Lise seviyesi)
│   ├── theory_core_concepts.md (Üniversite)
│   ├── theory_foundations.md (Sezgisel)
│   ├── theory_mathematical.md (Matematik)
│   ├── theory_mathematical_part2.md
│   ├── theory_advanced.md    (Pratik & Saha)
│   └── theory_closure.md     (Self-assessment)
│
├── week1_tensors/            👉 Linear Regression (şimdi!)
│   ├── README.md             (45 dk sprint + 5 gün plan)
│   ├── data_synth.py
│   ├── linreg_manual.py
│   ├── linreg_module.py
│   └── train.py
│
├── week2_mlp/                MLP + MNIST
├── week3_nlp/                BERT fine-tune
├── week4_rag/                RAG pipeline
├── week5_agent/              Tool-agent
├── week6_lora/               LoRA fine-tune
├── week7_service/            FastAPI + Docker
├── week8_capstone/           Demo
│
├── common/                   Shared utils
├── tests/                    Haftalık testler
├── outputs/                  Metrikler & grafikler
├── reports/                  Haftalık raporlar
│
├── pyproject.toml            Dependencies
├── .gitignore
└── README.md                 (Bu dosya)
```

---

## 📏 Değerlendirme (Gating Criteria)

### Her Hafta Gerekli

```
1. Metrik Eşiği (Geçilmeli!)
   Week 1: Val MSE < 0.5
   Week 2: Test acc ≥ 0.97
   ...

2. Artifact (Kanıt)
   - Grafik (loss curve, confusion matrix)
   - Log (exp_log.csv)
   - Rapor (weekX_report.md)

3. Özet (3-5 Madde)
   - Ne çalıştı?
   - Neden? (teori bağlantısı)
   - Bir dahaki sefere?
```

**Kural:** Eşiği geçmeden sonraki haftaya geçme!

**Detay:** [docs/overview.md](docs/overview.md#değerlendirme--geçiş-eşiği-gating)

---

## 🎯 Başarı Kriterleri (KPI)

```
✅ Week 0: Theory + Setup complete
👉 Week 1: Val MSE < 0.5
⬜ Week 2: Test accuracy ≥ 0.97
⬜ Week 3: F1 ≥ 0.85
⬜ Week 4: Recall@k ≥ 60%
⬜ Week 5: 2-step tool chain
⬜ Week 6: Qualitative improvement (A/B)
⬜ Week 7: p95 latency < 2.5s
⬜ Week 8: 5 dk demo + kurulum < 10 dk
```

---

## 🔧 Araçlar / Stack

**Kod & ML:**
```
Python 3.11+
PyTorch 2.x (MPS support)
scikit-learn
transformers (Hugging Face)
sentence-transformers
```

**Servis & Deploy:**
```
FastAPI
Docker & Docker Compose
```

**Arama & LLM:**
```
FAISS / Chroma
Ollama (lokal 7B/8B)
```

**Geliştirme:**
```
pytest (testing)
ruff (linting)
Git (version control)
```

---

## 🔥 Hızlı Linkler

### Week 0 (Teori) ✅
- [Week 0 README](week0_setup/README.md) - Teori notları genel bakış
- [theory_intro.md](week0_setup/theory_intro.md) - Lise seviyesi giriş
- [theory_core_concepts.md](week0_setup/theory_core_concepts.md) - Üniversite
- [theory_closure.md](week0_setup/theory_closure.md) - Self-assessment ⭐

### Week 1 (Pratik) 👉
- [Week 1 README](week1_tensors/README.md) - 45 dk sprint + 5 gün plan
- [data_synth.py](week1_tensors/data_synth.py) - Sentetik veri
- [linreg_manual.py](week1_tensors/linreg_manual.py) - Manuel GD
- [train.py](week1_tensors/train.py) - Full training loop

### Genel
- [docs/overview.md](docs/overview.md) - Program genel bakış ⭐
- [pyproject.toml](pyproject.toml) - Dependencies

---

## 💡 M3 Mac Notları

**MPS (Metal Performance Shaders):**
- PyTorch MPS backend aktif
- Week 0-3: MPS yeterli
- Week 4+: Ollama lokal LLM

**Büyük Modeller:**
- Ollama 7B/8B lokal
- LoRA: düşük batch + gradient accumulation
- Docker: CPU fallback (MPS container sınırlı)

**Donanım Gerçeği:**
```
M3 Mac ile:
✅ Küçük-orta modeller (< 1B): MPS
✅ 7B-8B LLM: Ollama + quantization
⚠️ 13B+: API fallback (OpenAI/Anthropic)
```

---

## 🧭 Pedagoji (Neden Böyle?)

### Sarmal Öğrenme
Her hafta önceki bilgi üzerine inşa edilir.

### Ablation Disiplini
"L2 gerçekten işe yarıyor mu?" → Deney ile kanıtla!

### Product-First
Her hafta çalışan bir çıktı (CLI/API/Demo)

### Hata Odaklı
Sorun çıkması = Öğrenme fırsatı

**Detay:** [docs/overview.md](docs/overview.md#pedagoji-neden-böyle)

---

## 🧯 Tıkandığında (Quick Debug)

```
1. LR çok mu? → Yarıya indir
2. Ölçekleme var mı? → StandardScaler ekle
3. zero_grad() var mı? → Kontrol et
4. Loss doğru mu? → Regresyon: MSE, Sınıflama: CE
5. Shape tutarlı mı? → print(x.shape, x.dtype, x.device)
6. Seed sabit mi? → torch.manual_seed(42)
```

**Referans:** [week0_setup/theory_closure.md](week0_setup/theory_closure.md)

---

## 📊 İlerleme Takibi

### Week 0: Temel Zihin ✅ TAMAMLANDI
```
✅ 7061 satır teori notları
✅ Self-assessment (theory_closure.md)
✅ MPS functional test
✅ pytest & ruff yeşil
```

### Week 1: Linear Regression 👉 ŞİMDİ
```
Hedef: Val MSE < 0.5

Bugün (45 dk):
  □ Data synth + Manuel GD (15 dk)
  □ nn.Module + Training (15 dk)
  □ Test + Lint (10 dk)
  □ Mini rapor (5 dk)

5 Gün Sprint:
  Day 1: E2E + Logging
  Day 2: Early stopping + LR schedule
  Day 3: Scaling ablation
  Day 4: Loss curves + overfit
  Day 5: Final report

Definition of Done:
  □ Val MSE < 0.5 ✓
  □ Loss curves (normal + overfit)
  □ exp_log.csv (5+ experiment)
  □ week1_report.md (teori bağlantılı)
  □ pytest yeşil
```

### Week 2-8: Devam edecek...

---

## ❓ SSS

**Bu bir kurs mu?**
> Hayır, **yaparak öğrenme protokolü**. Her hafta ölçülebilir hedef + gerçek çıktı.

**Sertifika var mı?**
> Portfolyo > Sertifika. Capstone + Repo = CV'de gösterebileceğin KANIT.

**Az vaktim var, ne yapmalıyım?**
> Metrik eşiğini koru, kapsamı küçült. Eşik geçilmeli, süre esnek.

**Daha detaylı bilgi?**
> [docs/overview.md](docs/overview.md) - Tüm detaylar burada.

---

## 📚 Kaynaklar

**Resmi Dökümantasyon:**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Hugging Face Course](https://huggingface.co/learn/nlp-course)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

**Araçlar:**
- [Ollama](https://ollama.ai/)
- [PEFT/LoRA](https://huggingface.co/docs/peft)
- [FAISS](https://github.com/facebookresearch/faiss)

**NovaDev Dökümanları:**
- [Program Overview](docs/overview.md) ⭐
- [Week 0: Teori](week0_setup/README.md)
- [Week 1: Pratik](week1_tensors/README.md)

---

## 🎉 Sonraki Adım

```bash
# 1. Overview'ı oku (10 dk)
cat docs/overview.md

# 2. Week 0'ı tamamladın mı? (Self-check)
cat week0_setup/theory_closure.md

# 3. Hepsi ✓ ise Week 1'e başla!
cd week1_tensors
python data_synth.py

# Week 0'da öğrendiklerini KODDA GÖR! 💪
```

---

**NovaDev Protocol — "Öğrenirken Gemi Yap"**

*Versiyon: 1.0*  
*Son Güncelleme: 2025-10-06*  
*Status: Week 0 ✅ Complete | Week 1 👉 Ready to Start*