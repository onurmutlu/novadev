# 🧠 NovaDev Master Program v1.1
> **AI + Crypto Dual-Track | 9-Week Mastery Roadmap**

[![Version](https://img.shields.io/badge/version-1.1.0-blue)](CHANGELOG.md)
[![AI Master Track](https://img.shields.io/badge/AI-Mastery%20(1%2C536%20lines)-brightgreen)](docs/AI_TRACK_OUTLINE.md)
[![Crypto Master Track](https://img.shields.io/badge/Crypto-Mastery%20(1%2C218%20lines)-9cf)](docs/CRYPTO_TRACK_OUTLINE.md)
[![Progress](https://img.shields.io/badge/Progress-Week%201%20Launch-orange)](WEEK1_MASTER_PLAN.md)
[![Docs](https://img.shields.io/badge/docs-31%2C617%2B%20lines-green)](docs/)
[![Security](https://img.shields.io/badge/crypto-read--only-orange)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)
[![Tests](https://img.shields.io/badge/tests-39%2F39%20passing-success)](#)

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

**8 Haftalık Master Track'ler:**
- 🧠 [docs/AI_TRACK_OUTLINE.md](docs/AI_TRACK_OUTLINE.md) ⭐ **YENİ!** (1,536 satır, Week 1-8 AI roadmap)
- 🪙 [docs/CRYPTO_TRACK_OUTLINE.md](docs/CRYPTO_TRACK_OUTLINE.md) ⭐ **YENİ!** (1,218 satır, Week 1-8 Crypto roadmap)

**Ek Detaylar:**
- [docs/overview.md](docs/overview.md) - AI hattı derinlemesine
- [docs/crypto_overview.md](docs/crypto_overview.md) - Crypto hattı derinlemesine

### 📖 Week 0: Teori Temelleri (✅ TAMAMLANDI)
**[week0_setup/README.md](week0_setup/README.md)**
- 7061 satır teori notları (7 döküman, 5 seviye)
- Self-assessment (theory_closure.md)
- Setup & verification

**[crypto/docs/w0_bootstrap/README.md](crypto/docs/w0_bootstrap/README.md)** ⭐ YENİ!
- 🎓 "Hoca Tahtası" serisi (10/10 ders tamamlandı)
- 19,005 satır dokümantasyon (10-12 saat ders)
- Production-ready kod örnekleri
- 13 runbook + troubleshooting guide
- Status: **COMPLETE** ✅

### 💻 Week 1: Linear Regression + Collector (👉 ŞİMDİ BAŞLA!)
**[WEEK1_MASTER_PLAN.md](WEEK1_MASTER_PLAN.md)** ⭐ YENİ PLAN!
- 1,248 satır detaylı 5-günlük sprint
- AI: Val MSE ≤ 0.50 hedefi
- Crypto: API p95 < 1s hedefi
- Günlük komutlar + DoD + metrik takibi

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
| **0** ✅ | Temel Zihin (7061 satır) + Tahta Serisi (19,005 satır) | RPC + Event Ingest + 10 Ders Complete | Setup ✓, Tests 39/39 ✓ |
| **1** 👉 | Linear Regression | Collector Loop + API Perf | AI: MSE≤0.5 / Crypto: p95<1s |
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

# 4. Dependencies (AI + Crypto)
pip install --upgrade pip
pip install -e ".[dev,crypto]"

# 5. Verify install
python week0_setup/hello_tensor.py  # MPS test (AI)
# Output: "MPS is available! ✓"

pytest -q       # Tests
ruff check .    # Lint
```

### AI Quick Start (30 dk)

```bash
# ✅ Week 0 Theory Complete!
cat week0_setup/theory_closure.md

# Self-check:
# ✓ Train/Val/Test farkını biliyorum
# ✓ MSE/MAE ne zaman kullanılır biliyorum
# ✓ LR semptomlarını tanıyorum
# ✓ Overfit/Underfit teşhis edebiliyorum

# 👉 Week 1'e başla (READY!)
cat WEEK1_MASTER_PLAN.md  # 1,248 satır detaylı 5-günlük sprint
python week1_tensors/linreg_module.py --lr 0.01 --l2 0.001
```

### Crypto Quick Start (30-45 dk) ✅ PRODUCTION-READY!

```bash
# 1. RPC Provider Setup (Alchemy/Infura)
# https://dashboard.alchemy.com → Create App → Sepolia

# 2. Configure .env
cd crypto/w0_bootstrap
cp .env.example .env
# vim .env → RPC_URL yapıştır

# 3. Test RPC (5 dk)
make c.health
# → ✅ RPC OK | latest block: 12345678 | 145.3 ms

# 4. Capture Events (10 dk) - Idempotent & State-tracked
make c.capture.idem
# → ✅ Scanning 123456..123500 (latest=123505, buffer=5)
# → ✅ Done. State last_scanned_block=123500

# 5. Wallet Report (JSON + Validation)
make report.schema W=0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
# → ✅ report_v1 schema valid

# 6. FastAPI Service (5 dk)
make c.api
# Terminal 2: curl http://localhost:8000/healthz
# Browser: http://localhost:8000/docs  (OpenAPI/Swagger)

# 7. Test Endpoint
curl "http://localhost:8000/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report?hours=24"
# → {"wallet":"0x...","inbound":12.34,"outbound":5.67,...}

# Hepsi ✓ ise:
cat crypto/README.md  # 8 haftalık roadmap
```

### Makefile Komutları (Kısayollar)

```bash
# ===== AI =====
make ai.test            # pytest
make ai.lint            # ruff check
make ai.week1           # Week 1 train

# ===== Crypto =====
make crypto.health      # RPC health check
make crypto.capture     # Basic capture (1500 blok)
make crypto.capture.idem  # 🔥 Idempotent + state tracking
make crypto.report W=0x...  # CLI pretty report
make crypto.report.json W=0x...  # JSON report
make crypto.api         # FastAPI service (uvicorn)

# ===== Shortcuts (aliases) =====
make c.health           # = crypto.health
make c.capture.idem     # = crypto.capture.idem
make c.api              # = crypto.api

# ===== Quality & CI =====
make docs.check         # Markdown link validation
make py.ci              # Ruff + pytest
make report.schema W=0x...  # JSON schema validation

# Tüm komutlar:
make help
```

---

## 📋 Günlük / Haftalık Ritim

### Günlük (2-3 saat) — AI + Crypto Paralel

```
Sabah (60-90 dk): AI Hattı
  1. Teori notunu oku (15-30 dk)
  2. Kod/Deney yap (AI feature/model)
  3. Metrikleri kaydet (exp_log.csv)

Öğlen (45-60 dk): Crypto Hattı
  4. Event capture kontrol / yeni özellik
  5. API/Servis geliştir
  6. Test + validation

Akşam (15 dk): Kapanış
  7. Log + Özet (exp_log.csv + report.md)
  8. Git commit (her iki hat)
```

**Kural:** _"Bugün 1 AI deneyi + 1 Crypto özelliği koşmadıysan, öğrenmedin."_

### Haftalık

```
Pazartesi:      Hedef & Plan (AI + Crypto)
Salı-Perşembe:  Paralel deneyler
                  • AI: Ablation + sweep
                  • Crypto: Endpoint + test
Cuma:           Rapor & Demo
                  • AI metrik grafiği
                  • Crypto API demo (curl/Postman)
```

### Örnek Günlük Plan (W1)

```
Sabah (AI):
  □ linreg_manual.py → val MSE < 0.5 ✓
  □ LR sweep {1e-2,5e-3,1e-3} ✓
  □ loss_curve.png kaydet ✓

Öğlen (Crypto):
  □ capture_transfers_idempotent test ✓
  □ /wallet/{addr}/report endpoint ✓
  □ JSON schema validate ✓

Akşam:
  □ exp_log.csv güncelle ✓
  □ git commit -m "W1D2: AI MSE=0.42, Crypto /report OK" ✓
```

---

## 🗂️ Repo Yapısı

```
novadev-protocol/
├── docs/                     📚 Dökümantasyon (2899 satır)
│   ├── program_overview.md  ⭐⭐⭐ TAM SYLLABUS (önce oku!)
│   ├── overview.md          AI hattı detayı
│   ├── crypto_overview.md   Crypto hattı detayı
│   └── week0_kapanis.md     Self-assessment
│
├── schemas/                  🔐 API Contracts
│   └── report_v1.json       WalletReportV1 JSON schema
│
├── .github/workflows/        🤖 CI/CD
│   ├── docs-link-check.yml  Markdown validation
│   └── python-ci.yml        Ruff + pytest
│
├── week0_setup/              ✅ AI Teori (7061 satır)
│   ├── README.md
│   ├── theory_intro.md      (Lise seviyesi)
│   ├── theory_core_concepts.md (Üniversite)
│   ├── theory_foundations.md (Sezgisel)
│   ├── theory_mathematical.md (Matematik)
│   ├── theory_mathematical_part2.md
│   ├── theory_advanced.md   (Pratik & Saha)
│   └── theory_closure.md    (Self-assessment)
│
├── week1_tensors/            👉 AI W1: Linear Regression
│   ├── README.md            (45 dk sprint + 5 gün plan)
│   ├── data_synth.py
│   ├── linreg_manual.py
│   ├── linreg_module.py
│   └── train.py
│
├── week2_mlp/                AI W2: MLP + MNIST
├── week3_nlp/                AI W3: BERT fine-tune
├── week4_rag/                AI W4: RAG pipeline
├── week5_agent/              AI W5: Tool-agent
├── week6_lora/               AI W6: LoRA fine-tune
├── week7_service/            AI W7: FastAPI + Docker
├── week8_capstone/           AI+Crypto W8: Demo
│
├── crypto/                   🪙 On-Chain Intelligence
│   ├── README.md            ⭐ Crypto roadmap & setup
│   │
│   ├── w0_bootstrap/        ✅ Week 0: RPC + Ingest + Report
│   │   ├── README.md        Quick start (30-45 dk)
│   │   ├── .env.example     Config template
│   │   ├── rpc_health.py    RPC check
│   │   ├── capture_transfers.py  Basic ingest
│   │   ├── capture_transfers_idempotent.py  🔥 Production ingest
│   │   ├── report_v0.py     CLI report (pretty)
│   │   ├── report_json.py   JSON report (API-ready)
│   │   └── validate_report.py  Schema validator
│   │
│   ├── service/             🚀 FastAPI Service
│   │   └── app.py          /healthz, /wallet/{addr}/report
│   │
│   ├── w1_ingest/           (Coming: Collector loop)
│   ├── w2_alerts/           (Coming: Telegram bot)
│   ├── w3_classifier/       (Coming: Event classifier)
│   └── ...                  (W4-W8)
│
├── common/                   Shared utils
├── tests/                    Haftalık testler
├── outputs/                  Metrikler & grafikler
├── reports/                  Haftalık raporlar
│
├── pyproject.toml            Dependencies ([dev], [crypto], etc.)
├── Makefile                  🔧 Command shortcuts (15+ targets)
├── CHANGELOG.md              📋 Version history
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