# 🎓 NovaDev v1.0 — 8 Haftalık AI Rotası

**"AI öğren, gemiyi kendin yap."**

M3 Mac uyumlu, günlük 2–3 saat, haftada 5 gün. Her hafta: mini teori + kod + teslim.

---

## 📋 Haftalık Roadmap

| Hafta | Konu | Teslim | KPI |
|-------|------|--------|-----|
| **0** | Kurulum & Kas Isıtma | `nova-setup.md` + `hello-tensor.py` | MPS çalışmalı |
| **1** | Tensör, Autograd, Regresyon | `linreg_from_scratch.ipynb` + testler | MSE < X |
| **2** | MLP & Sınıflandırma | `mlp_classifier.py` + train/inference | MNIST ≥ %90 |
| **3** | NLP Temelleri | `nlp_sentiment/` | F1 ≥ 0.85 |
| **4** | RAG | `rag_basic/` CLI | Top-k recall ≥ %60 |
| **5** | Araç Kullanan Ajan | `agent_tools/` | 2+ tool chain |
| **6** | Fine-tune (LoRA) | `finetune_lora/` | A/B eval raporu |
| **7** | Servis & İzleme | `service/` + Docker | p95 < 2.5s |
| **8** | Capstone | Demo + `REPORT.md` | 5 dk video |

---

## 🗂️ Repo Yapısı

```
novadev-protocol/
├── week0_setup/          # Kurulum doğrulama
├── week1_tensors/        # Linear regression, autograd
├── week2_mlp/            # MLP sınıflandırıcı
├── week3_nlp/            # NLP sentiment analysis
├── week4_rag/            # RAG pipeline
├── week5_agent/          # Tool-calling agent
├── week6_lora/           # LoRA fine-tuning
├── week7_service/        # FastAPI servis
├── week8_capstone/       # Uçtan uca demo
├── common/               # Paylaşılan utils
├── tests/                # Haftalık testler
├── pyproject.toml        # Dependencies & tools
└── README.md             # Bu dosya
```

---

## 🚀 Kurulum (Hafta 0)

```bash
# Python 3.11+ kontrol
python3 --version

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Temel paketler
pip install --upgrade pip
pip install -e ".[dev]"

# Ollama kur (opsiyonel, lokal LLM için)
brew install ollama
ollama pull qwen2.5:7b  # veya llama3.2:7b

# MPS (Metal) kontrol
python week0_setup/hello_tensor.py
```

---

## 📝 Günlük Ritim

**2–3 saat blok:**
- **00:00–00:45** → Teori/okuma (not çıkar)
- **00:45–02:15** → Kod/pratik (tek hedef)
- **02:15–02:30** → Kapanış logu (ne yaptım/yarın ne)

**Kural:** "Bugün 1 deney koşmadıysan, öğrenmedin."

---

## 📦 Teslim Disiplini

- Her gün **1 commit**: `dayX: <kısa hedef>`
- Her hafta **1 README güncellemesi**: metrik & öğrendiklerim
- Her modülde **1 test** veya **1 ölçülebilir metrik**

---

## 🎯 Başarı Kriterleri (KPI)

- ✅ Hafta 2: MNIST accuracy ≥ %90
- ✅ Hafta 3: Sentiment F1 ≥ 0.85
- ✅ Hafta 4: RAG top-k recall ≥ %60
- ✅ Hafta 7: API p95 latency < 2.5s
- ✅ Hafta 8: 5 dk canlı demo + kurulum < 10 dk

---

## 🔧 M3 Mac Notları

- **MPS (Metal Performance Shaders)** PyTorch backend
- Büyük modeller için **Ollama** (7B/8B lokal)
- LoRA fine-tune: düşük batch + gradient accumulation
- Docker için CPU fallback (MPS container desteği sınırlı)

---

## 📊 İlerleme Takibi

### Hafta 0: Kurulum ✅
- [ ] Python 3.11+ kurulu
- [ ] PyTorch MPS çalışıyor
- [ ] Ollama lokal model çalıştı
- [ ] `hello-tensor.py` başarılı

### Hafta 1: Tensörler & Regresyon
- [ ] Autograd manuel hesap
- [ ] Linear regression sıfırdan
- [ ] Train/val split + L2 reg
- [ ] Test: MSE < threshold

### Hafta 2-8: Devam edecek...

---

## 💡 Bonus Mini-Projeler

- **Anomaly detection**: CSV sensör verisi, z-score/ESD
- **Telegram bot bridge**: Servis `/chat`'e proxy
- **Caption builder**: SeferVerse için prompt aracı

---

## 📚 Kaynaklar

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [HuggingFace Course](https://huggingface.co/learn/nlp-course)
- [Ollama](https://ollama.ai/)
- [PEFT/LoRA Docs](https://huggingface.co/docs/peft)

---

**Son güncelleme:** Hafta 0 (Kurulum)
**Sonraki hedef:** Linear regression from scratch
