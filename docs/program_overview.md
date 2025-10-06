# NovaDev — AI + Kripto Paralel Kurs Programı

**"Temeller + Pratik: 8 Haftada İki Sistem"**

> Bu bir AI kursu veya trading kursu DEĞİL; teori + pratik + ürün birlikte ilerleyen **yaparak öğrenme protokolü**. İki paralel hat: **AI (ML temelleri)** + **Crypto (on-chain istihbarat)**.

**⚠️ Yasal Uyarı:** Crypto hattı bilgilendirme amaçlıdır, finansal tavsiye değildir. Read-only, testnet-first, no custody.

---

## 🎯 Amaç ve Çıktılar

### Temel Hedef
8 haftada:
1. **ML zihni** kur → Teori + sezgi + pratik deneyim
2. **Çalışan iki sistem** yap → AI servis + On-chain Intel Copilot
3. **Portfolio** oluştur → GitHub + demo + rapor

### Program Sonunda Elinizde Olacak

#### AI Hattı
```
✓ Mini-modeller (Linear regression, MLP, BERT fine-tune)
✓ RAG (dokümandan kaynaklı yanıt)
✓ Tool-Agent (araç çağırma akışı)
✓ LoRA (opsiyonel domain adaptasyonu)
```

#### Crypto Hattı
```
✓ On-chain veri toplayıcı (EVM, read-only)
✓ DuckDB depolama + cüzdan raporu
✓ Telegram uyarı botu (eşik + etiketleme)
✓ Protokol RAG (kaynaklı açıklama)
✓ Simülasyon araçları (quote, risk check)
```

#### Servis (İki Hat Entegre)
```
✓ FastAPI endpoints:
  - /wallet/<addr>/report (crypto)
  - /alerts (crypto)
  - /chat (AI)
  - /rag (AI + crypto)
  - /simulate (crypto)
  
✓ Docker Compose deployment
✓ Basic monitoring (p95 latency, error rate)
```

#### Capstone
```
✓ 5 dakikalık demo video
✓ README (≤10 dk kurulum)
✓ Retrospektif rapor
```

### Başarı Kriteri
> **"Aynısını yarın tek başına kurabilir misin?" → EVET**

---

## 👥 Kime Göre? Ön Koşullar

### Hedef Kitle
- ✅ **Python bilen** (pandas/CLI rahat)
- ✅ **ML/AI'ye sistemli** girmek isteyenler
- ✅ **On-chain ekosistem** meraklıları
- ✅ **Kod ezberi değil**, neden çalıştığını anlamak isteyenler

### Donanım
```
Önerilen:
  - Apple Silicon (M1/M2/M3) + MPS
  - NVIDIA GPU + CUDA (alternatif)
  - CPU fallback (yavaş ama çalışır)

Kripto için:
  - İnternet bağlantısı (RPC)
  - Testnet faucet erişimi
```

### Zaman Taahhüdü
```
Günlük:   2-3 saat (esnek bölüm)
Haftalık: 5 gün
Toplam:   80-100 saat (8 hafta)
```

### Yazılım
```
Temel:
  - Python 3.11+
  - Git
  - Docker (Week 7+)

AI Stack:
  - PyTorch, scikit-learn
  - transformers (Week 3+)
  - sentence-transformers (Week 4+)
  - FastAPI (Week 7+)

Crypto Stack:
  - web3.py / requests
  - DuckDB
  - python-telegram-bot (Week 2+)
  - (Opsiyonel) Ollama
```

---

## 🧱 Program Ritmi: T → P → X

Her gün üç fazdan geçersiniz:

### 1️⃣ Temel (T) — Kavram + Sezgi
```
Süre:   15-30 dk
Format: Teori notları, kısa okuma
Amaç:   "Neden?" sorularına cevap
```

**Neler Öğreniliyor:**
- Loss fonksiyonlarının kökenleri (MSE → Gaussian MLE)
- Optimizasyon matematiği (GD, momentum, Adam)
- Overfit/underfit dinamikleri
- On-chain event anatomy (block, tx, log, topic)

### 2️⃣ Pratik (P) — Kod & Deney
```
Süre:   60-90 dk
Format: Python scripts, Jupyter
Amaç:   Küçük ama ölçülebilir koşular
```

**Neler Yapılıyor:**
- Manuel gradient descent
- Ablation studies (bir değişken prensibi)
- Hyperparameter sweeps
- Event capture + database insert
- Loss curve / alert log analysis

### 3️⃣ Ürün (X) — Servis & İzleme
```
Süre:   30-60 dk
Format: API endpoints, CLI tools, Telegram
Amaç:   Çıktıyı kullanıcıya ulaştır
```

**Neler Kurulyor:**
- REST API endpoints
- Telegram bot commands
- Health checks
- Basic monitoring

### Günlük Döngü
```
1. Hedef belirle (1 cümle)
   "Val MSE < 0.4 yap"

2. Teori oku (15-30 dk)
   İlgili theory bölümü

3. Kod/Deney (60-90 dk)
   Script'leri koş, metrikleri kaydet

4. Ürüne bağla (30-60 dk)
   API/CLI/Telegram entegre et

5. Log + Özet (10-15 dk)
   exp_log.csv + daily_log.md
   
6. Git commit
   "dayX: hedef ✓"
```

---

## 📏 Değerlendirme ve DoD (Definition of Done)

### Her Hafta Gerekli (Gating Criteria)

```
1. METRİK EŞİĞİ (Geçilmeli!)
   □ Week 1 AI:     Val MSE < 0.5
   □ Week 1 Crypto: /report JSON working
   □ Week 2 AI:     Test acc ≥ 0.97
   □ Week 2 Crypto: 2+ meaningful alerts
   ... (her hafta detayda)

2. ARTIFACT (Kanıt)
   □ Grafik (loss curve, confusion matrix)
   □ Log (exp_log.csv, alert_log.json)
   □ Rapor (weekX_report.md)

3. ÖZET (3-5 Madde)
   □ Ne çalıştı?
   □ Neden? (teori bağlantısı)
   □ Bir dahaki sefere?
```

**Altın Kural:** Eşiği geçmeden sonraki haftaya geçme! (Borç büyütme)

### Örnek Rubrik (Self-Assessment)
```
40% Haftalık Lab'ler (DoD + temiz loglar)
30% Capstone (demo + rapor + kurulum)
20% Kod kalitesi (pytest, ruff, clean)
10% Teknik anlatım (README, rapor)
```

---

## 🗺️ 8 Haftalık Plan (AI + Crypto Paralel)

Her hafta için: **Hedefler**, **Teori (T)**, **Pratik (P)**, **Ürün (X)**, **DoD/KPI**

---

### Week 0 — Temel Zihin ve Bootstrap ✅ TAMAMLANDI

#### AI Hattı

**Teori (T):**
- Veri ayrımı (train/val/test) neden kritik?
- Loss seçimi (MSE/CE/MAE)
- Learning rate semptomları
- Overfit/underfit teşhisi
- Tensor & autograd sezgisi

**Pratik (P):**
- MPS/CPU doğrulama
- Minik matmul benchmark
- pytest + ruff setup
- `week0_kapanis.md` (self-assessment)

**DoD/KPI:**
```
✅ 7061 satır teori notları
✅ Self-assessment complete
✅ MPS functional test
✅ pytest & ruff yeşil
```

#### Crypto Hattı

**Teori (T):**
- EVM anatomy (block, tx, log, topic)
- RPC nedir? Event imzaları (topic0)
- Custodian vs self-custody
- **Read-only ilkesi** (no private keys!)
- .env hijyeni

**Pratik (P):**
- RPC health check
- ERC-20 `Transfer` event ingest → DuckDB
- `report_v0.py` (24h inbound/outbound)

**DoD/KPI:**
```
✅ RPC health < 300ms
✅ Event capture rate ≥ 99%
✅ Wallet report v0 (JSON skeleton)
```

---

### Week 1 — Lineer Regresyon & On-Chain Veri Katmanı

#### AI Hattı

**Teori (T):**
- Lineer regresyon geometrisi (projeksiyon)
- MSE'nin anlamı (Gaussian MLE)
- Ölçekleme etkisi (condition number)
- Early stopping vs L2 regularization

**Pratik (P):**
- Sentetik veri oluşturma
- Manuel gradient descent
- `nn.Module` ile training
- Train/val split + loss curve

**Ürün (X):**
- CLI: `python train.py --lr 0.01 --l2 0.001`
- Grafik: `outputs/loss_curve.png`

**DoD/KPI:**
```
🎯 Val MSE < 0.5
□ loss_curve.png (train + val)
□ exp_log.csv (5+ deney)
□ week1_report.md
□ Overfit örneği (L2=0 koşusu)
□ pytest yeşil
```

#### Crypto Hattı

**Teori (T):**
- Log → tablo eşleme
- Token decimals, wei/gwei/ether
- Zaman pencereleri (24h, 7d)
- Temel istatistik (moving average, volatility)

**Pratik (P):**
- Transfer + Swap (DEX) event schema
- Price feeds (CoinGecko API)
- Cüzdan raporu JSON prototipi

**Ürün (X):**
- CLI: `python report.py --wallet 0x... --hours 24 --json`
- JSON response yapısı

**DoD/KPI:**
```
🎯 /wallet/<addr>/report JSON working
□ Transfer + Swap tables populated
□ Price cache (5 min TTL)
□ Report: 24h net flow (USD)
□ Token breakdown
□ Top 3 counterparties
```

---

### Week 2 — MLP (MNIST) & Telegram Uyarı v0

#### AI Hattı

**Teori (T):**
- Aktivasyonlar (ReLU/GELU) neden gerekli?
- Optimizer seçimi (AdamW vs SGD+Momentum)
- LR schedule (ReduceLROnPlateau, Cosine decay)
- Early stopping stratejisi

**Pratik (P):**
- MLP 2 katman (MNIST)
- DataLoader + minibatch
- Cross-entropy loss
- Confusion matrix

**Ürün (X):**
- CLI: `python train.py --epochs 20 --early-stop`
- API: `/predict` endpoint (skeleton)

**DoD/KPI:**
```
🎯 Test accuracy ≥ 0.97
□ Confusion matrix PNG
□ 10+ yanlış örnek analizi
□ Early stopping logları
□ LR schedule grafiği
□ Ablation: scaler var/yok
```

#### Crypto Hattı

**Teori (T):**
- Eşik tabanlı uyarı (threshold alerts)
- Yanlış alarm vs kaçırma (precision/recall)
- Alert deduplication (1h window)
- "Neden?" açıklama şablonları

**Pratik (P):**
- Telegram bot v0 setup (BotFather)
- Volume spike detection (Z-score > 2)
- Large transfer alert (> $10k)
- Alert message template

**Ürün (X):**
- Telegram: `/start`, `/subscribe <addr>`
- Alert message: "🚨 Balina hareketi! 1.2M USDT çıktı. Neden: ..."

**DoD/KPI:**
```
🎯 2+ meaningful alerts sent
□ Telegram bot running
□ Alert log (timestamp, rule, wallet)
□ Dedup working (no spam)
□ False positive rate < 10%
□ Alert includes "Neden?" text
```

---

### Week 3 — NLP Fine-Tune (TR) & Olay Sınıflayıcı

#### AI Hattı

**Teori (T):**
- Tokenization (BPE/WordPiece)
- Fine-tuning küçük veri ile
- Data leakage risks (val/test contamination)
- F1 score & error analysis

**Pratik (P):**
- Türkçe sentiment classification
- DistilBERT / dbmdz/bert-base-turkish-cased
- Train/val split (stratified)
- Error categorization

**Ürün (X):**
- CLI: `python predict.py --text "..."`
- API: `/classify` endpoint

**DoD/KPI:**
```
🎯 F1 ≥ 0.85
□ Confusion matrix
□ Error analysis table (FP/FN categories)
□ Precision/Recall/F1 per class
□ 20+ test examples
□ Tokenization pipeline documented
```

#### Crypto Hattı

**Teori (T):**
- Event taxonomy: Swap/Mint/Burn/Bridge/Airdrop/Internal
- Heuristic vs ML classification
- Türkçe özet şablonları
- Context extraction (gas, timestamp, token info)

**Pratik (P):**
- Event classifier (heuristic/ML hybrid)
- Türkçe 1-2 cümle özet
- Integration: alerts show event class

**Ürün (X):**
- Alert: "🔄 DEX Swap: 10 ETH → 18.5k USDT (Uniswap V3)"
- Classification confidence > 80%

**DoD/KPI:**
```
🎯 Classification F1 ≥ 0.80
□ Test set: 20+ events
□ Confusion matrix
□ Summary examples (5+)
□ Integration: alert messages show class + summary
□ Türkçe özet quality check
```

---

### Week 4 — RAG (Kaynaklı Yanıt) & Protokol RAG

#### AI Hattı

**Teori (T):**
- Chunking strategies (500-800 token)
- Embedding seçimi (bge-small, MiniLM)
- Top-k recall metrics
- Prompt templates (system, user, context)
- Kanıt gösterimi (citation)

**Pratik (P):**
- Document corpus (100+ pages)
- FAISS/Chroma index
- RAG pipeline (query → retrieve → generate)
- Top-k retrieval

**Ürün (X):**
- CLI: `python rag_query.py "soru"`
- Response: paragraph + source links

**DoD/KPI:**
```
🎯 Recall@3 ≥ 60%
□ Document corpus indexed (500+ chunks)
□ FAISS index working
□ RAG query response time < 2s
□ 95% responses have ≥1 source link
□ Example Q&A pairs (10+)
```

#### Crypto Hattı

**Teori (T):**
- Protokol docs (Uniswap/Curve/Chainlink)
- Risk sözlüğü (front-run, MEV, rug)
- Document chunking for crypto context
- Prompt engineering for sourced explanations

**Pratik (P):**
- Index protocol documentation
- RAG query for alert context
- Telegram: [Kaynakla Açıkla] button

**Ürün (X):**
- Alert + inline button
- Callback: RAG query → 2-3 paragraphs + links
- Format: "Bu swap front-run olabilir çünkü... (Kaynak: Flashbots docs, Section 3.2)"

**DoD/KPI:**
```
🎯 Sourced responses ≥ 95%
□ Protocol docs indexed (Uniswap, Curve, Chainlink)
□ RAG recall@3 ≥ 60%
□ [Açıkla] button working
□ Response includes ≥1 doc URL + section
□ Examples (5+ alerts with sourced explanations)
```

---

### Week 5 — Tool-Agent & Simülasyon

#### AI Hattı

**Teori (T):**
- Function calling / tool use
- Agent loop: plan → call → observe → respond
- Tool schema definition (JSON)
- Error handling & retry logic

**Pratik (P):**
- 2 araçlı mini-agent:
  - `search(query)` (mock/stub)
  - `calculate(expression)` (eval/sympy)
- Agent conversation log
- Multi-step chains

**Ürün (X):**
- CLI: `python agent.py "calculate 15% of 1250 then search for average salary"`
- Log: plan/tool/observe/respond

**DoD/KPI:**
```
🎯 2-step tool chain success
□ Tool definitions (JSON schema)
□ Agent loop implemented
□ Example: search → calculate → summary
□ Logs: clear plan/tool/observe/respond
□ Error handling tested
```

#### Crypto Hattı

**Teori (T):**
- Quote simulation (0x/1inch API)
- Gas estimation (eth_estimateGas)
- Slippage guard heuristics
- Rug check (Honeypot.is API / heuristic)
- Risk checklist (no custody!)

**Pratik (P):**
- Tool definitions:
  - `get_quote(token_in, token_out, amount)`
  - `estimate_gas(tx_data)`
  - `check_slippage(quote, max_slippage)`
  - `rug_check(token_addr)`
- Agent: "User wants swap X→Y" → check → simulate → report

**Ürün (X):**
- CLI: `python simulate.py --swap "ETH->USDT" --amount 1.0`
- Response: quote + gas + slippage + risk score + recommendation
- Format: "Simülasyon: 1 ETH → 2,450 USDT. Gas: ~$5. Slippage: 0.5%. Risk: Düşük. Öneri: ..."

**DoD/KPI:**
```
🎯 Tool chain success ≥ 95%, latency < 2s
□ 4 tools implemented (quote, gas, slippage, rug)
□ Agent: 2-step chain working
□ Example: quote → rug_check → summary
□ Logs: plan/tool/observe/respond
□ Paper trading mode (no real tx)
```

---

### Week 6 — LoRA Mini-Tune & Üslup Uyarlama

#### AI Hattı

**Teori (T):**
- PEFT (Parameter-Efficient Fine-Tuning)
- LoRA (Low-Rank Adaptation) mantığı
- QLoRA farkları (quantization)
- Domain adaptation vs task adaptation
- Evaluation (qualitative + perplexity)

**Pratik (P):**
- 7B model (Llama 3.2 / Qwen 2.5)
- LoRA config (r=8, alpha=16)
- Custom corpus (kendi notların)
- Before/after comparison

**Ürün (X):**
- CLI: `python generate.py --prompt "..." --use-lora`
- A/B test: generic vs LoRA summary

**DoD/KPI:**
```
🎯 A/B LoRA preferred ≥ 60%
□ LoRA checkpoint saved
□ 5+ before/after pairs
□ Blind evaluation
□ Perplexity improvement
□ Integration test (bot uses LoRA)
```

#### Crypto Hattı

**Teori (T):**
- Alert üslup standardizasyonu
- Hallucination bastırma (kanıt zorunlu)
- Tutarlılık metrikleri
- Custom corpus (crypto notları, analizler)

**Pratik (P):**
- LoRA on crypto corpus (100+ paragraphs)
- Style transfer: generic → branded
- Quality check: consistency + citation

**Ürün (X):**
- Alert example:
  - Generic: "Large transfer detected"
  - LoRA: "Dikkat: Balina hareketi! 1.2M USDT cüzdandan ayrıldı. Geçmiş veri: benzer çıkışlar 48h içinde fiyat düşüşü takip etti. (Kaynak: ...)"

**DoD/KPI:**
```
🎯 Style consistency + citation ≥ 95%
□ LoRA on crypto corpus trained
□ 5+ before/after examples
□ Citation rate ≥ 95%
□ Hallucination check (manual review)
□ Integration: bot uses LoRA for alerts
```

---

### Week 7 — Servisleştir & İzle

#### AI + Crypto Entegre

**Teori (T):**
- FastAPI structure
- Endpoint design (RESTful)
- Rate limiting strategies
- Monitoring basics (p50, p95, p99)
- Error handling & logging

**Pratik (P):**
- API endpoints:
  - `/healthz` (AI + Crypto)
  - `/chat` (AI)
  - `/wallet/<addr>/report` (Crypto)
  - `/alerts?since=<timestamp>` (Crypto)
  - `/rag/explain?event_id=<id>` (Crypto + AI)
  - `/quote/simulate` (Crypto)
- Docker Compose setup
- Prometheus-style `/metrics`

**Ürün (X):**
- `docker compose up` → all services running
- Postman/curl test suite
- Basic dashboard (optional)

**DoD/KPI:**
```
🎯 API p95 < 2.5s, error rate < 1%
□ All endpoints tested (200 OK)
□ Rate limiting enforced (IP-based)
□ /metrics endpoint (Prometheus format)
□ Docker Compose working
□ Logs: structured JSON
□ Telegram integration active
```

---

### Week 8 — Capstone (E2E Demo)

#### AI × Crypto Entegre

**Teori (T):**
- System integration strategies
- Demo storytelling (problem → solution → impact)
- Documentation best practices
- Retrospective framework

**Pratik (P):**
- End-to-end flow:
  1. Wallet monitoring (read-only)
  2. Event detection + classification
  3. Alert threshold check
  4. RAG explanation (sourced)
  5. Simulation (quote + risk)
  6. Telegram notification
- Video recording (5 min)
- README polish (≤10 min setup)

**Ürün (X):**
- 5 min demo video:
  - 0-1 min: Problem statement
  - 1-2 min: Architecture overview
  - 2-4 min: Live demo (3 scenarios)
  - 4-5 min: Learnings + v2 roadmap
- README: setup steps, screenshots
- REPORT.md: retrospective

**DoD/KPI:**
```
🎯 3 scenario demo + setup < 10 min
□ Video recorded (5 min)
□ 3 scenarios demonstrated:
  1. Large swap detection + RAG explanation
  2. New token airdrop + classification
  3. Whale movement + simulation
□ README tested (fresh install < 10 min)
□ REPORT.md: learnings + next steps
□ Git tag: v1.0-capstone
□ API p95 < 2.5s (production-ready)
```

---

## 🔒 Güvenlik ve Uyum İlkeleri (Non-Negotiable)

### Crypto Hattı İçin Kritik

```
1. READ-ONLY FIRST
   ❌ Private key yok
   ❌ Custody yok
   ❌ Auto-execute yok
   ✅ RPC read-only
   ✅ Testnet (Sepolia)
   ✅ Paper trading / simulation only

2. .ENV YÖNETİMİ
   ✅ .env.example → Git'e GİRER
   ❌ .env → Git'e GİRMEZ (.gitignore)
   ✅ Hassas data: RPC URL, API keys, Telegram token

3. İMZALAMA (İleride, opsiyonel)
   ✅ EIP-712 typed data
   ✅ Hardware wallet (Ledger/Trezor)
   ✅ Confirm every tx
   ✅ Gas limit caps
   ❌ ASLA otomatik signing

4. RATE LIMITING
   ✅ RPC: 10 calls/sec
   ✅ Telegram: 20 msg/min
   ✅ API: 60 req/min (IP-based)

5. LEGAL DISCLAIMER
   ⚠️ Her uyarı mesajında:
   "Bu bilgi amaçlıdır. Yatırım tavsiyesi değildir. DYOR."

6. PRIVACY
   ❌ Log'larda PII yok
   ✅ Wallet adresleri hashlenebilir (optional)
   ✅ Metrikler anonim
```

### Genel Güvenlik
```
- Git: .env, *.key, *.pem ignore
- API keys: environment variables only
- Error messages: no sensitive data leak
- Logging: structured, no secrets
```

---

## 📊 Haftalık Rapor Disiplini

### Dosya Yapısı
```
outputs/
  ├── ai/
  │   ├── loss_curves/
  │   ├── confusion_matrices/
  │   └── exp_log.csv
  │
  └── crypto/
      ├── alert_logs/
      ├── metrics/
      └── event_log.csv

reports/
  ├── week1_report.md
  ├── week2_report.md
  └── ...
```

### Haftalık Rapor Şablonu (weekX_report.md)
```markdown
# Week X Report

## Hedefler (Hafta Başında)
- [ ] AI: Val MSE < 0.5
- [ ] Crypto: /report JSON working

## Deneyler (Günlük Log)
| Tarih | Deney | Sonuç | Gözlem |
|-------|-------|-------|--------|
| 10/7  | LR=0.01, L2=0.001 | MSE=0.48 ✓ | Ölçekleme kritik |
| 10/8  | Capture 10k blocks | 9.8k events | Rate limit hit |

## Sonuçlar (Hafta Sonu)
- ✅ AI: MSE=0.42 (eşik geçildi)
- ✅ Crypto: JSON rapor çalışıyor
- ⚠️ Crypto: Rate limit issue (çözüldü)

## Öğrendiklerim (3-5 Madde)
1. **Ölçekleme kritik:** StandardScaler olmadan MSE 10x daha kötü
2. **Rate limit:** RPC provider limitleri gerçek (throttle ekledim)
3. **Early stopping:** 3 epoch early stop ile overfit önlendi
4. **DuckDB hızlı:** 10k insert < 2s (indexler sayesinde)
5. **Teori bağlantısı:** MSE=Gaussian MLE gerçekten tuttu

## Sıradaki Hafta (Hazırlık)
- Week 2 AI: MLP 2 katman + MNIST
- Week 2 Crypto: Telegram bot v0 + alert engine
- Okuma: Activation functions (ReLU vs GELU)
```

### Günlük Kapanış Şablonu (daily_log.md)
```markdown
## Day X - [Tarih]

### Hedef (Sabah)
Val MSE < 0.4 yap

### Plan (3 Madde)
1. LR sweep (1e-3, 5e-3, 1e-2)
2. Early stopping on
3. L2=1e-3 sabit

### Deneyler
- LR=1e-3: MSE=0.52 (yavaş)
- LR=5e-3: MSE=0.38 ✓ (optimal)
- LR=1e-2: MSE=0.91 (overshot)

### Sonuç
✅ Hedef: MSE=0.38 (eşik geçildi)

### Öğrenme (1-2 Cümle)
LR=5e-3 optimal noktasıydı. 1e-2'de momentum salınım yaptı.

### Yarın İlk 30 Dakika
Scaling ablation: StandardScaler var/yok karşılaştır
```

---

## 🛠️ İlk Hafta Komut Planı (Week 1 Özet)

### AI Week 1 (Komutlar)
```bash
# Day 1: E2E + Logging
python ai/week1_tensors/data_synth.py
python ai/week1_tensors/linreg_manual.py
python ai/week1_tensors/linreg_module.py
pytest tests/test_linreg.py

# Day 2: Early Stopping + LR Schedule
python ai/week1_tensors/train.py --lr 0.01 --early-stop

# Day 3: Scaling Ablation
python ai/week1_tensors/train.py --no-scale  # Baseline
python ai/week1_tensors/train.py --scale     # Compare

# Day 4: Loss Curves + Overfit
python ai/week1_tensors/train.py --l2 0.0    # Overfit example
python ai/week1_tensors/train.py --l2 0.001  # Regularized

# Day 5: Final Report
python scripts/generate_report.py --week 1
```

### Crypto Week 1 (Komutlar)
```bash
# Day 1: Ingest Generalization
python crypto/w1_ingest/capture_swaps.py --blocks 5000

# Day 2: Price Feeds
python crypto/w1_ingest/price_fetcher.py --tokens ETH,USDT,USDC

# Day 3: Report JSON
python crypto/w1_ingest/report.py --wallet 0x... --json

# Day 4: Collector Loop (30s polling)
python crypto/w1_ingest/collector_loop.py --poll-interval 30

# Day 5: Integration Test
python crypto/w1_ingest/test_report_api.py
```

---

## 📚 Bağımsız Kaynaklar (Opsiyonel Okuma)

### ML Temelleri
- "Deep Learning" (Goodfellow, Bengio, Courville) - Ch. 2-5
- "Dive into Deep Learning" (d2l.ai) - Linear regression, MLP
- PyTorch Tutorials (pytorch.org/tutorials)

### NLP
- "Speech and Language Processing" (Jurafsky, Martin) - Ch. 6-7
- Hugging Face Course (huggingface.co/learn/nlp-course)

### Crypto/On-Chain
- Ethereum Yellow Paper (özet bölümler: block, tx, log)
- Uniswap V2/V3 Whitepaper
- Curve Finance Documentation
- Flashbots MEV Resources (flashbots.net)

### Yazılım Mühendisliği
- "Clean Code" (Robert Martin) - Ch. 2-3, 10
- FastAPI Documentation (fastapi.tiangolo.com)
- DuckDB Guide (duckdb.org/docs)

---

## 🎓 Sonuç: Neden Bu Program?

### Farklılıklar
```
Klasik Kurs:
  Teori → (Belki) Pratik → (Nadiren) Proje

NovaDev:
  Teori + Pratik + Ürün (her hafta!)
  
  Week 1 sonu: Çalışan regresyon CLI
  Week 4 sonu: RAG + Telegram bot active
  Week 8 sonu: Full system deployed
```

### Neden Paralel (AI + Crypto)?
```
1. GERÇEK DÜNYA: Sistemler entegre çalışır
   - RAG hem AI hem Crypto'da kullanılır
   - Tool-Agent her iki domain'de uygulanır
   
2. MOTİVASYON: İki farklı problem → daha zengin öğrenme
   - AI: Classification, NLP, embeddings
   - Crypto: Event parsing, time-series, alerts
   
3. PORTFOLIO: İki çalışan sistem → daha güçlü CV
   - "AI ML Engineer" + "Blockchain Data Analyst"
   - Hem model hem pipeline experience
```

### Başarı Formülü
```
T (Teori: Neden?) 
  + 
P (Pratik: Nasıl?) 
  + 
X (Ürün: Kullanılabilir)
  = 
Sürdürülebilir Öğrenme + Portfolio
```

---

## 🚀 İlk Adım (Bugün)

```bash
# 1. Overview'ı oku (bu dosya)
cat docs/program_overview.md

# 2. Week 0'ı tamamla
# AI:
cat week0_setup/theory_closure.md  # Self-assessment

# Crypto:
cd crypto/w0_bootstrap
cat README.md  # Setup guide
python rpc_health.py

# 3. Hepsi ✓ ise Week 1'e geç!
cd week1_tensors
cat README.md  # 45 dk sprint planı
```

---

**NovaDev — "AI + Crypto: Öğrenirken İki Gemi Yap"**

*Versiyon: 1.1 (Paralel Program)*  
*Son Güncelleme: 2025-10-06*  
*Status: Week 0 Complete, Week 1 Ready!*
