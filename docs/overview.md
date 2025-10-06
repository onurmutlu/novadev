# NovaDev — Program Özeti (0→Prod, 8 Hafta)

**"Öğrenirken Gemi Yapan" Üretim Programı**

> Bu bir "AI kursu" değil; teori + pratik + ürün birlikte ilerleyen bir **yaparak öğrenme protokolü**.

---

## 🎯 Amaç (North Star)

### Temel Hedef
**"ML zihni" kur + uçtan uca bir AI servisini çalışır halde yayına al.**

### Çıktı Seti
1. ✅ **Ölçülebilir mini-modeller**
   - Linear regression (MSE < 0.5)
   - MLP + MNIST (accuracy ≥ 0.97)
   - BERT fine-tune (F1 ≥ 0.85)

2. ✅ **RAG + Tool-Agent prototipi**
   - Retrieval-augmented generation
   - Araç kullanan ajan (tool calling)

3. ✅ **FastAPI servis**
   - `/healthz`, `/chat`, `/rag` endpoints
   - Rate limiting + basic metrics
   - Docker Compose deployment

4. ✅ **Capstone demo**
   - 5 dakikalık video
   - Kurulum adımları (≤ 10 dk)
   - README + çalışan kod

### Başarı Kriteri
> **"Aynısını yarın tek başına kurabilir misin?" → EVET**

---

## 👥 Kime Göre?

### Hedef Kitle
- ✅ Python bilen
- ✅ ML/AI'ye **sistemli** girmek isteyen geliştirici
- ✅ Kod ezberi değil, **neden çalıştığını** anlamak isteyen

### Donanım
- **Apple Silicon (M1/M2/M3)**
- MPS (Metal Performance Shaders) ile rahat ilerliyoruz
- CPU fallback mevcut

### Zaman Taahhüdü
```
Günlük:  2-3 saat
Haftalık: 5 gün
Toplam: ~80-100 saat (8 hafta)
```

---

## 🧱 Program Mimarisi (3 Hat Birlikte)

### 1️⃣ Temel (T) — Kavram + Sezgi
```
Süre: Haftalık 45-60 dk
Format: Teori notları (theory_*.md)
Amaç: "Neden?" sorularına cevap
```

**Neler Öğreniliyor:**
- Loss fonksiyonlarının probabilistik kökenleri
- Optimizasyon matematiği
- Overfit/underfit dinamikleri
- Regularization teorisi

### 2️⃣ Pratik (P) — Kod & Deney
```
Süre: Günlük 60-90 dk
Format: Python scripts + Jupyter notebooks
Amaç: Küçük ama ölçülebilir koşular
```

**Neler Yapılıyor:**
- Manuel gradient descent
- Ablation studies
- Hyperparameter sweeps
- Loss curve analysis

### 3️⃣ Ürün (X) — Servis & İzleme
```
Süre: Haftalık 60-90 dk
Format: API endpoints + Docker
Amaç: Çıktıyı kullanıcıya ulaştır
```

**Neler Kurulyor:**
- REST API endpoints
- Health checks
- Basic monitoring
- Deployment pipeline

### T → P → X Döngüsü
```
┌─────────────────────────────────────┐
│ TEMEL (T)                           │
│ "Neden MSE?"                        │
│ → Gaussian MLE bağlantısı           │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ PRATİK (P)                          │
│ MSE kodla, L2 ekle                  │
│ → Val MSE < 0.5 yap                 │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ ÜRÜN (X)                            │
│ /predict endpoint kur               │
│ → Docker'da çalıştır                │
└─────────────────────────────────────┘
```

---

## ⏱ Akış (Gün / Hafta Ritmi)

### Günlük Ritim (2-3 saat)

#### Sabah (30 dk)
```
1. Hedef belirle (1 cümle)
   "Val MSE < 0.4 yapmak"

2. Plan yap (3 madde)
   - LR sweep (1e-3, 5e-3, 1e-2)
   - Early stopping on
   - L2=1e-3 sabit

3. Teori notunu oku (15-30 dk)
   İlgili theory bölümünü hızlı tara
```

#### Öğlen (90 dk)
```
4. Kod/Deney yap (60-90 dk)
   - Script'leri koş
   - Metrikleri kaydet
   - Grafikleri oluştur

5. Gözlem yap
   - Ne işe yaradı?
   - Neden?
```

#### Akşam (15 dk)
```
6. Log + Özet (10-15 dk)
   - exp_log.csv'ye yaz
   - daily_log.md'ye not düş
   - Git commit

7. Yarın için hazırlık (5 dk)
   - Sonraki hedefi belirle
```

### Haftalık Ritim

```
┌────────────────────────────────────────┐
│ PAZARTESİ: Hedef & Plan               │
├────────────────────────────────────────┤
│ - Haftalık metrik eşiğini belirle     │
│ - 5 günlük plan yap                    │
│ - Teori notlarını tara                 │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ SALI-ÇARŞAMBA-PERŞEMBE: Deneyler      │
├────────────────────────────────────────┤
│ - Baseline kur                         │
│ - Ablation studies yap                 │
│ - Hyperparameter sweep                 │
│ - Her gün: commit + log                │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ CUMA: Rapor & Demo                     │
├────────────────────────────────────────┤
│ - Haftalık rapor yaz (1 sayfa)        │
│ - Kod temizliği (lint, test)          │
│ - Küçük demo (CLI/HTTP)                │
│ - Git tag (week-X-complete)            │
└────────────────────────────────────────┘
```

---

## 🗺 Syllabus & Kilometre Taşları

### Week 0 — Temel Zihin ✅ TAMAMLANDI

**Konular:**
- Veri ayrımı (train/val/test)
- Loss fonksiyonları (MSE, CE)
- Learning rate davranışı
- Overfit/underfit teşhisi
- Tensor & autograd

**Çıktılar:**
- ✅ 7061 satır teori notları (7 döküman)
- ✅ Week 0 kapanış dökümanı (self-assessment)
- ✅ MPS test geçti
- ✅ pytest/ruff yeşil

**Definition of Done:**
```
□ theory_closure.md tamamlandı
□ Self-check listesi ✓
□ MPS functional test geçti
□ Setup verified
```

---

### Week 1 — Linear Regression (Konveks Kampı)

**Konular:**
- Neden MSE? (Gaussian MLE)
- Ölçekleme etkisi (condition number)
- Early stopping vs L2
- LR davranışı (semptomlar)

**Pratik:**
- Sentetik veri oluşturma
- Manuel gradient descent
- nn.Module ile training
- Train/val split + early stopping

**Metrik Eşiği:** 🎯 **Val MSE < 0.5**

**Çıktılar:**
- Loss curve grafiği (train + val)
- exp_log.csv (5+ deney)
- week1_report.md (teori bağlantılı)
- Overfit örneği (L2=0 koşusu)

**Definition of Done:**
```
□ Val MSE < 0.5 ✓
□ Loss curves (normal + overfit)
□ Ablation: scaling var/yok
□ Week 0 teori bağlantısı açıklandı
□ pytest yeşil
```

---

### Week 2 — MLP + MNIST

**Konular:**
- Non-linearity neden gerekli?
- Multi-layer backpropagation
- DataLoader & batching
- Classification metrics (accuracy, precision, recall)

**Pratik:**
- MNIST dataset
- 2-layer MLP
- ReLU/GELU activation
- Cross-entropy loss
- Confusion matrix

**Metrik Eşiği:** 🎯 **Test Accuracy ≥ 0.97**

**Çıktılar:**
- Accuracy curve
- Confusion matrix
- Error analysis (yanlış sınıflanan örnekler)
- week2_report.md

**Definition of Done:**
```
□ Test acc ≥ 0.97 ✓
□ Confusion matrix görselleştirildi
□ 10+ yanlış örnek analizi
□ Overfit kontrolü (early stopping logları)
```

---

### Week 3 — NLP Temeli (Türkçe BERT)

**Konular:**
- Tokenization (BPE/WordPiece)
- Pre-trained models (DistilBERT)
- Fine-tuning küçük veri ile
- F1 score & error analysis

**Pratik:**
- Sentiment analysis (Türkçe)
- dbmdz/bert-base-turkish-cased
- Train/val split (küçük dataset)
- Error categorization

**Metrik Eşiği:** 🎯 **F1 ≥ 0.85**

**Çıktılar:**
- F1, precision, recall scores
- Confusion matrix
- Error analysis table (false positives/negatives)
- week3_report.md

**Definition of Done:**
```
□ F1 ≥ 0.85 ✓
□ Confusion matrix + analiz
□ Error categorization (3+ kategori)
□ Tokenization pipeline dokümente edildi
```

---

### Week 4 — RAG (Retrieval-Augmented Generation)

**Konular:**
- Chunking strategies (500-800 token)
- Sentence embeddings (bge-small)
- Vector databases (FAISS/Chroma)
- Prompt composition

**Pratik:**
- Kendi notlarını indexle
- FAISS index oluştur
- Top-k retrieval
- LLM ile yanıt oluştur (Ollama)

**Metrik Eşiği:** 🎯 **Top-k Recall ≥ %60**

**Çıktılar:**
- `cli.py "soru"` → kaynaklı yanıt
- Recall@k metrics
- Chunking ablation
- week4_report.md

**Definition of Done:**
```
□ Top-k recall ≥ %60 ✓
□ CLI interface çalışıyor
□ Kaynak atıfları doğru
□ Chunking stratejisi dokümente edildi
```

---

### Week 5 — Tool-Agent (Araç Kullanan Ajan)

**Konular:**
- Function calling
- Tool schema definition
- Agent loop (plan → call → observe → respond)
- Error handling

**Pratik:**
- En az 2 tool: `search()`, `math()`
- Tool calling pipeline
- Agent döngüsü
- Logging (her adım)

**Metrik Eşiği:** 🎯 **2-step tool chain başarılı**

**Çıktılar:**
- Tool definitions (JSON schema)
- Agent conversation logs
- Multi-step example (2+ tools)
- week5_report.md

**Definition of Done:**
```
□ 2-step tool chain çalışıyor ✓
□ Tool calling logları net
□ Error handling test edildi
□ Example dialog dokümente edildi
```

---

### Week 6 — LoRA Fine-tune (7B Model)

**Konular:**
- PEFT (Parameter-Efficient Fine-Tuning)
- LoRA (Low-Rank Adaptation)
- Domain adaptation
- Evaluation (qualitative)

**Pratik:**
- 7B model (Llama/Qwen)
- LoRA configuration (r=8, alpha=16)
- Domain dataset (kendi notların)
- Before/after comparison

**Metrik Eşiği:** 🎯 **Qualitative improvement (A/B blind test)**

**Çıktılar:**
- LoRA checkpoint
- Before/after examples (5+ pairs)
- Blind evaluation results
- week6_report.md

**Definition of Done:**
```
□ LoRA training tamamlandı ✓
□ Before/after örnekleri var (5+)
□ Blind evaluation yapıldı
□ Improvement documented
```

---

### Week 7 — Servisleştir & İzle

**Konular:**
- FastAPI structure
- Endpoint design
- Rate limiting
- Basic monitoring (metrics)
- Docker deployment

**Pratik:**
- `/healthz` endpoint
- `/chat` endpoint (LLM)
- `/rag` endpoint (retrieval)
- Prometheus-style `/metrics`
- Docker Compose

**Metrik Eşiği:** 🎯 **p95 latency < 2.5s**

**Çıktılar:**
- FastAPI service
- Docker Compose config
- Metrics dashboard (basic)
- week7_report.md

**Definition of Done:**
```
□ Docker Compose up çalışıyor ✓
□ 3 endpoint (healthz, chat, rag) test edildi
□ p95 latency < 2.5s
□ Rate limiting aktif
□ /metrics endpoint var
```

---

### Week 8 — Capstone

**Konular:**
- System integration
- Demo preparation
- Documentation
- Retrospective

**Pratik:**
- RAG + Agent + Service entegrasyonu
- End-to-end flow
- Video demo (5 dk)
- README kurulum (≤ 10 dk)

**Metrik Eşiği:** 🎯 **3 soru akışı videosu**

**Çıktılar:**
- 5 dakikalık demo video
- README.md (kurulum adımları)
- REPORT.md (retrospektif)
- "Ne öğrendim / Ne eksik" analizi
- v2 hedefleri

**Definition of Done:**
```
□ 5 dk video demo ✓
□ 3 soru akışı gösterildi
□ README kurulum < 10 dk test edildi
□ Retrospektif rapor tamamlandı
□ v2 roadmap var
```

---

## 📏 Değerlendirme & Geçiş Eşiği (Gating)

### Her Hafta Gerekli (Gating Criteria)

#### 1. Metrik Eşiği
```
Week 1: Val MSE < 0.5
Week 2: Test acc ≥ 0.97
Week 3: F1 ≥ 0.85
Week 4: Recall@k ≥ 60%
Week 5: 2-step tool chain
Week 6: Qualitative improvement
Week 7: p95 < 2.5s
Week 8: 3 soru akışı
```

**Kural:** Eşiği geçmeden sonraki haftaya geçme!

#### 2. Artifact (Kanıt)
```
- Grafik (loss curve, confusion matrix)
- Log (exp_log.csv, metrics.json)
- Rapor (weekX_report.md)
```

#### 3. Özet (3-5 Madde)
```
Ne çalıştı?
Ne çalışmadı?
Neden? (teori bağlantısı)
Bir dahaki sefere?
```

### Geçiş Kuralı
```
❌ Eşik geçilmedi → Aynı haftayı tekrar
   (Borç büyür, Week 8'e ulaşamazsın)

✅ Eşik geçildi → Sonraki haftaya
   (Özgüvenle devam)
```

---

## 🗃 Çıktılar & Portfolio

### Klasör Yapısı
```
novadev-protocol/
├── outputs/               # Metrikler & grafikler
│   ├── loss_curves/
│   ├── confusion_matrices/
│   ├── exp_log.csv
│   └── metrics.json
│
├── reports/               # Haftalık raporlar
│   ├── week1_report.md
│   ├── week2_report.md
│   └── ...
│
├── demo/                  # Capstone
│   ├── capstone_demo.mp4
│   ├── demo.gif
│   └── screenshots/
│
└── docs/                  # Dökümantasyon
    ├── overview.md (bu dosya)
    ├── week0_kapanis.md
    └── architecture.md
```

### Portfolio İçeriği
```
1. GitHub Repo
   - README.md (kurulum < 10 dk)
   - Clean code (lint, test)
   - Commit history (her gün)

2. Metrics & Graphs
   - Loss curves
   - Confusion matrices
   - Recall@k tables

3. Reports
   - 8 haftalık rapor
   - Teori bağlantıları
   - Ablation studies

4. Demo
   - 5 dk video
   - GIF preview
   - Live deployment (opsiyonel)
```

**Portfolyo > Sertifika**
> Bu repo = CV'de gösterebileceğin **KANIT**

---

## 🔧 Araçlar / Stack

### Kod & ML
```
Python 3.11+
PyTorch 2.x (MPS support)
scikit-learn
transformers (Hugging Face)
sentence-transformers
```

### Veri
```
torchvision/datasets (MNIST)
Hugging Face datasets
Kendi notların (RAG için)
```

### Servis & Deployment
```
FastAPI
Uvicorn
Docker & Docker Compose
```

### Arama & Embedding
```
FAISS / Chroma
bge-small (sentence embeddings)
```

### LLM
```
Ollama (lokal 7B/8B)
API fallback (OpenAI/Anthropic)
```

### İzleme (Basic)
```
/metrics endpoint (Prometheus-style)
JSONL logs
```

### Geliştirme
```
pytest (testing)
ruff (linting)
Git (version control)
```

---

## 🧭 Pedagoji (Neden Böyle?)

### 1. Sarmal Öğrenme
```
Her hafta = Önceki hafta + Yeni kavram

Week 1: Linear regression (MSE)
Week 2: MLP (MSE → CE)
Week 3: BERT (CE → F1)
Week 4: RAG (Embeddings)
...

Her adımda önceki bilgi PEKİŞİYOR
```

### 2. Minimum Teori, Ölçülebilir Pratik
```
Teori: Sadece "neden"ler (45-60 dk/hafta)
Pratik: Ölçülebilir metrik (her gün)

❌ "Şöyle yapılır" (ezber)
✅ "Neden böyle yapıyoruz?" (anlayış)
```

### 3. Ablation Disiplini
```
Soru: "L2 gerçekten işe yarıyor mu?"

Cevap: Tahmin değil, KANIT!
  - L2=0 koşusu: Val MSE = 0.45
  - L2=1e-3 koşusu: Val MSE = 0.18
  → L2 işe yarıyor ✓

Her iddia = Deney ile doğrulanmış
```

### 4. Product-First
```
Klasik Kurs:
  Teori → Pratik → (Belki) Proje

NovaDev:
  Teori → Pratik → ÜRÜN (her hafta)
  
Week 1 sonu: Çalışan regresyon scripti
Week 4 sonu: RAG CLI çalışıyor
Week 7 sonu: API deploy edildi

Çalışan uç nokta → Motivasyon ↑
```

### 5. Hata Odaklı Öğrenme
```
Sorun çıkması = Öğrenme fırsatı

NaN gördün mü?
→ LR çok büyük (theory'de vardı!)

Val loss yükseliyor mu?
→ Overfit (theory_closure.md, Bölüm 4)

Hata → Teori → Çözüm → ÖĞRENME
```

---

## 🧯 Tıkandığında Hızlı Teşhis

### 6 Adımlık Checklist

```
1. LR çok mu? (NaN/Inf/salınım)
   → LR'ı yarıya indir

2. Ölçekleme var mı? (StandardScaler)
   → Feature standardization ekle

3. Grad sıfırlama doğru mu?
   → optimizer.zero_grad() kontrol et

4. Loss/Metric uyumu doğru mu?
   → Regresyon: MSE, Sınıflama: CE

5. Shape/Dtype/Device tutarlı mı?
   → print(x.shape, x.dtype, x.device)

6. Seed & Tekrar (sonuç niye zıplıyor?)
   → torch.manual_seed(42) ekle
```

**Referans:** `week0_setup/theory_closure.md`

---

## 📝 Beklentiler (Gerçekçi)

### Zaman
```
Haftalık 5-8 saat  → Tamam ✓
Haftalık 10+ saat  → Şahane ✓✓
Haftalık < 5 saat  → Eşikleri geçemezsin ✗
```

### Kod Kalitesi
```
❌ "Mükemmel" kod (gereksiz)
✅ Ölçülebilir gelişim (gerekli)
✅ Test geçen kod (şart)
✅ Lint temiz kod (şart)
```

### Demo
```
Haftada en az 1 demo:
  - CLI tool
  - HTTP endpoint
  - Jupyter notebook
  - Grafik/visualizasyon
```

### Rapor
```
Haftalık 1 sayfa rapor (ŞART)

Format:
  - Hedef (1 cümle)
  - Sonuç (metrik)
  - Gözlem (3-5 madde)
  - Teori bağlantısı
  - Sıradaki adım
```

---

## ✅ Bugünün Sonraki Adımı

### Week 0 Kapanışı (Bugün)
```
1. docs/overview.md oluşturuldu ✓ (bu dosya)
2. week0_setup/theory_closure.md tamamlandı ✓
3. Self-check yapıldı mı?
   □ Train/Val/Test farkını biliyorum
   □ MSE/MAE/Huber ne zaman kullanılır biliyorum
   □ LR semptomlarını tanıyorum
   □ Overfit/Underfit teşhis edebiliyorum
   □ Week 0'ı ÖZÜMSEDIM ✓
```

### Week 1 Başlangıcı (Yarın)
```
1. week1_tensors/README.md'yi oku (5 dk)
2. 45 dakikalık hızlı sprint'i koş
   - Data synth
   - Manuel GD
   - nn.Module
   - Test
   - Mini rapor
3. Hedef: Val MSE < 0.5
4. Sonuç: week1_summary.md + commit
```

---

## ❓ SSS (Kısa)

### Bu bir kurs mu?
```
❌ Kurs değil
✅ Yaparak öğrenme protokolü

Her hafta:
  - Ölçülebilir hedef
  - Gerçek çıktı
  - Kanıt (artifact)
```

### Sertifika var mı?
```
Portfolyo > Sertifika

Capstone + Repo = KANIT
GitHub profil = CV'de link
"İşte yaptığım sistem" → Interview'da göster
```

### "Az vaktim var" modu?
```
Haftalık metrik eşiğini KORU
Kapsamı küçült:
  - Week 3: Daha küçük dataset
  - Week 6: Daha küçük model (3B)
  - Week 7: Temel endpoint'ler (monitoring skip)

Eşik geçilmeli, süre esnek olabilir
```

### Takıldım, ne yapmalıyım?
```
1. theory_closure.md'deki checklist (6 adım)
2. Sorununu daily_log.md'ye yaz
3. Ablation yap (ne zaman başladı?)
4. Git history kontrol et (son değişiklik ne?)
5. Geri dön önceki çalışan commit'e
```

### Daha hızlı ilerleyebilir miyim?
```
✅ Metrik eşiğini geçtiysen → Evet
✅ Artifact'ları oluşturduysan → Evet
✅ Raporu yazdıysan → Evet

→ Sonraki haftaya geç
(Bazı haftalar 3-4 günde biter)
```

### Week 8'den sonra ne olacak?
```
v2 Roadmap:
  - Daha büyük modeller (13B/70B)
  - Production deployment (AWS/GCP)
  - A/B testing framework
  - Monitoring dashboard (Grafana)
  - CI/CD pipeline
  - Load testing
```

---

## 🎓 Sonuç

**NovaDev = "Öğrenirken Gemi Yapan" Program**

```
8 Hafta
80-100 Saat
8 Çalışan Sistem
1 Capstone Demo

= Portfolyo + Özgüven + ML Zihni
```

**Başarı Formülü:**
```
Teori (Neden?) + Pratik (Nasıl?) + Ürün (Kullanılabilir)
= Sürdürülebilir Öğrenme
```

**İlk Adım:**
```bash
cd /Users/onur/code/novadev-protocol
source .venv/bin/activate

# Week 0'ı tamamladın mı?
cat week0_setup/theory_closure.md

# Hepsi ✓ ise:
cd week1_tensors
cat README.md

# Hadi başla! 🚀
python data_synth.py
```

---

**NovaDev Protocol Başlıyor! 💪**

*Son Güncelleme: 2025-10-06*
*Versiyon: 1.0*
