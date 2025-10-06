# 🔗 NovaDev Crypto — On-Chain Intel Copilot

**"Okuyan, Anlayan, Uyarı Veren, Simüle Eden"**

> Bu bir trading sinyal kursu DEĞİL; güvenli, read-only, paper-trading odaklı **On-Chain Intelligence** sistemi.

---

## 🎯 Vizyon

### Hedef (North Star)
**8 hafta sonunda elimizde:**
```
On-Chain Intel Copilot
  ├─ Cüzdan/kontrat izleme (read-only)
  ├─ Olay etiketleme + risk kontrolü
  ├─ RAG ile kaynaklı açıklama
  ├─ Telegram özet + uyarı
  └─ Simülasyon (paper trading)
```

### Çıktı Seti
```
1. ✅ Event Collector
   - EVM RPC event listener
   - Off-chain price feeds
   - DuckDB storage

2. ✅ Classification & RAG
   - Event classification (Swap/Mint/Bridge/etc.)
   - Protocol docs RAG (Uniswap/Curve/etc.)
   - Risk scoring

3. ✅ Alert System
   - Telegram bot (özet + kaynak)
   - Threshold-based triggers
   - "Neden?" açıklaması

4. ✅ Simulation Tools
   - Quote simulation (0x/1inch API)
   - Gas estimation
   - Slippage guard
   - Rug check (heuristic)

5. ✅ API Service
   - /wallet/<addr>/report
   - /alerts
   - /quote/simulate
   - FastAPI + Docker

6. ✅ Capstone Demo
   - 5 dk video (gerçek adres izleme)
   - README kurulum ≤ 10 dk
```

### Başarı Kriteri
> **"Bu sistemi güvenle kullanıp gerçek cüzdanımı izleyebilir miyim?" → EVET**

---

## 🔒 Güvenlik İlkeleri (Non-Negotiable)

### 1. Read-Only First
```
❌ Private key yok
❌ Custody yok
❌ Auto-execute yok

✅ RPC read-only
✅ Testnet (Sepolia)
✅ Paper trading
✅ Simulation only
```

### 2. .env Yönetimi
```
.env.example  → Git'e GİRER
.env          → Git'e GİRMEZ (.gitignore)

Hassas data:
  - RPC URLs
  - API keys
  - Telegram bot token
```

### 3. Signing (İleride)
```
Eğer execute gerekirse:
  ✅ EIP-712 typed data
  ✅ Hardware wallet (Ledger/Trezor)
  ✅ Confirm every tx
  ✅ Gas limit caps
  ✅ Dry-run first
```

### 4. Rate Limiting
```
- IP-based rate limits
- RPC throttling
- Telegram message caps
- API endpoint limits
```

### 5. Legal Disclaimer
```
Her uyarı mesajında:
"⚠️ Bu bilgi amaçlıdır.
Yatırım tavsiyesi değildir.
DYOR (Do Your Own Research)."
```

### 6. Privacy
```
- Log'larda PII yok
- Cüzdan adresleri hashlenebilir
- Metrikler anonim
```

---

## 🛠️ Mimari (Kuşbakışı)

```
┌─────────────────────────────────────────────────┐
│ COLLECTOR                                       │
│ • RPC event listener (web3.py/ethers.js)        │
│ • Off-chain price feeds (CoinGecko/Binance)     │
│ • Block polling / webhook                       │
└────────────┬────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────┐
│ STORE                                           │
│ • DuckDB (OLAP, fast analytics)                 │
│ • Tables: transfers, swaps, balances, prices    │
│ • Time-series views                             │
└────────────┬────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────┐
│ FEATURES                                        │
│ • Volatility (rolling std)                      │
│ • Volume spike (Z-score)                        │
│ • Whale flow (large transfers)                  │
│ • Token distribution (Gini coefficient)         │
└────────────┬────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────┐
│ AI                                              │
│ ├─ NLP: Event classification + summary          │
│ ├─ RAG: Protocol docs → sourced answers         │
│ └─ Agent: Tools (quote, gas, rug check)         │
└────────────┬────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────┐
│ NOTIFIER                                        │
│ • Telegram bot (alerts + context)               │
│ • Threshold triggers                            │
│ • "Neden?" explanation                          │
└────────────┬────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────┐
│ SERVICE                                         │
│ • FastAPI: /report, /alerts, /simulate          │
│ • Rate limiting                                 │
│ • Metrics (p95, alert rate)                     │
│ • Docker Compose                                │
└─────────────────────────────────────────────────┘
```

---

## 🗺️ AI + Crypto Paralel Roadmap (8 Hafta)

### Week 0: Kurulum & Sağlık Kontrolü ✅ BAŞLA

**AI Hattı:**
- Tensor/autograd/MPS
- PyTorch test
- Project skeleton

**Crypto Hattı:**
```
1. RPC Sağlayıcı
   □ Alchemy/Infura/Ankr hesap
   □ Sepolia testnet RPC URL
   □ .env.example oluştur

2. Testnet Cüzdan
   □ Sepolia cüzdan (Metamask/CLI)
   □ Faucet'ten test ETH
   □ Address: 0x... kaydet

3. Sağlık Kontrolü
   □ eth_blockNumber çağrısı (< 300ms)
   □ USDT Sepolia Transfer event'leri (son 100 blok)
   □ Event count > 0

4. Database
   □ DuckDB kurulum
   □ Schema: transfers table
   □ Insert/query test

5. Wallet Report v0
   □ Input: wallet address
   □ Output: 24h inbound/outbound summary
   □ Top 3 counterparties
```

**Definition of Done:**
```
□ crypto/README.md hazır
□ .env.example (RPC_URL, CHAIN_ID)
□ RPC health check < 300ms
□ Transfer events captured (log)
□ DuckDB schema + test query
□ /report skeleton (JSON response)
□ Git commit: "crypto W0: setup + health + report v0"
```

**KPI:**
- RPC health: < 300ms
- Event capture rate: ≥ 99%

---

### Week 1: On-Chain Veri Katmanı

**AI Hattı:**
- Linear regression
- Val MSE < 0.5

**Crypto Hattı:**
```
1. Event Collector
   □ Transfer events → DB
   □ Swap events (Uniswap V2/V3)
   □ Block metadata
   □ Polling loop (30s interval)

2. Price Feeds
   □ CoinGecko API integration
   □ ETH/USDT/USDC prices
   □ Price cache (5 min TTL)

3. Wallet Report v1
   □ 24h net flow (USD)
   □ Token breakdown
   □ Counterparty list
   □ Gas spent

4. API Endpoint
   □ GET /wallet/<addr>/report
   □ JSON response
   □ Cache (1 min TTL)
```

**Definition of Done:**
```
□ Collector running (30s poll)
□ transfers + swaps tables populated
□ /wallet/<addr>/report returns JSON
□ Example report (test wallet)
□ p95 latency < 1s (cached)
```

**KPI:**
- /report p95 < 1s
- DB insert rate: > 10 tx/s

---

### Week 2: Telegram Bot v0 (Threshold Alerts)

**AI Hattı:**
- MLP + MNIST
- Test acc ≥ 0.97

**Crypto Hattı:**
```
1. Alert Rules
   □ Volume spike (Z-score > 2)
   □ Large transfer (> $10k)
   □ New token first seen
   □ Gas spike (> 2x avg)

2. Telegram Bot
   □ Bot setup (BotFather)
   □ /start, /subscribe <addr>
   □ Alert message template
   □ "Neden?" short text

3. Alert Engine
   □ Rule evaluation (每 30s)
   □ Deduplication (1h window)
   □ Priority scoring

4. Notification
   □ Telegram message (formatted)
   □ Link to Etherscan
   □ "Neden bu uyarı?" açıklama
```

**Definition of Done:**
```
□ Telegram bot running
□ 2+ meaningful alerts sent
□ Alert log (timestamp, rule, wallet)
□ Dedup working (no spam)
□ False positive rate < 10%
```

**KPI:**
- Alert accuracy: > 90%
- Telegram delivery: < 5s

---

### Week 3: Olay Sınıflayıcı (Event Classification)

**AI Hattı:**
- NLP fine-tune (BERT)
- F1 ≥ 0.85

**Crypto Hattı:**
```
1. Event Taxonomy
   □ Swap (DEX)
   □ Mint (token creation)
   □ Burn (token destroy)
   □ Bridge (cross-chain)
   □ Airdrop
   □ Internal transfer

2. Training Data
   □ 100+ labeled events
   □ Balanced classes
   □ Etherscan annotations

3. Classifier Model
   □ DistilBERT fine-tune
   □ Input: event signature + params
   □ Output: class + confidence
   □ F1 ≥ 0.80

4. Summary Generator
   □ Türkçe özet (1-2 cümle)
   □ Key entities (token, amount, from/to)
   □ Context (gas, timestamp)
```

**Definition of Done:**
```
□ Classifier F1 ≥ 0.80
□ Test set: 20+ events
□ Confusion matrix
□ Summary examples (5+)
□ Integration: alert messages show class
```

**KPI:**
- Classification F1: ≥ 0.80
- Summary generation: < 1s

---

### Week 4: Protokol RAG (Sourced Explanations)

**AI Hattı:**
- RAG (FAISS/Chroma)
- Recall@k ≥ 60%

**Crypto Hattı:**
```
1. Document Corpus
   □ Uniswap docs (100+ pages)
   □ Curve docs
   □ Chainlink docs
   □ MEV/front-running guides

2. Chunking & Embedding
   □ Chunk size: 500 tokens
   □ bge-small embeddings
   □ FAISS index

3. RAG Pipeline
   □ Query: alert context
   □ Retrieve: top-3 chunks
   □ Generate: sourced answer
   □ Links: doc URL + section

4. Alert Enhancement
   □ Telegram: [Kaynakla Açıkla] button
   □ Callback: RAG query
   □ Response: 2-3 paragraphs + links
```

**Definition of Done:**
```
□ Document corpus indexed (500+ chunks)
□ RAG recall@3 ≥ 60%
□ Alert messages have [Açıkla] button
□ Callback returns sourced answer
□ 95% responses have ≥1 link
```

**KPI:**
- RAG recall@3: ≥ 60%
- Sourced responses: ≥ 95%

---

### Week 5: Tool-Agent (Simulation)

**AI Hattı:**
- Tool-calling agent
- 2-step tool chain

**Crypto Hattı:**
```
1. Tool Definitions
   □ get_quote(token_in, token_out, amount)
     → 0x/1inch API (simulation)
   □ estimate_gas(tx_data)
     → eth_estimateGas
   □ check_slippage(quote, max_slippage)
     → heuristic check
   □ rug_check(token_addr)
     → Honeypot.is API / heuristic

2. Agent Loop
   □ Plan: "User wants to swap X→Y"
   □ Call: get_quote()
   □ Observe: price, slippage
   □ Call: rug_check()
   □ Observe: risk score
   □ Respond: summary + recommendation

3. Simulation
   □ No real transactions
   □ Paper results logged
   □ "Eğer şu anda yapsaydın..." analysis
```

**Definition of Done:**
```
□ 4 tools implemented
□ Agent: 2-step chain working
□ Example: quote → rug check → summary
□ Logs: plan/tool/observe/respond
□ Response time < 2s
```

**KPI:**
- Tool chain success: ≥ 95%
- Simulation latency: < 2s

---

### Week 6: LoRA Fine-tune (Özel Üslup)

**AI Hattı:**
- LoRA 7B model
- Qualitative improvement

**Crypto Hattı:**
```
1. Custom Corpus
   □ Kendi crypto notların
   □ Eski analiz metinlerin
   □ Üslup örnekleri (100+ paragraphs)

2. LoRA Training
   □ Base: Llama 3.2 7B / Qwen 2.5
   □ r=8, alpha=16
   □ 100 steps
   □ Validation: perplexity

3. Style Transfer
   □ Generic alert: "Large transfer detected"
   □ LoRA alert: "Dikkat: Balina hareketi! 
     1.2M USDT cüzdandan ayrıldı.
     Geçmiş veri benzer çıkışları 
     48h içinde fiyat düşüşü takip etti."

4. A/B Test
   □ 10 örnek event
   □ Generic vs LoRA summary
   □ Blind evaluation: hangisi daha net?
```

**Definition of Done:**
```
□ LoRA checkpoint saved
□ 5+ before/after pairs
□ A/B test: LoRA preferred ≥ 60%
□ Integration: bot uses LoRA
```

**KPI:**
- A/B preference: ≥ 60%
- Summary quality (subjective)

---

### Week 7: Servisleştir & İzleme

**AI Hattı:**
- FastAPI + Docker
- p95 < 2.5s

**Crypto Hattı:**
```
1. API Endpoints
   □ GET /healthz
   □ GET /wallet/<addr>/report
   □ GET /alerts?since=<timestamp>
   □ POST /quote/simulate
   □ GET /rag/explain?event_id=<id>

2. Rate Limiting
   □ IP-based: 10 req/min
   □ API key: 100 req/min
   □ Telegram: 20 msg/min

3. Monitoring
   □ /metrics (Prometheus-style)
   □ RPC call latency (p50, p95, p99)
   □ Alert rate (per hour)
   □ Error rate
   □ Cache hit rate

4. Docker Compose
   □ collector service
   □ api service
   □ telegram bot service
   □ DuckDB volume
```

**Definition of Done:**
```
□ docker compose up → all services running
□ /healthz returns 200
□ All endpoints tested
□ p95 latency < 2.5s
□ /metrics endpoint working
□ Rate limiting enforced
```

**KPI:**
- API p95: < 2.5s
- Error rate: < 1%

---

### Week 8: Capstone Demo

**AI Hattı:**
- E2E integration
- 5 dk demo video

**Crypto Hattı:**
```
1. End-to-End Flow
   □ Real wallet monitoring (read-only)
   □ Event detection
   □ Classification
   □ RAG explanation
   □ Telegram alert
   □ Simulation (quote)

2. Demo Scenarios
   □ Scenario 1: Large swap detection
   □ Scenario 2: New token airdrop
   □ Scenario 3: Whale movement

3. Video Content (5 min)
   □ 0-1 min: Problem statement
   □ 1-2 min: Architecture overview
   □ 2-4 min: Live demo (3 scenarios)
   □ 4-5 min: Key learnings + v2 roadmap

4. Documentation
   □ README.md (setup ≤ 10 min)
   □ REPORT.md (retrospective)
   □ API docs (OpenAPI/Swagger)
   □ Troubleshooting guide
```

**Definition of Done:**
```
□ 5 min video recorded
□ 3 scenarios demonstrated
□ README tested (fresh install < 10 min)
□ REPORT.md: learnings + v2 plans
□ Git tag: v1.0-capstone
```

**KPI:**
- Setup time: < 10 min
- Demo quality (subjective)

---

## 📏 Haftalık KPI Özeti

```
Week 0: RPC health < 300ms, event capture ≥ 99%
Week 1: /report p95 < 1s
Week 2: Alert accuracy > 90%
Week 3: Classification F1 ≥ 0.80
Week 4: RAG recall@3 ≥ 60%, sourced ≥ 95%
Week 5: Tool chain success ≥ 95%, latency < 2s
Week 6: A/B LoRA preference ≥ 60%
Week 7: API p95 < 2.5s, error < 1%
Week 8: Setup < 10 min, 3 scenarios
```

---

## 🔧 Tech Stack (Crypto)

### Blockchain
```
EVM (Ethereum Virtual Machine)
  - Testnet: Sepolia
  - Mainnet: Ethereum (later)
  - L2: Base, Arbitrum (future)
```

### RPC Providers
```
Alchemy   → Generous free tier, good docs
Infura    → Reliable, WebSocket support
Ankr      → Public endpoints (rate limited)
QuickNode → Premium, low latency
```

### Libraries
```
Python:
  - web3.py (Ethereum interaction)
  - eth-abi, eth-utils
  - duckdb (analytics DB)
  - asyncio (async event loop)

JavaScript (optional):
  - ethers.js
  - viem
```

### APIs
```
Price Feeds:
  - CoinGecko (free, 50 calls/min)
  - Binance API (real-time)
  - CryptoCompare

DEX Aggregators:
  - 0x API (quote simulation)
  - 1inch API

Security:
  - Honeypot.is (rug check)
  - Token Sniffer
```

### Database
```
DuckDB:
  - OLAP (analytics-first)
  - Embedded (no server)
  - Fast time-series queries
  - SQL interface

Schema:
  - transfers (block, tx, from, to, value, token)
  - swaps (pool, token_in, token_out, amount_in, amount_out)
  - balances (wallet, token, balance, timestamp)
  - prices (token, price_usd, timestamp)
```

### Messaging
```
Telegram Bot API:
  - python-telegram-bot library
  - Webhook or polling
  - Inline buttons
```

---

## 🗂️ Crypto Klasör Yapısı

```
crypto/
├── README.md                    Week 0 setup guide
├── .env.example                 Config template
│
├── collector/                   On-chain data collection
│   ├── __init__.py
│   ├── rpc_health.py            RPC health check
│   ├── event_capture.py         Event listener (Transfer, Swap)
│   ├── price_feeds.py           Off-chain price fetching
│   └── polling.py               Main polling loop
│
├── store/                       Database layer
│   ├── __init__.py
│   ├── db.py                    DuckDB connection
│   ├── models.py                Table schemas
│   └── queries.py               Common queries
│
├── db/
│   ├── schema.sql               DDL statements
│   └── seed.sql                 Test data (optional)
│
├── features/                    Signal generation
│   ├── __init__.py
│   ├── volatility.py            Rolling volatility
│   ├── volume.py                Volume spike detection
│   └── whale.py                 Large transfer detection
│
├── ai/
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── classifier.py        Event classification
│   │   └── summarizer.py        Türkçe özet
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexer.py           Document indexing
│   │   └── retriever.py         RAG query
│   │
│   └── agent/
│       ├── __init__.py
│       ├── tools.py             Tool definitions
│       └── loop.py              Agent loop
│
├── notifier/
│   ├── __init__.py
│   ├── telegram_bot.py          Bot logic
│   ├── alert_engine.py          Rule evaluation
│   └── templates.py             Message templates
│
├── service/                     FastAPI
│   ├── __init__.py
│   ├── main.py                  App entry
│   ├── routes/
│   │   ├── wallet.py            /wallet/<addr>
│   │   ├── alerts.py            /alerts
│   │   └── simulate.py          /quote/simulate
│   └── middleware/
│       ├── rate_limit.py
│       └── auth.py
│
├── tests/
│   ├── test_collector.py
│   ├── test_features.py
│   └── test_api.py
│
├── scripts/
│   ├── setup.sh                 Initial setup
│   └── seed_db.sh               Seed test data
│
├── docker/
│   ├── Dockerfile.collector
│   ├── Dockerfile.api
│   └── docker-compose.yml
│
└── docs/
    ├── api.md                   API documentation
    └── architecture.md          System design
```

---

## 🧯 Troubleshooting (Kripto Özel)

### RPC Issues

**Sorun: Rate limit (429)**
```
Çözüm:
1. .env'de farklı RPC provider dene
2. Polling interval'i artır (30s → 60s)
3. Cache TTL'i uzat (1 min → 5 min)
4. Free tier → paid tier
```

**Sorun: Slow response (> 1s)**
```
Teşhis:
- RPC provider latency?
- Network congestion?
- Query too complex?

Çözüm:
- Archive node yerine full node
- Batch requests (eth_getLogs)
- Fallback provider
```

### Event Capture

**Sorun: Eksik event**
```
Teşhis:
- Block range too wide?
- Filter too specific?
- Reorg (chain reorganization)?

Çözüm:
- Smaller block batches (100 → 10)
- Broader filter (all Transfer events)
- Finality confirmation (12+ blocks)
```

### Database

**Sorun: DuckDB file locked**
```
Çözüm:
- Tek writer process (collector)
- Read-only connections (API)
- WAL mode (Write-Ahead Logging)
```

### Telegram

**Sorun: Message flood**
```
Çözüm:
- Deduplication (1h window)
- Rate limit (20 msg/min)
- Priority queue (critical first)
```

---

## 📚 Kaynaklar (Kripto)

### Official Docs
```
Ethereum:
  - ethereum.org/developers
  - EIP proposals

Web3.py:
  - web3py.readthedocs.io

Ethers.js:
  - docs.ethers.org

DuckDB:
  - duckdb.org/docs
```

### Learning
```
Smart Contract Security:
  - consensys.github.io/smart-contract-best-practices
  - solidity-by-example.org

MEV & Front-running:
  - flashbots.net
  - mev.wiki
```

### APIs
```
Alchemy:
  - docs.alchemy.com

CoinGecko:
  - coingecko.com/api/documentation

0x API:
  - 0x.org/docs/api
```

---

## ❓ SSS (Kripto)

### Trade edecek miyiz?
```
Varsayılan: HAYIR

Mod 1: Read-only (W0-W8)
  - Sadece izleme
  - Alert generation
  - Paper trading

Mod 2: Testnet (future)
  - Sepolia/Goerli
  - Test tokens
  - Real signing (but no value)

Mod 3: Mainnet Simulation (future)
  - Real data
  - Simulated execution
  - No on-chain tx
```

### Hangi zincir?
```
Phase 1 (W0-W8):
  - Sepolia (testnet)
  - Ethereum mainnet (read-only)

Phase 2 (v2):
  - L2: Base, Arbitrum, Optimism
  - Multi-chain wallet tracking
```

### Neden RAG/LLM?
```
Problem:
  "Large swap detected" → Kullanıcı: "So what?"

Çözüm (RAG):
  "Large swap detected.
  
  Bu MEV bot olabilir çünkü:
  - Gas price 2x normal
  - Tx frontrun pozisyonunda
  - Kaynak: Flashbots docs, Section 3.2
  
  Link: https://docs.flashbots.net/..."

→ Uyarıya BAĞLAM ve KAYNAK ekler
```

### Private key gerekli mi?
```
Week 0-8: HAYIR
  - Read-only RPC
  - No signing

İleride (opsiyonel):
  - Hardware wallet (Ledger)
  - EIP-712 signing
  - Confirm every tx
  - Gas caps
```

### DuckDB neden?
```
Alternatifler:
  - PostgreSQL: Güçlü ama heavy
  - SQLite: Hafif ama analytics zayıf
  - ClickHouse: Powerful ama overkill

DuckDB:
  ✅ Embedded (no server)
  ✅ OLAP-first (analytics)
  ✅ Fast time-series
  ✅ SQL interface
  ✅ Python integration
  
→ Prototip için ideal
```

---

## 🎉 Sonraki Adım

### Bugün (Week 0 Crypto Setup - 30 dk)

```bash
# 1. Crypto klasörüne git
cd /Users/onur/code/novadev-protocol/crypto

# 2. README'yi oku (5 dk)
cat README.md

# 3. .env ayarla (5 dk)
cp .env.example .env
# RPC_URL'ini ekle

# 4. RPC health check (5 dk)
python collector/rpc_health.py

# 5. Event capture test (10 dk)
python collector/event_capture.py

# 6. DB schema (5 dk)
duckdb db/crypto.db < db/schema.sql
```

### Yarın (Week 1 - AI + Crypto Paralel)

```
AI:    Linear regression (45 dk sprint)
Crypto: On-chain data layer (collector loop)

Toplam: 2-3 saat
```

---

**NovaDev Crypto — "Okuyan, Anlayan, Uyarı Veren"**

*Versiyon: 1.0*  
*Son Güncelleme: 2025-10-06*  
*Status: Week 0 Ready to Start! 🔗*
