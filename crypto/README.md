# 🔗 NovaDev Crypto — On-Chain Intel Copilot

**"Okuyan, Anlayan, Uyarı Veren, Simüle Eden"**

> Bu bir trading sinyal kursu DEĞİL; güvenli, read-only, paper-trading odaklı **On-Chain Intelligence** sistemi.

**⚠️ Yasal Uyarı:** Bu sistem bilgilendirme amaçlıdır, finansal tavsiye değildir. DYOR (Do Your Own Research).

---

## 📚 Dökümantasyon Hiyerarşisi

```
1. Program Genel Bakış (AI + Crypto Paralel)
   📄 docs/program_overview.md ⭐⭐⭐ ÖNCE OKU!
   
2. Crypto Detaylı Roadmap
   📄 docs/crypto_overview.md (8 hafta detay)
   
3. Week 0 Hızlı Başlangıç
   📄 crypto/w0_bootstrap/README.md ✅ COMPLETE!
   
4. 🎓 "Hoca Tahtası" Teori Serisi (W0) ⭐ 10/10 COMPLETE!
   📄 crypto/docs/w0_bootstrap/README.md (19,005 satır dokümantasyon)
   📄 crypto/docs/w0_bootstrap/01_tahta_evm_giris.md ✅ (1,277 satır)
   📄 crypto/docs/w0_bootstrap/02_tahta_rpc_101.md ✅ (1,012 satır)
   📄 crypto/docs/w0_bootstrap/03_tahta_transfer_anatomi.md ✅ (1,094 satır)
   📄 crypto/docs/w0_bootstrap/04_tahta_getlogs_pencere_reorg.md ✅ (2,266 satır)
   📄 crypto/docs/w0_bootstrap/05_tahta_duckdb_idempotent.md ✅ (1,791 satır)
   📄 crypto/docs/w0_bootstrap/06_tahta_state_resume.md ✅ (2,349 satır)
   📄 crypto/docs/w0_bootstrap/07_tahta_rapor_json_schema.md ✅ (1,971 satır)
   📄 crypto/docs/w0_bootstrap/08_tahta_fastapi_mini.md ✅ (2,069 satır)
   📄 crypto/docs/w0_bootstrap/09_tahta_kalite_ci.md ✅ (2,157 satır)
   📄 crypto/docs/w0_bootstrap/10_tahta_troubleshoot_runbooks.md ✅ (2,727 satır)
   
5. Haftalık Klasörler
   📁 crypto/w1_ingest/      (Week 1)
   📁 crypto/w2_telegram/    (Week 2)
   ...
```

**İlk Adım:** [w0_bootstrap/README.md](w0_bootstrap/README.md) → 30-45 dk setup

---

## 🎯 Crypto Hattı Özeti

### Hedef (8 Hafta Sonunda)
```
✓ On-chain veri toplayıcı (EVM, read-only)
✓ DuckDB depolama + analytics
✓ Cüzdan raporu (24h, 7d, custom)
✓ Telegram uyarı botu (eşik + etiketleme)
✓ Event classifier (Swap/Mint/Bridge/etc)
✓ Protokol RAG (kaynaklı açıklama)
✓ Simülasyon araçları (quote, gas, risk)
✓ FastAPI servis (/wallet, /alerts, /simulate)
```

### Güvenlik İlkeleri (Non-Negotiable)
```
❌ Private key YOK
❌ Custody YOK
❌ Auto-execute YOK
✅ Read-only RPC
✅ Testnet (Sepolia) first
✅ Paper trading / simulation only
```

---

## 🗺️ 8 Haftalık Roadmap (Crypto Hattı)

| Week | Konu | Metrik/DoD |
|------|------|------------|
| **0** ✅ | Bootstrap + Tahta Serisi (10/10) | RPC health<300ms, Tests 39/39 ✓, Docs 19,005 satır |
| **1** 👉 | Collector Loop + API Perf | p95<1s, cache hit>70%, error=0% |
| **2** | Telegram Bot v0 | 2+ meaningful alerts |
| **3** | Event Classifier | F1≥0.80, Türkçe özet |
| **4** | Protokol RAG | Sourced responses≥95% |
| **5** | Simülasyon | Quote<2s, Risk check |
| **6** | Üslup Uyarlama | Citation≥95% |
| **7** | Servis + İzleme | p95<2.5s, error<1% |
| **8** | Capstone | 3 scenario demo |

**Detay:** [docs/crypto_overview.md](../docs/crypto_overview.md)

---

## 🚀 Hızlı Başlangıç (Week 0)

### 1. Dependencies (5 dk)
```bash
cd /Users/onur/code/novadev-protocol
pip install -e ".[crypto]"
```

### 2. RPC Provider (5 dk)
```bash
# Alchemy'ye kayıt (önerilen)
# https://dashboard.alchemy.com
# "Create App" → Sepolia
# API Key kopyala
```

### 3. Setup (5 dk)
```bash
cd crypto/w0_bootstrap
cp .env.example .env
# vim .env → RPC_URL ekle
```

### 4. Test (15 dk)
```bash
# Health check
python rpc_health.py

# Event capture
python capture_transfers.py --blocks 5000

# Wallet report
python report_v0.py --wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```

**Detaylı Adımlar:** [w0_bootstrap/README.md](w0_bootstrap/README.md)

---

## 🔧 Tech Stack

### Blockchain
```
EVM (Ethereum Virtual Machine)
  - Testnet: Sepolia (Week 0-1)
  - Mainnet: Ethereum (Week 2+, read-only)
  - L2: Base, Arbitrum (future)
```

### RPC Providers
```
Alchemy   → Generous free tier
Infura    → Reliable, WebSocket support
Ankr      → Public endpoints (rate limited)
```

### Libraries
```
Python:
  - web3.py (Ethereum interaction)
  - duckdb (analytics DB)
  - requests (HTTP/API)
  - python-telegram-bot (Week 2+)
```

### APIs
```
Price Feeds:
  - CoinGecko (free, 50 calls/min)
  - Binance API (real-time)

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
  - swaps (pool, token_in, token_out, amounts)
  - balances (wallet, token, balance, timestamp)
  - prices (token, price_usd, timestamp)
  - alerts (type, wallet, description, status)
```

---

## 📊 Klasör Yapısı

```
crypto/
├── README.md                    (bu dosya - genel bakış)
│
├── w0_bootstrap/                ✅ Week 0 (30-45 dk setup)
│   ├── README.md
│   ├── .env.example
│   ├── rpc_health.py
│   ├── capture_transfers.py
│   └── report_v0.py
│
├── w1_ingest/                   Week 1 (veri katmanı)
│   ├── README.md
│   ├── capture_swaps.py
│   ├── price_fetcher.py
│   ├── report.py
│   └── collector_loop.py
│
├── w2_telegram/                 Week 2 (uyarı botu)
│   ├── README.md
│   ├── bot.py
│   ├── alert_engine.py
│   └── templates.py
│
├── w3_classifier/               Week 3 (event sınıflama)
│   ├── README.md
│   ├── classifier.py
│   └── summarizer.py
│
├── w4_rag/                      Week 4 (protokol RAG)
│   ├── README.md
│   ├── indexer.py
│   └── retriever.py
│
├── w5_simulation/               Week 5 (quote + risk)
│   ├── README.md
│   ├── tools.py
│   └── agent.py
│
├── w6_lora/                     Week 6 (üslup)
│   ├── README.md
│   └── train_lora.py
│
├── w7_service/                  Week 7 (FastAPI)
│   ├── README.md
│   ├── main.py
│   ├── routes/
│   └── docker-compose.yml
│
└── w8_capstone/                 Week 8 (demo)
    ├── README.md
    ├── demo.py
    └── REPORT.md
```

---

## 🧯 Troubleshooting

### RPC Issues

**Problem: Rate limit (429)**
```
Çözüm:
1. .env'de farklı RPC provider
2. Polling interval artır (30s → 60s)
3. Cache TTL uzat (1 min → 5 min)
```

**Problem: Slow response (> 1s)**
```
Teşhis:
- RPC provider latency?
- Network congestion?

Çözüm:
- Archive node → full node
- Batch requests
- Fallback provider
```

### Event Capture

**Problem: Eksik event**
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

**Problem: DuckDB file locked**
```
Çözüm:
- Tek writer process (collector)
- Read-only connections (API)
- WAL mode
```

---

## 📚 Kaynaklar

### Official Docs
```
Ethereum:
  - ethereum.org/developers
  - EIP proposals

Web3.py:
  - web3py.readthedocs.io

DuckDB:
  - duckdb.org/docs
```

### Learning
```
Smart Contract Security:
  - consensys.github.io/smart-contract-best-practices

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

## ❓ SSS

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
  - Kaynak: Flashbots docs, Section 3.2"

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

---

## 🎉 Sonraki Adım

### Bugün (Week 0 - 30-45 dk)

```bash
# 1. Setup
cd /Users/onur/code/novadev-protocol/crypto/w0_bootstrap
cat README.md

# 2. Dependencies
pip install -e ".[crypto]"

# 3. Configure
cp .env.example .env
# vim .env → RPC_URL

# 4. Test
python rpc_health.py
python capture_transfers.py --blocks 5000
python report_v0.py --wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045

# Hepsi ✓ ise:
echo "Crypto W0 Complete ✓"
```

### Yarın (Week 1 - AI + Crypto Paralel)

```
Sabah (60-90 dk): AI Linear Regression
  cd week1_tensors
  python train.py

Öğlen (45-60 dk): Crypto Veri Katmanı
  cd crypto/w1_ingest
  python collector_loop.py

Akşam (15 dk): Commit + log
  git commit -m "W1: AI MSE=0.42 ✓, Crypto collector running ✓"
```

---

**NovaDev Crypto — "Read-Only, Safe, Informative"**

*Versiyon: 1.1 (Paralel Program)*  
*Son Güncelleme: 2025-10-06*  
*Status: Week 0 Ready! 🔗*

**Program Ana Sayfa:** [docs/program_overview.md](../docs/program_overview.md)