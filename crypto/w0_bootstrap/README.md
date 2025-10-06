# NovaDev-Crypto — W0 Bootstrap

**Amaç:** RPC health ✅, transfer ingest ✅, basit rapor ✅ (read-only, Sepolia)

**⚠️ Yasal Uyarı:** Bu sistem bilgi amaçlıdır, finansal tavsiye değildir. DYOR (Do Your Own Research).

---

## 🎯 Week 0 Hedefler

```
□ RPC sağlayıcı bağlantısı (< 300ms)
□ Transfer event'lerini DuckDB'ye kaydet
□ Wallet raporu v0 (inbound/outbound)
```

**Definition of Done:**
- ✅ `rpc_health.py` → "RPC OK"
- ✅ `capture_transfers.py` → DuckDB'ye kayıt (basic)
- ✅ `capture_transfers_idempotent.py` → İdempotent ingest + state tracking
- ✅ `report_v0.py` → 24h özet (CLI pretty print)
- ✅ `report_json.py` → JSON format (API-ready)
- ✅ `validate_report.py` → JSON schema validation
- ✅ FastAPI service → `/wallet/{addr}/report` endpoint

---

## 🚀 Hızlı Başlangıç (30-45 dk)

### 1. RPC Sağlayıcı Hesabı (5 dk)

**Alchemy (Önerilen):**
```bash
# 1. https://dashboard.alchemy.com → Kayıt
# 2. "Create App" → Sepolia seç
# 3. "View Key" → HTTPS URL kopyala
```

**Alternatif:**
- Infura: https://infura.io
- Ankr: https://www.ankr.com/rpc/ (public, rate limited)

---

### 2. .env Dosyası (5 dk)

```bash
cd /Users/onur/code/novadev-protocol/crypto/w0_bootstrap
cp .env.example .env
# vim .env → RPC_URL'ini yapıştır
```

**Örnek `.env`:**
```env
RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
CHAIN_ID=11155111
TOKEN_ADDRESS=0x0000000000000000000000000000000000000000
TOKEN_DECIMALS=18
START_BLOCK=0
```

**Token seçenekleri:**
- `0x0000...` = ETH native transfers (tüm ağ)
- Sepolia USDT: `0x7169D38820dfd117C3FA1f22a697dBA58d90BA06`
- Herhangi bir ERC-20 contract address

---

### 3. Dependencies (5 dk)

```bash
cd /Users/onur/code/novadev-protocol
pip install -e ".[crypto]"
```

**Gereksinimler:**
- `web3` (veya `requests` ile raw JSON-RPC)
- `duckdb`
- `python-dotenv`

---

### 4. RPC Health Check (2 dk)

```bash
python crypto/w0_bootstrap/rpc_health.py
```

**Beklenen çıktı:**
```
✅ RPC OK | latest block: 12345678 | 145.3 ms
```

**Hata durumları:**
```
❌ "RPC_URL yok"
   → .env dosyasını kontrol et

❌ "Connection timeout"
   → RPC provider down veya yanlış URL

❌ "> 300ms"
   → Farklı provider dene veya coğrafi yakın seç
```

---

### 5. Transfer Capture (15 dk)

```bash
# Son 5000 blok'u tara
python crypto/w0_bootstrap/capture_transfers.py --blocks 5000
```

**Ne yapıyor?**
- `eth_getLogs` ile Transfer event'lerini çeker
- ERC-20 Transfer signature: `Transfer(address,address,uint256)`
- DuckDB'ye kaydeder: `crypto/w0_bootstrap/onchain.duckdb`

**Çıktı:**
```
Scanning logs 12340000..12345000 (latest=12345000)
+156 logs (total 156)
+203 logs (total 359)
...
✅ Done. Inserted 1542 transfer logs into onchain.duckdb
```

**Parametreler:**
```bash
# Daha fazla blok
python crypto/w0_bootstrap/capture_transfers.py --blocks 10000

# Spesifik aralık (gelecekte)
# python crypto/w0_bootstrap/capture_transfers.py --start 12340000 --end 12345000
```

---

### 6. Wallet Report (5 dk)

```bash
# Örnek cüzdan (Vitalik'in Sepolia adresi)
python crypto/w0_bootstrap/report_v0.py \
  --wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 \
  --hours 24
```

**Çıktı:**
```
✅ Wallet Report
- Window: last 24h
- Inbound:  1.234567
- Outbound: 0.567890
- Tx count: 12
- Top counterparties:
  • 0xabc... : 0.500000
  • 0xdef... : 0.300000
  • 0x123... : 0.150000
```

**Kendi cüzdanın:**
```bash
python crypto/w0_bootstrap/report_v0.py \
  --wallet 0xYOUR_WALLET_ADDRESS \
  --hours 24
```

---

## 📊 Database Yapısı

**Tablo: `transfers`**
```sql
CREATE TABLE transfers (
    block_number BIGINT,
    block_time   TIMESTAMP,
    tx_hash      TEXT,
    log_index    INTEGER,
    token        TEXT,
    from_addr    TEXT,
    to_addr      TEXT,
    raw_value    DECIMAL(38,0),  -- Wei/smallest unit
    value_unit   DOUBLE          -- Decimal adjusted
)
```

**Query örnekleri:**
```bash
# DuckDB CLI aç
duckdb crypto/w0_bootstrap/onchain.duckdb

# Toplam transfer sayısı
SELECT COUNT(*) FROM transfers;

# Son 10 transfer
SELECT * FROM transfers ORDER BY block_number DESC LIMIT 10;

# Belirli cüzdan
SELECT * FROM transfers 
WHERE lower(from_addr) = '0x...' OR lower(to_addr) = '0x...'
ORDER BY block_number DESC;

# Günlük hacim
SELECT DATE(block_time) AS day, SUM(value_unit) AS volume
FROM transfers
GROUP BY day
ORDER BY day DESC;
```

---

## 🧯 Troubleshooting

### Problem: "RPC_URL yok"
```bash
# .env var mı?
ls -la crypto/w0_bootstrap/.env

# Yoksa kopyala
cp crypto/w0_bootstrap/.env.example crypto/w0_bootstrap/.env
vim crypto/w0_bootstrap/.env
```

### Problem: "No logs found"
```bash
# 1. Testnet'te aktivite az olabilir
# Daha fazla blok tara:
python crypto/w0_bootstrap/capture_transfers.py --blocks 10000

# 2. Token filtresi çok dar
# .env'de TOKEN_ADDRESS=0x0000... yap (tüm transferler)

# 3. Mainnet'e geç (daha fazla aktivite)
# .env → RPC_URL mainnet endpoint
```

### Problem: "Rate limit (429)"
```bash
# capture_transfers.py'de time.sleep() artır
# Veya farklı RPC provider
```

### Problem: DuckDB locked
```bash
# Başka bir process DB'yi kullanıyor mu?
ps aux | grep duckdb

# Kill et veya DB dosyasını sil + yeniden capture
rm crypto/w0_bootstrap/onchain.duckdb
python crypto/w0_bootstrap/capture_transfers.py --blocks 5000
```

---

## 🎯 Week 0 Checklist

```bash
# Tüm adımları tek seferde test et
cd /Users/onur/code/novadev-protocol/crypto/w0_bootstrap

# 1. Health
python rpc_health.py
# → ✅ RPC OK

# 2. Capture
python capture_transfers.py --blocks 5000
# → ✅ Inserted N logs

# 3. Report
python report_v0.py --wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 --hours 24
# → ✅ Wallet Report

# Hepsi OK ise:
echo "Week 0 Complete ✓"
```

---

## 📚 Sonraki Adımlar (Week 1)

### Crypto W1: Ingest Katmanı
```
□ Event collector loop (30s polling)
□ Price feeds (CoinGecko API)
□ Wallet report → JSON function (FastAPI hazırlık)
□ Docker: collector service + DB volume
```

### AI W1: Linear Regression
```
□ Data synth + manual GD
□ nn.Module training
□ Val MSE < 0.5
```

**Paralel çalışma:**
```bash
# Sabah: AI (60 dk)
cd ai/week1_tensors
python train.py

# Öğlen: Crypto (45 dk)
cd crypto/w1_ingest
python collector_loop.py

# Akşam: Commit + log
git add -A
git commit -m "W1: AI MSE=0.42 ✓, Crypto collector running ✓"
```

---

## 📄 Dosyalar

```
crypto/w0_bootstrap/
├── README.md                           (bu dosya)
├── .env.example                        Config template
├── .env                                (Git'e GİRMEZ)
│
├── rpc_health.py                       RPC check
├── capture_transfers.py                Event ingest (basic)
├── capture_transfers_idempotent.py     ⭐ Idempotent ingest + state
├── report_v0.py                        Wallet report (CLI pretty)
├── report_json.py                      ⭐ Wallet report (JSON)
├── validate_report.py                  ⭐ JSON schema validator
│
└── onchain.duckdb                      (otomatik oluşur)
    ├── transfers (table)
    └── scan_state (table)              ⭐ State tracking
```

**Yeni Özellikler (v1.1):**
- ⭐ `capture_transfers_idempotent.py`: Production-ready ingest
  - State tracking (resume from last block)
  - Reorg protection (CONFIRMATIONS buffer)
  - Anti-join pattern (no duplicates)
  
- ⭐ `report_json.py`: JSON output for API
  
- ⭐ `validate_report.py`: JSON schema validation
  - Schema: `schemas/report_v1.json`
  
- ⭐ FastAPI service: `crypto/service/app.py`
  - `/healthz`, `/wallet/{addr}/report`
  
- ⭐ Makefile shortcuts: `make c.health`, `make c.capture.idem`, etc.

---

**Week 0 Bootstrap — Read-Only, Testnet, Safe! 🔒**

*Son Güncelleme: 2025-10-06*
