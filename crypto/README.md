# 🔗 Crypto Week 0 — Kurulum & Sağlık Kontrolü

**Hedef:** RPC bağlantısı + Event yakalama + Wallet raporu v0 (30-45 dk)

---

## 🎯 Week 0 Hedefleri

```
□ RPC sağlayıcı hesabı (Alchemy/Infura)
□ Sepolia testnet cüzdan
□ .env configurasyonu
□ RPC health check (< 300ms)
□ Transfer event capture (son 100 blok)
□ DuckDB schema + test query
□ Wallet report v0 (JSON)
```

**Definition of Done:**
- ✅ RPC health: < 300ms
- ✅ Event capture rate: ≥ 99%
- ✅ /report skeleton çalışıyor

---

## 📦 Gereksinimler

### Python Packages

```bash
# Core
pip install web3>=6.0.0
pip install eth-abi eth-utils

# Database
pip install duckdb>=0.9.0

# API (Week 1+)
pip install fastapi uvicorn
pip install python-telegram-bot
pip install aiohttp

# Optional
pip install python-dotenv
```

### External Services

```
1. RPC Provider (birini seç):
   □ Alchemy  (alchemy.com) - Free: 3M compute units/month
   □ Infura   (infura.io)   - Free: 100k requests/day
   □ Ankr     (ankr.com)    - Public endpoints (rate limited)

2. Testnet Faucet:
   □ Sepolia: faucet.sepolia.dev
   □ Sepolia ETH: sepoliafaucet.com

3. Telegram (Week 2+):
   □ BotFather (@BotFather) - create bot
```

---

## 🚀 Kurulum (Adım Adım)

### 1. RPC Sağlayıcı (Alchemy Örnek - 5 dk)

```bash
# 1. Alchemy'ye kayıt ol
# https://dashboard.alchemy.com

# 2. "Create App" tıkla
# - Name: NovaDev Sepolia
# - Chain: Ethereum
# - Network: Sepolia

# 3. "View Key" → HTTPS URL'ini kopyala
# https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
```

**Alternatif (Infura):**
```bash
# 1. infura.io → kayıt
# 2. Create Project → Sepolia
# 3. URL kopyala
# https://sepolia.infura.io/v3/YOUR_PROJECT_ID
```

---

### 2. .env Dosyası (5 dk)

```bash
# crypto/.env oluştur
cd /Users/onur/code/novadev-protocol/crypto
cp .env.example .env
```

**crypto/.env içeriği:**
```bash
# RPC Configuration
RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
CHAIN_ID=11155111
NETWORK_NAME=Sepolia

# Wallet (read-only, monitoring için)
WATCH_WALLET=0x0000000000000000000000000000000000000000

# Database
DB_PATH=db/crypto.db

# Telegram (Week 2+)
# TELEGRAM_BOT_TOKEN=your_token_here
# TELEGRAM_CHAT_ID=your_chat_id

# API Keys (Week 4+)
# COINGECKO_API_KEY=
# ZERO_X_API_KEY=

# Rate Limiting
RPC_MAX_CALLS_PER_SECOND=10
POLL_INTERVAL_SECONDS=30

# Logging
LOG_LEVEL=INFO
```

**Güvenlik:**
```bash
# .env dosyası Git'e GİRMEZ
echo "crypto/.env" >> ../.gitignore
```

---

### 3. Testnet Cüzdan (5 dk)

**Opsiyonel** (Week 0'da gerekmez, ama hazırlayalım)

```bash
# Yöntem 1: Metamask
# 1. Metamask yükle
# 2. Network: Sepolia ekle
# 3. Address kopyala

# Yöntem 2: CLI (eth-account)
pip install eth-account
python -c "from eth_account import Account; acc = Account.create(); print(f'Address: {acc.address}\nPrivate: {acc.key.hex()}')"

# ⚠️ Private key'i .env'e YAZMA (Week 0'da gerekmez)
```

**Test ETH Al:**
```bash
# Faucet: sepoliafaucet.com
# Address'ini yapıştır → "Send Me ETH"
# 0.5 ETH gelecek (testnet)
```

---

### 4. DuckDB Schema (5 dk)

```bash
# Schema oluştur
cd /Users/onur/code/novadev-protocol/crypto
duckdb db/crypto.db < db/schema.sql

# Test query
duckdb db/crypto.db -c "SELECT name FROM sqlite_master WHERE type='table';"
```

**Beklenen output:**
```
transfers
swaps
balances
prices
```

---

### 5. RPC Health Check (5 dk)

```bash
# Test script
cd /Users/onur/code/novadev-protocol/crypto
python collector/rpc_health.py
```

**Beklenen output:**
```
=== RPC Health Check ===
RPC URL: https://eth-sepolia.g.alchemy.com/v2/...
Network: Sepolia (11155111)

[✓] Connection OK
[✓] Latest block: 12345678
[✓] Latency: 145ms (< 300ms) ✓
[✓] Chain ID: 11155111 ✓

Status: HEALTHY ✓
```

**Sorun giderleri:**
```bash
# Hata: "HTTPError 401"
# → .env'de RPC_URL yanlış/eksik

# Hata: "Timeout"
# → RPC provider down / network issue

# Hata: "Wrong chain_id"
# → .env'de CHAIN_ID yanlış (Sepolia = 11155111)
```

---

### 6. Event Capture Test (10 dk)

```bash
# Transfer event'lerini yakala
python collector/event_capture.py --blocks 100
```

**Beklenen output:**
```
=== Event Capture Test ===
Scanning last 100 blocks...

Block range: 12345578 → 12345678
Filter: Transfer(address,address,uint256)

[✓] Block 12345578: 3 transfers
[✓] Block 12345579: 1 transfer
...
[✓] Block 12345678: 2 transfers

Total: 156 transfers captured
Capture rate: 100% (156/156)
Avg block time: 12.1s

Status: OK ✓
```

**Ne yapıyor?**
```python
# Pseudo-code
for block in latest_100_blocks:
    events = get_logs(
        from_block=block,
        to_block=block,
        topics=[TRANSFER_EVENT_SIGNATURE]
    )
    for event in events:
        parse_and_store(event)
```

---

### 7. Database Test (5 dk)

```bash
# DuckDB connection test
duckdb db/crypto.db

# SQL prompt açılacak:
```

**Test queries:**
```sql
-- Tablo sayısı
SELECT COUNT(*) FROM information_schema.tables;

-- Transfer tablosu (henüz boş)
SELECT COUNT(*) FROM transfers;

-- Test insert
INSERT INTO transfers (
    block_number, tx_hash, from_addr, to_addr, value, token_addr
) VALUES (
    12345678,
    '0xabc...',
    '0xsender...',
    '0xreceiver...',
    1000000000000000000,  -- 1 ETH (wei)
    '0x0000000000000000000000000000000000000000'  -- ETH
);

-- Verify
SELECT * FROM transfers LIMIT 1;

-- Clean up test
DELETE FROM transfers WHERE tx_hash = '0xabc...';

-- Exit
.quit
```

**Beklenen:**
```
Insert: 1 row affected
Select: 1 row returned
Delete: 1 row affected
```

---

### 8. Wallet Report v0 (10 dk)

```bash
# Wallet raporu oluştur
python -m crypto.api.wallet_report --address 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb
```

**Beklenen output (JSON):**
```json
{
  "wallet": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
  "network": "Sepolia",
  "period": "24h",
  "summary": {
    "inbound": {
      "count": 12,
      "total_eth": 0.45,
      "total_usd": 780.50
    },
    "outbound": {
      "count": 8,
      "total_eth": 0.32,
      "total_usd": 550.20
    },
    "net_flow": {
      "eth": 0.13,
      "usd": 230.30
    }
  },
  "top_counterparties": [
    {
      "address": "0xabc...",
      "label": "Uniswap V3: Router",
      "interactions": 5
    },
    {
      "address": "0xdef...",
      "label": "Unknown",
      "interactions": 3
    }
  ],
  "timestamp": "2025-10-06T12:34:56Z"
}
```

**Ne yapıyor?**
```python
# Pseudo-code
def wallet_report(address, period="24h"):
    # 1. Son 24h transfer'leri çek
    transfers = db.query(
        "SELECT * FROM transfers 
         WHERE (from_addr = ? OR to_addr = ?)
         AND timestamp > NOW() - INTERVAL 24 HOURS",
        address, address
    )
    
    # 2. Inbound/outbound topla
    inbound = [t for t in transfers if t.to_addr == address]
    outbound = [t for t in transfers if t.from_addr == address]
    
    # 3. USD hesapla (price feed)
    prices = get_prices(['ETH', 'USDT', ...])
    
    # 4. Top counterparties
    counterparties = Counter(
        t.from_addr if t.to_addr == address else t.to_addr
        for t in transfers
    ).most_common(3)
    
    # 5. JSON döndür
    return {
        "wallet": address,
        "summary": {...},
        "top_counterparties": [...]
    }
```

---

## ✅ Week 0 Checklist

```bash
# Hepsini tek komutla test et
cd /Users/onur/code/novadev-protocol/crypto
python scripts/week0_check.py
```

**Beklenen output:**
```
=== NovaDev Crypto Week 0 Check ===

[✓] .env file exists
[✓] RPC_URL configured
[✓] RPC connection OK (latency: 145ms)
[✓] DuckDB schema loaded (4 tables)
[✓] Event capture test passed (156 events)
[✓] Wallet report skeleton works

Status: Week 0 COMPLETE ✓

Next: Week 1 (On-Chain Data Layer)
  - Event collector loop (30s polling)
  - Price feeds integration
  - Wallet report v1 (24h net flow)
```

---

## 🧯 Troubleshooting

### RPC Issues

**Problem: 429 Rate Limit**
```bash
# Çözüm 1: Daha yavaş poll et
# .env → POLL_INTERVAL_SECONDS=60

# Çözüm 2: Farklı RPC provider
# Ankr public endpoint (slower but free)
RPC_URL=https://rpc.ankr.com/eth_sepolia
```

**Problem: Timeout**
```bash
# Teşhis
curl -X POST $RPC_URL \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# Eğer cevap gelmiyorsa → RPC down
# Fallback RPC ekle
```

### Event Capture

**Problem: 0 events captured**
```bash
# Teşhis 1: Block range çok dar?
# Son 1000 blok dene:
python collector/event_capture.py --blocks 1000

# Teşhis 2: Filter yanlış?
# Tüm logları al (filtre yok):
python collector/event_capture.py --no-filter

# Teşhis 3: Testnet'te aktivite az?
# Mainnet'e geç (okuma sadece):
RPC_URL=https://eth-mainnet.g.alchemy.com/v2/...
```

### Database

**Problem: DuckDB file locked**
```bash
# Birden fazla process aynı DB'yi yazıyor

# Çözüm: Sadece 1 writer
# collector → write
# api → read-only

# DuckDB connection string:
# read-only mode
conn = duckdb.connect('db/crypto.db', read_only=True)
```

---

## 📚 Kaynaklar

**RPC Providers:**
- [Alchemy Docs](https://docs.alchemy.com)
- [Infura Docs](https://docs.infura.io)
- [Ankr RPC](https://www.ankr.com/rpc/)

**Web3.py:**
- [Quickstart](https://web3py.readthedocs.io/en/stable/quickstart.html)
- [Event Logs](https://web3py.readthedocs.io/en/stable/web3.eth.html#web3.eth.Eth.get_logs)

**DuckDB:**
- [Python API](https://duckdb.org/docs/api/python/overview)
- [SQL Reference](https://duckdb.org/docs/sql/introduction)

**Sepolia:**
- [Etherscan](https://sepolia.etherscan.io)
- [Faucet](https://sepoliafaucet.com)

---

## 🎯 Sonraki Adımlar

### Week 1 Preview
```
1. Event Collector Loop
   - 30s polling
   - Transfer + Swap events
   - Auto-recovery

2. Price Feeds
   - CoinGecko integration
   - 5 min cache
   - Fallback providers

3. Wallet Report v1
   - 24h net flow (USD)
   - Token breakdown
   - Gas spent
   - API endpoint: GET /wallet/<addr>/report
```

### Commit
```bash
cd /Users/onur/code/novadev-protocol
git add crypto/
git commit -m "crypto W0: setup complete - rpc health + event capture + report v0"
```

---

**Week 0 Tamamlandı! 🎉**

*Sonraki: Week 1 (On-Chain Data Layer)*
