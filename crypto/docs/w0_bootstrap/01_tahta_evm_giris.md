# 🧑‍🏫 Tahta 01 — Blockchain'i Okumak: EVM Veri Modeli Deep-Dive

> **Amaç:** "Kripto" deyince sihir değil, **zincirin kendisini okuyup** anlamlı bilgi çıkarmak. Block → Transaction → Receipt → Log → Event akışını **temelden üretim seviyesine** anlamak.
> **Mod:** Read-only (özel anahtar yok), testnet-first, **yatırım tavsiyesi değildir**.

---

## 🗺️ Plan (Detaylı Tahta)

1. **Blockchain nedir?** (Defter metaforu + gerçek mimari)
2. **EVM veri katmanları:** Block → Tx → Receipt → Log → Event
3. **Event-driven architecture** (Neden log'lar kritik?)
4. **ERC-20 Transfer anatomisi** (Solidity → Blockchain)
5. **JSON-RPC üçlüsü:** blockNumber, getBlock, getLogs
6. **İdempotent + Reorg + State** (Production essentials)
7. **Mini rapor hedefi:** 24h wallet summary
8. **Pratik örnekler** + kod şablonları
9. **Sık hatalar** + troubleshooting
10. **Quiz, ödevler, ve next steps**

---

## 1) Blockchain Nedir? (Defter Metaforundan Gerçeğe)

### 1.1 Basit Metafor: Defter

```
Blockchain = Bir defter
├── Her sayfa = Block (blok)
├── Sayfadaki satırlar = Transactions (işlemler)
└── Satır kenarındaki notlar = Logs/Events (olaylar)
```

**Örnek sayfa (Block 5,234,567):**
```
┌─────────────────────────────────────────────┐
│ Block #5,234,567                            │
│ Timestamp: 2025-10-06 14:33:05 UTC         │
│ Parent: #5,234,566                          │
├─────────────────────────────────────────────┤
│ Tx 1: Alice → Bob (1.5 ETH)                │
│   └─ Event: Transfer(Alice, Bob, 1.5)      │
│                                             │
│ Tx 2: Contract.swap(USDC → DAI)            │
│   ├─ Event: Transfer(User, Pool, 1000)     │
│   ├─ Event: Swap(...)                      │
│   └─ Event: Transfer(Pool, User, 950)      │
│                                             │
│ Tx 3: Token.mint(NewUser, 100)             │
│   └─ Event: Transfer(0x0, NewUser, 100)    │
└─────────────────────────────────────────────┘
```

### 1.2 Gerçek Mimari: Merkle Tree + State

```
Block Header (32 bytes)
├── parentHash        ← Önceki blok (chain continuity)
├── stateRoot         ← Account state Merkle root
├── transactionsRoot  ← Tx Merkle root
├── receiptsRoot      ← Receipt Merkle root (logs burada!)
├── timestamp         ← Unix epoch (seconds)
├── number            ← Block height
└── ... (difficulty, nonce, etc.)

Block Body
├── Transactions[]    ← Signed user actions
└── Uncles[]          ← Stale blocks (PoW artifact)

Receipts (separate, indexed by node)
└── Logs[]            ← Events emitted during execution
```

**Neden bu önemli?**
- **Logs zincirde yoktur!** (sadece receipt'lerde)
- Node'lar log'ları **indexler** (getLogs için)
- Reorg olunca **receipt'ler kaybolabilir** (son N blok)

---

## 2) EVM Veri Katmanları: Block → Tx → Receipt → Log → Event

### 2.1 Akış Şeması (Complete Flow)

```
┌─────────────────────────────────────────────────────────────┐
│                    User Action                              │
│          (wallet.transfer, contract.swap, etc.)             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRANSACTION (Tx)                         │
│  • from, to, value, data, nonce, gasPrice, signature       │
│  • Status: Pending → Mined                                 │
└────────────────────────┬────────────────────────────────────┘
                         ↓
           🔥 EVM Execution (State Change) 🔥
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    RECEIPT                                  │
│  • status (1=success, 0=revert)                            │
│  • gasUsed                                                  │
│  • contractAddress (if deployment)                         │
│  • logs[] ← ⭐ Event'ler burada!                           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    LOGS (Raw)                               │
│  • address (contract)                                       │
│  • topics[0..3] (indexed params)                           │
│  • data (non-indexed params)                               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    EVENTS (Decoded)                         │
│  Transfer(from, to, value)                                  │
│  Swap(sender, amount0In, amount1Out, ...)                  │
│  Approval(owner, spender, value)                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Her Katmanın Amacı

| Katman | Amaç | İçerik | Sorgulama |
|--------|------|--------|-----------|
| **Block** | Zaman + bağlam | Timestamp, parent, tx root | `eth_getBlockByNumber` |
| **Transaction** | Kullanıcı aksiyonu | from, to, value, data | `eth_getTransactionByHash` |
| **Receipt** | Sonuç + yan etkiler | status, gasUsed, **logs** | `eth_getTransactionReceipt` |
| **Log** | Kontrat olayı (ham) | address, topics, data | `eth_getLogs` ⭐ |
| **Event** | Log'un anlamlı hali | Decoded params | ABI + parsing |

### 2.3 Neden Log/Event Kritik?

**Problem:** "Bu cüzdana son 24 saatte ne girdi/çıktı?"

**Çözüm yolları:**

❌ **Kötü:** Her bloktaki her transaction'ı oku → parse et
```python
for block in range(start, end):
    txs = get_block(block)["transactions"]
    for tx in txs:
        if tx["to"] == my_wallet:
            # Parse value...
```
**Neden kötü?** 
- 1000 blok × 200 tx/blok = 200,000 HTTP call!
- Value transfer ≠ token transfer
- Internal transfers (contract → user) görünmez

✅ **İyi:** Event log'larını filtrele
```python
logs = eth_getLogs({
    "fromBlock": start,
    "toBlock": end,
    "topics": [TRANSFER_SIG, None, my_wallet_topic]  # to=my_wallet
})
# 1 HTTP call, node indexer kullanır, hızlı!
```

**Event-driven architecture:**
- Blockchain = **event stream** (immutable log)
- Off-chain analytics = event'leri consume et
- Real-time alerts = event'leri listen et

---

## 3) Event-Driven Architecture (Neden Bu Yaklaşım?)

### 3.1 Traditional vs Blockchain

**Traditional Database:**
```sql
-- State query
SELECT balance FROM accounts WHERE user_id = 123;

-- History query
SELECT * FROM transactions WHERE user_id = 123;
```

**Blockchain (Event Sourcing):**
```python
# State: Derived from events
events = get_transfer_events(wallet)
balance = sum(e.value for e in events if e.to == wallet) - \
          sum(e.value for e in events if e.from == wallet)

# History: Events themselves
history = events  # Immutable, append-only
```

### 3.2 Event Sourcing Benefits

✅ **Immutability:** Event'ler asla değişmez (audit trail)  
✅ **Replayability:** Event'leri tekrar oynatarak state'i rebuild et  
✅ **Transparency:** Her değişiklik izlenebilir  
✅ **Efficiency:** Node'lar event'leri indexler (fast query)

### 3.3 Blockchain'in Özel Durumu

**Challenge:** Blockchain event'ler **asenkron** ve **eventually consistent**

```
Block N-2     Block N-1     Block N   ← Chain tip
  ✅ Finalized   ✅ Likely      ⚠️ Pending (reorg risk)
                              
← Safe to index ─────────────┤├─ Buffer ─┤
```

**Çözüm:** **Reorg buffer** (CONFIRMATIONS)

---

## 4) ERC-20 Transfer Anatomisi: Solidity → Blockchain

### 4.1 Solidity Event Tanımı

```solidity
// ERC-20 Interface (EIP-20)
interface IERC20 {
    event Transfer(
        address indexed from,    // indexed → topic
        address indexed to,      // indexed → topic
        uint256 value            // non-indexed → data
    );
    
    function transfer(address to, uint256 amount) external returns (bool);
}

// Implementation example
contract MyToken is IERC20 {
    mapping(address => uint256) public balances;
    
    function transfer(address to, uint256 amount) external returns (bool) {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;  // State change
        balances[to] += amount;          // State change
        
        emit Transfer(msg.sender, to, amount);  // ⭐ Event emission
        
        return true;
    }
}
```

### 4.2 Blockchain'e Nasıl Yazılır?

**Event signature hesaplama:**
```python
import hashlib

def keccak256(text):
    """Ethereum'un keccak256'sı (SHA-3 varyantı)"""
    from Crypto.Hash import keccak
    k = keccak.new(digest_bits=256)
    k.update(text.encode())
    return "0x" + k.hexdigest()

# Transfer signature
sig = keccak256("Transfer(address,address,uint256)")
print(sig)
# 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef
```

**Log yapısına mapping:**
```
Solidity Event:
  Transfer(
    address indexed from,     ──→  topics[1]
    address indexed to,       ──→  topics[2]
    uint256 value             ──→  data
  )

Blockchain Log:
{
  "address": "0xTokenContract...",
  "topics": [
    "0xddf252ad...",           # topic[0] = signature (sabit)
    "0x000...from_address",    # topic[1] = from (32 byte padded)
    "0x000...to_address"       # topic[2] = to (32 byte padded)
  ],
  "data": "0x...value_hex"     # 32 byte uint256
}
```

### 4.3 Neden "indexed"? (Topics vs Data)

**Topics (indexed):**
```solidity
event Transfer(
    address indexed from,  // ← Filtrelenebilir!
    address indexed to     // ← Filtrelenebilir!
)
```

**getLogs filtresi:**
```python
# "Bu adresten çıkan tüm transferler"
logs = eth_getLogs({
    "topics": [
        TRANSFER_SIG,
        from_address_as_topic,  # ← Bu filtre çalışır!
        None  # to: any
    ]
})
```

**Data (non-indexed):**
```solidity
event Transfer(
    uint256 value  // ← Filtreleyemezsin!
)
```

**Trade-off:**
- Indexed: Filtrelenebilir ama max 3 param (+ signature = 4 topic)
- Non-indexed: Sınırsız ama filtreleyemezsin, tüm log'u çekip parse et

### 4.4 Görsel: ABI → Blockchain Dönüşümü

```
╔══════════════════════════════════════════════════════════════╗
║                    SOLIDITY (ABI)                            ║
╠══════════════════════════════════════════════════════════════╣
║ event Transfer(                                              ║
║   address indexed from,     // 0xAlice = 20 bytes           ║
║   address indexed to,       // 0xBob = 20 bytes             ║
║   uint256 value             // 1500000000000000000 (1.5e18) ║
║ )                                                            ║
╚═════════════════════════════╦════════════════════════════════╝
                              ↓
                    🔥 EVM Execution 🔥
                              ↓
╔══════════════════════════════════════════════════════════════╗
║                    BLOCKCHAIN (Log)                          ║
╠══════════════════════════════════════════════════════════════╣
║ address: 0x6B175474E89094C44Da98b954EedeAC495271d0F (DAI)    ║
║ topics[0]: 0xddf252ad1be2c89b69c2b068fc378daa952ba...       ║
║            ↑ keccak256("Transfer(address,address,uint256)") ║
║                                                              ║
║ topics[1]: 0x000000000000000000000000Alice1234567890...     ║
║            ↑ from = 0xAlice (padded to 32 bytes)            ║
║                                                              ║
║ topics[2]: 0x000000000000000000000000Bob1234567890...       ║
║            ↑ to = 0xBob (padded to 32 bytes)                ║
║                                                              ║
║ data: 0x00000000000000000000000000000000000000000000        ║
║       00014D1120D7B16000                                    ║
║       ↑ value = 1500000000000000000 (hex)                   ║
║       = 1.5 * 10^18 (with 18 decimals = 1.5 tokens)         ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 5) JSON-RPC Üçlüsü: Hayatta Kalma Kiti

### 5.1 Çekirdek 3 Komut

| Komut | Amaç | Kullanım | Sıklık |
|-------|------|----------|--------|
| `eth_blockNumber` | Sağlık + latency | Health check | Her 10s |
| `eth_getBlockByNumber` | Timestamp + metadata | Zaman çözümleme | İhtiyaç halinde |
| `eth_getLogs` | Event tarama | Veri çekme | Sürekli (loop) |

### 5.2 eth_blockNumber (Nabız Yoklama)

**Ne sağlar?**
- Node canlı mı?
- En son blok numarası?
- Network latency?

**Örnek implementation:**
```python
import time, requests

def check_health(rpc_url, timeout=10):
    """RPC health check with metrics"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_blockNumber",
        "params": []
    }
    
    t0 = time.perf_counter()
    try:
        r = requests.post(rpc_url, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}"}
        
        data = r.json()
        if "error" in data:
            return {"ok": False, "error": data["error"]}
        
        block_num = int(data["result"], 16)
        
        # Latency assessment
        status = "🟢" if latency_ms < 300 else \
                 "🟡" if latency_ms < 1000 else "🔴"
        
        return {
            "ok": True,
            "block": block_num,
            "latency_ms": round(latency_ms, 1),
            "status": status
        }
    
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Usage
result = check_health("https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY")
if result["ok"]:
    print(f"{result['status']} Block: {result['block']:,} | "
          f"Latency: {result['latency_ms']}ms")
else:
    print(f"❌ Health check failed: {result['error']}")
```

### 5.3 eth_getBlockByNumber (Zaman Makinesi)

**Ne sağlar?**
- Blok timestamp (Unix epoch)
- Parent hash (chain doğrulama)
- Transaction count

**Timestamp kullanımı:**
```python
def get_block_timestamp(rpc_url, block_number):
    """Get human-readable timestamp"""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "eth_getBlockByNumber",
        "params": [hex(block_number), False]  # False = tx hash only
    }
    
    r = requests.post(rpc_url, json=payload, timeout=10)
    block = r.json()["result"]
    
    ts_hex = block["timestamp"]
    ts_int = int(ts_hex, 16)  # Unix epoch (seconds)
    
    # Human-readable
    import time
    utc_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", 
                             time.gmtime(ts_int))
    
    return {
        "block": block_number,
        "timestamp": ts_int,
        "time_utc": utc_time,
        "tx_count": len(block["transactions"])
    }

# Usage
info = get_block_timestamp(rpc_url, 5_234_567)
print(f"Block {info['block']:,}")
print(f"  Time: {info['time_utc']}")
print(f"  Txs: {info['tx_count']}")
```

**Blok zamandan tahmin etme:**
```python
def estimate_block_at_time(rpc_url, target_timestamp, 
                           avg_block_time=12):
    """
    Belirli bir zamana denk gelen blok numarasını tahmin et
    
    Args:
        target_timestamp: Unix timestamp
        avg_block_time: Ortalama blok süresi (saniye)
                        Mainnet: ~12s, Sepolia: ~12s
    """
    # Latest block
    latest_num = int(
        requests.post(rpc_url, json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_blockNumber",
            "params": []
        }).json()["result"], 16
    )
    
    # Latest timestamp
    latest_block = requests.post(rpc_url, json={
        "jsonrpc": "2.0",
        "id": 2,
        "method": "eth_getBlockByNumber",
        "params": [hex(latest_num), False]
    }).json()["result"]
    
    latest_ts = int(latest_block["timestamp"], 16)
    
    # Calculate difference
    time_diff = latest_ts - target_timestamp
    block_diff = time_diff // avg_block_time
    
    estimated_block = latest_num - block_diff
    
    return max(0, estimated_block)

# Example: Blok 24 saat önce
target = int(time.time()) - (24 * 3600)
block_24h_ago = estimate_block_at_time(rpc_url, target)
print(f"~24h ago: Block #{block_24h_ago:,}")
```

### 5.4 eth_getLogs (Çekirdek İş)

**En kritik komut!** Event tarama için biricik yöntem.

**Temel kullanım:**
```python
def fetch_transfer_logs(rpc_url, start_block, end_block, 
                        token_address=None):
    """
    Transfer event'lerini çek
    
    Args:
        token_address: Spesifik token (None = tüm tokenlar)
    """
    TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
    
    filter_obj = {
        "fromBlock": hex(start_block),
        "toBlock": hex(end_block),
        "topics": [TRANSFER_SIG]
    }
    
    # Opsiyonel: Spesifik token
    if token_address:
        filter_obj["address"] = token_address.lower()
    
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "eth_getLogs",
        "params": [filter_obj]
    }
    
    r = requests.post(rpc_url, json=payload, timeout=30)
    return r.json()["result"]

# Usage
logs = fetch_transfer_logs(
    rpc_url, 
    start_block=5_230_000, 
    end_block=5_231_000,
    token_address="0x6B175474E89094C44Da98b954EedeAC495271d0F"  # DAI
)
print(f"Found {len(logs)} Transfer events")
```

**Pencere boyutu optimizasyonu:**
```python
def scan_logs_chunked(rpc_url, start, end, chunk_size=1500):
    """
    Büyük aralığı küçük parçalara böl
    
    chunk_size: Pencere boyutu (1000-2000 ideal)
    """
    all_logs = []
    
    for a in range(start, end + 1, chunk_size):
        b = min(a + chunk_size - 1, end)
        
        print(f"Scanning {a:,} → {b:,}...")
        
        logs = fetch_transfer_logs(rpc_url, a, b)
        all_logs.extend(logs)
        
        print(f"  Found: {len(logs)} logs")
        
        # Rate limit protection
        time.sleep(0.1)
    
    return all_logs

# Usage
logs = scan_logs_chunked(rpc_url, 5_230_000, 5_240_000)
print(f"Total: {len(logs)} logs")
```

---

## 6) İdempotent + Reorg + State (Production Essentials)

### 6.1 Idempotency (Tekrarlanabilirlik)

**Problem:** Script crash olursa veya tekrar çalıştırırsan **çift kayıt** olur

```python
# ❌ Naive approach
for log in logs:
    db.insert({
        "tx": log["transactionHash"],
        "from": parse_from(log),
        "to": parse_to(log),
        "value": parse_value(log)
    })
# Tekrar çalıştır → duplicate entries!
```

**Çözüm 1: UNIQUE constraint**
```sql
CREATE TABLE transfers (
    tx_hash TEXT,
    log_index INTEGER,
    ...
    UNIQUE(tx_hash, log_index)  -- ⭐ Idempotency key
);
```

**Çözüm 2: Anti-join pattern**
```python
# Staging table kullan
conn.execute("CREATE TEMP TABLE staging AS SELECT * FROM transfers WHERE 1=0")

# Stage'e yaz
for log in logs:
    conn.execute("INSERT INTO staging VALUES (...)")

# Anti-join ile sadece yeni kayıtları ekle
conn.execute("""
    INSERT INTO transfers
    SELECT s.*
    FROM staging s
    LEFT JOIN transfers t 
        ON t.tx_hash = s.tx_hash AND t.log_index = s.log_index
    WHERE t.tx_hash IS NULL
""")
```

### 6.2 Reorg Protection (Chain Reorganization)

**Problem:** Son N blok "pending", reorg olabilir

```
Before Reorg:
Block N-2 → Block N-1 → Block N
  ✅         ✅          ⚠️

After Reorg:
Block N-2 → Block N-1' → Block N'
  ✅         🔄 New!      🔄 New!
```

**Çözüm: Confirmation buffer**
```python
CONFIRMATIONS = 12  # Mainnet
# CONFIRMATIONS = 5   # Testnet (faster, riskier)

def get_safe_latest(rpc_url):
    """
    Safe block to process (confirmed)
    """
    latest = int(requests.post(rpc_url, json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_blockNumber",
        "params": []
    }).json()["result"], 16)
    
    safe_latest = latest - CONFIRMATIONS
    
    return {
        "latest": latest,
        "safe": safe_latest,
        "buffer": CONFIRMATIONS
    }

# Usage
blocks = get_safe_latest(rpc_url)
print(f"Latest: {blocks['latest']:,}")
print(f"Safe:   {blocks['safe']:,} (confirmed)")
print(f"Buffer: {blocks['buffer']} blocks")
```

**Reorg detection:**
```python
def detect_reorg(db, current_block_hash, block_number):
    """
    Reorg tespit et: stored hash ≠ current hash
    """
    stored = db.execute(
        "SELECT block_hash FROM blocks WHERE number = ?", 
        (block_number,)
    ).fetchone()
    
    if stored and stored[0] != current_block_hash:
        print(f"⚠️  REORG detected at block {block_number:,}")
        print(f"   Stored:  {stored[0][:10]}...")
        print(f"   Current: {current_block_hash[:10]}...")
        return True
    
    return False
```

### 6.3 State Tracking (Resume Capability)

**Problem:** Script durunca, kaldığın yerden devam et

**Çözüm: State table**
```sql
CREATE TABLE scan_state (
    key TEXT PRIMARY KEY,
    last_scanned_block INTEGER,
    updated_at TIMESTAMP
);
```

**Implementation:**
```python
def get_last_scanned_block(db, key="transfers_v1"):
    """Kaldığın yeri al"""
    row = db.execute(
        "SELECT last_scanned_block FROM scan_state WHERE key = ?", 
        (key,)
    ).fetchone()
    
    return row[0] if row else None

def set_last_scanned_block(db, block_num, key="transfers_v1"):
    """İlerlemeyi kaydet"""
    db.execute("""
        INSERT OR REPLACE INTO scan_state (key, last_scanned_block, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    """, (key, block_num))
    db.commit()

# Usage
last = get_last_scanned_block(db)
start = last + 1 if last else START_BLOCK

# Scan...
for a in range(start, safe_latest, CHUNK_SIZE):
    b = min(a + CHUNK_SIZE - 1, safe_latest)
    logs = fetch_logs(a, b)
    process_logs(logs)
    set_last_scanned_block(db, b)  # ⭐ Save progress
    
print(f"✅ Scanned up to block {b:,}")
```

---

## 7) Mini Rapor Hedefi: 24h Wallet Summary

### 7.1 Hedef Output

```json
{
  "wallet": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
  "window_hours": 24,
  "inbound": 125.5,
  "outbound": 73.2,
  "net_flow": 52.3,
  "tx_count": 18,
  "top_counterparties": [
    {"address": "0xUniswap...", "amount": 45.0},
    {"address": "0xAave...", "amount": 28.3},
    {"address": "0x1inch...", "amount": 12.5}
  ]
}
```

### 7.2 SQL Query (DuckDB)

```sql
-- 24h window
WITH recent AS (
  SELECT *
  FROM transfers
  WHERE block_time >= NOW() - INTERVAL 24 HOUR
    AND (LOWER(from_addr) = LOWER($wallet) 
         OR LOWER(to_addr) = LOWER($wallet))
),

-- Inbound/Outbound aggregation
agg AS (
  SELECT
    SUM(CASE WHEN LOWER(to_addr) = LOWER($wallet) 
             THEN value_unit ELSE 0 END) AS inbound,
    SUM(CASE WHEN LOWER(from_addr) = LOWER($wallet) 
             THEN value_unit ELSE 0 END) AS outbound,
    COUNT(*) AS tx_count
  FROM recent
),

-- Top counterparties
top_cp AS (
  SELECT
    CASE 
      WHEN LOWER(to_addr) = LOWER($wallet) THEN from_addr
      ELSE to_addr
    END AS counterparty,
    SUM(value_unit) AS total_amount
  FROM recent
  GROUP BY 1
  ORDER BY total_amount DESC
  LIMIT 3
)

SELECT * FROM agg;  -- Main stats
SELECT * FROM top_cp;  -- Top 3
```

### 7.3 Python Implementation

```python
import duckdb
from decimal import Decimal

def generate_wallet_report(db_path, wallet, hours=24):
    """
    Generate wallet activity report
    
    Returns dict with inbound/outbound/top counterparties
    """
    conn = duckdb.connect(db_path)
    wallet_lower = wallet.lower()
    
    # Main aggregation
    agg_query = """
    WITH recent AS (
      SELECT *
      FROM transfers
      WHERE block_time >= NOW() - INTERVAL ? HOUR
        AND (LOWER(from_addr) = ? OR LOWER(to_addr) = ?)
    )
    SELECT
      COALESCE(SUM(CASE WHEN LOWER(to_addr) = ? 
                        THEN value_unit ELSE 0 END), 0) AS inbound,
      COALESCE(SUM(CASE WHEN LOWER(from_addr) = ? 
                        THEN value_unit ELSE 0 END), 0) AS outbound,
      COUNT(*) AS tx_count
    FROM recent
    """
    
    agg = conn.execute(agg_query, [hours, wallet_lower, wallet_lower, 
                                    wallet_lower, wallet_lower]).fetchone()
    
    # Top counterparties
    top_query = """
    WITH recent AS (
      SELECT *
      FROM transfers
      WHERE block_time >= NOW() - INTERVAL ? HOUR
        AND (LOWER(from_addr) = ? OR LOWER(to_addr) = ?)
    )
    SELECT
      CASE WHEN LOWER(to_addr) = ? THEN from_addr ELSE to_addr END AS counterparty,
      SUM(value_unit) AS amount
    FROM recent
    GROUP BY 1
    ORDER BY amount DESC
    LIMIT 3
    """
    
    tops = conn.execute(top_query, [hours, wallet_lower, wallet_lower, 
                                     wallet_lower]).fetchall()
    
    # Build report
    inbound, outbound, tx_count = agg
    
    return {
        "wallet": wallet_lower,
        "window_hours": hours,
        "inbound": float(inbound or 0),
        "outbound": float(outbound or 0),
        "net_flow": float((inbound or 0) - (outbound or 0)),
        "tx_count": int(tx_count or 0),
        "top_counterparties": [
            {"address": addr, "amount": float(amt)}
            for addr, amt in tops
        ]
    }

# Usage
report = generate_wallet_report(
    "onchain.duckdb",
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    hours=24
)

import json
print(json.dumps(report, indent=2))
```

---

## 8) Pratik Örnekler + Kod Şablonları

### 8.1 Complete Ingest Pipeline

```python
#!/usr/bin/env python3
"""
Production-grade Transfer ingest pipeline
"""
import os, time, requests, duckdb
from pathlib import Path
from dotenv import load_dotenv

# Config
load_dotenv()
RPC_URL = os.getenv("RPC_URL")
DB_PATH = Path("onchain.duckdb")
TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
CONFIRMATIONS = 12
CHUNK_SIZE = 1500
STATE_KEY = "transfers_v1"

def init_db():
    """Initialize database schema"""
    conn = duckdb.connect(str(DB_PATH))
    
    # Transfers table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS transfers (
        block_number BIGINT,
        block_time TIMESTAMP,
        tx_hash TEXT,
        log_index INTEGER,
        token TEXT,
        from_addr TEXT,
        to_addr TEXT,
        raw_value DECIMAL(38,0),
        value_unit DOUBLE,
        UNIQUE(tx_hash, log_index)  -- Idempotency
    )
    """)
    
    # State tracking
    conn.execute("""
    CREATE TABLE IF NOT EXISTS scan_state (
        key TEXT PRIMARY KEY,
        last_scanned_block BIGINT,
        updated_at TIMESTAMP
    )
    """)
    
    # Indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_block ON transfers(block_number)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_from ON transfers(from_addr)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_to ON transfers(to_addr)")
    
    return conn

def rpc_call(method, params):
    """RPC helper"""
    r = requests.post(RPC_URL, json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }, timeout=30)
    
    r.raise_for_status()
    data = r.json()
    
    if "error" in data:
        raise RuntimeError(data["error"])
    
    return data["result"]

def get_safe_latest():
    """Get confirmed block number"""
    latest_hex = rpc_call("eth_blockNumber", [])
    latest = int(latest_hex, 16)
    return latest - CONFIRMATIONS

def get_block_timestamp(block_num):
    """Get block timestamp"""
    block = rpc_call("eth_getBlockByNumber", [hex(block_num), False])
    ts = int(block["timestamp"], 16)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))

def fetch_logs(start, end):
    """Fetch Transfer logs"""
    return rpc_call("eth_getLogs", [{
        "fromBlock": hex(start),
        "toBlock": hex(end),
        "topics": [TRANSFER_SIG]
    }])

def parse_log(log, decimals=18):
    """Parse Transfer log"""
    topics = log["topics"]
    
    from_addr = "0x" + topics[1][-40:]
    to_addr = "0x" + topics[2][-40:]
    raw_value = int(log["data"], 16)
    value_unit = raw_value / (10 ** decimals)
    
    return {
        "block_number": int(log["blockNumber"], 16),
        "tx_hash": log["transactionHash"].lower(),
        "log_index": int(log["logIndex"], 16),
        "token": log["address"].lower(),
        "from_addr": from_addr.lower(),
        "to_addr": to_addr.lower(),
        "raw_value": raw_value,
        "value_unit": value_unit
    }

def main():
    """Main ingest loop"""
    conn = init_db()
    
    # Get start block
    last = conn.execute(
        "SELECT last_scanned_block FROM scan_state WHERE key = ?",
        [STATE_KEY]
    ).fetchone()
    
    start = (last[0] + 1) if last else 5_000_000  # Sepolia start
    safe_latest = get_safe_latest()
    
    if start >= safe_latest:
        print(f"✅ Up to date: {start:,} >= {safe_latest:,}")
        return
    
    print(f"🔍 Scanning {start:,} → {safe_latest:,}")
    
    # Chunk scanning
    for a in range(start, safe_latest + 1, CHUNK_SIZE):
        b = min(a + CHUNK_SIZE - 1, safe_latest)
        
        print(f"   [{a:,} - {b:,}]", end=" ")
        
        logs = fetch_logs(a, b)
        
        if logs:
            # Get block timestamp (cache for batch)
            ts_cache = {}
            for log in logs:
                bn = int(log["blockNumber"], 16)
                if bn not in ts_cache:
                    ts_cache[bn] = get_block_timestamp(bn)
            
            # Parse & insert (idempotent)
            rows = []
            for log in logs:
                parsed = parse_log(log)
                bn = parsed["block_number"]
                rows.append((
                    bn,
                    ts_cache[bn],
                    parsed["tx_hash"],
                    parsed["log_index"],
                    parsed["token"],
                    parsed["from_addr"],
                    parsed["to_addr"],
                    parsed["raw_value"],
                    parsed["value_unit"]
                ))
            
            # Anti-join insert
            conn.execute("CREATE TEMP TABLE _staging AS SELECT * FROM transfers WHERE 1=0")
            conn.executemany("INSERT INTO _staging VALUES (?,?,?,?,?,?,?,?,?)", rows)
            conn.execute("""
                INSERT INTO transfers
                SELECT s.*
                FROM _staging s
                LEFT JOIN transfers t ON t.tx_hash = s.tx_hash AND t.log_index = s.log_index
                WHERE t.tx_hash IS NULL
            """)
            conn.execute("DROP TABLE _staging")
            
            print(f"✅ +{len(rows)} logs")
        else:
            print("⚪ 0 logs")
        
        # Update state
        conn.execute("""
            INSERT OR REPLACE INTO scan_state (key, last_scanned_block, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, [STATE_KEY, b])
        conn.commit()
        
        time.sleep(0.1)  # Rate limit
    
    print(f"✅ Done. Last scanned: {b:,}")

if __name__ == "__main__":
    main()
```

---

## 9) Sık Hatalar + Troubleshooting

### ❌ Hata 1: Decimals Unutmak
```python
# YANLIŞ
raw = 1000000000000000000
print(f"Value: {raw}")  # 1000000000000000000 (anlamsız!)

# DOĞRU
raw = 1000000000000000000
decimals = 18
value = raw / (10 ** decimals)
print(f"Value: {value} tokens")  # 1.0 (anlamlı!)
```

### ❌ Hata 2: Geniş getLogs Penceresi
```python
# YANLIŞ (timeout risk!)
logs = eth_getLogs({
    "fromBlock": "0x0",  # Genesis!
    "toBlock": "latest"  # Tüm chain!
})
# Result: Timeout / 429 / query too large

# DOĞRU (küçük parçalar)
for start in range(5_000_000, 5_100_000, 1500):
    end = min(start + 1499, 5_100_000)
    logs = eth_getLogs({"fromBlock": hex(start), "toBlock": hex(end)})
```

### ❌ Hata 3: Reorg Görmezden Gelmek
```python
# YANLIŞ
latest = get_block_number()
logs = scan_logs(latest - 1000, latest)  # Son bloklar pending!

# DOĞRU
latest = get_block_number()
safe = latest - CONFIRMATIONS
logs = scan_logs(safe - 1000, safe)
```

### ❌ Hata 4: Idempotency Key Eksik
```python
# YANLIŞ (aynı tx'te 2+ transfer olabilir!)
unique_key = log["transactionHash"]

# DOĞRU
unique_key = (log["transactionHash"], log["logIndex"])
```

### ❌ Hata 5: State Tracking Yok
```python
# YANLIŞ (her seferinde baştan)
scan_logs(START_BLOCK, latest)  # Crash olsa baştan!

# DOĞRU
last_scanned = get_last_scanned_block(db)
start = last_scanned + 1 if last_scanned else START_BLOCK
scan_logs(start, latest)
set_last_scanned_block(db, latest)  # ⭐ Save progress
```

---

## 10) Quiz + Ödevler

### Mini Quiz (8 Soru)

1. Blockchain'de log'lar nerede saklanır?
2. Event-driven architecture'ın 3 avantajı?
3. `eth_getLogs` neden `eth_getTransactionByHash`'ten daha verimli?
4. Transfer event'inde topic0 ne içerir?
5. İdempotency key neden `(tx_hash, log_index)` çifti?
6. Reorg buffer neden gerekli?
7. DuckDB anti-join pattern'i nasıl çalışır?
8. 24h wallet report için hangi SQL aggregate'leri kullanılır?

### Cevap Anahtarı

1. **Receipt'lerde** (block body'de değil); node'lar indexler
2. (a) Immutability, (b) Replayability, (c) Transparency
3. getLogs **filtrelenebilir** (topics), node index'i kullanır; tx-by-hash her tx'i tek tek sorgular
4. Event signature: `keccak256("Transfer(address,address,uint256)")`
5. Bir tx'te birden fazla log olabilir; `log_index` tekliği garanti eder
6. Son N blok **reorg** (fork) olabilir; confirmed block'lara odaklan
7. Staging table → LEFT JOIN → WHERE main.key IS NULL → INSERT (sadece yeni kayıtlar)
8. `SUM(CASE WHEN ...)`, `COUNT(*)`, `GROUP BY counterparty`, `LIMIT 3`

### Ödevler (4 Pratik)

#### Ödev 1: Health Monitor
```python
# 5 dakika boyunca her 10 saniyede blockNumber çağır
# Latency'leri kaydet (CSV)
# Min/max/avg latency hesapla
```

#### Ödev 2: Block Time Analysis
```python
# Son 1000 bloğun timestamp'lerini al
# Ortalama blok süresini hesapla (saniye)
# Histogram çiz (matplotlib)
```

#### Ödev 3: Mini Ingest
```python
# 1000 blok Transfer log'u çek
# DuckDB'ye yaz (idempotent)
# Tekrar çalıştır → çift kayıt olmamalı
```

#### Ödev 4: Wallet Report
```python
# Bir cüzdan seç (Sepolia testnet)
# 24h raporu üret (JSON format)
# Top 3 counterparty'yi validate et
```

---

## 11) Terimler Sözlüğü

| Terim | Tanım |
|-------|-------|
| **Block** | Zincirdeki sıralı sayfa; timestamp + tx'ler |
| **Transaction** | Kullanıcı aksiyonu (signed) |
| **Receipt** | Tx sonucu; status + gasUsed + **logs** |
| **Log** | Kontrat event'i (ham); topics + data |
| **Event** | Log'un decoded hali (ABI ile) |
| **Topic** | Indexed event parametresi (max 3 + signature) |
| **Data** | Non-indexed parametreler |
| **Event Sourcing** | State'i event'lerden türet |
| **Idempotency** | Tekrar çalıştırılabilirlik (aynı sonuç) |
| **Reorg** | Chain reorganization (fork) |
| **Confirmation** | Block kesinleşme seviyesi |
| **State Tracking** | İlerleme kaydetme (resume) |

---

## 🔗 İlgili Kaynaklar

### Repo Kodları
- `crypto/w0_bootstrap/rpc_health.py` → blockNumber + latency
- `crypto/w0_bootstrap/capture_transfers_idempotent.py` → Full ingest pipeline
- `crypto/w0_bootstrap/report_json.py` → Wallet report generator

### Next Lessons
- **→ Tahta 02:** [JSON-RPC 101](02_tahta_rpc_101.md) (detaylı RPC komutları)
- **→ Tahta 03:** [Transfer Anatomisi](03_tahta_transfer_anatomi.md) (topics/data parsing)
- **→ Tahta 04:** [getLogs + Reorg](04_tahta_getlogs_pencere_reorg.md) (Coming)

### External
- **Ethereum JSON-RPC:** https://ethereum.org/en/developers/docs/apis/json-rpc/
- **ERC-20 Standard:** https://eips.ethereum.org/EIPS/eip-20
- **DuckDB SQL:** https://duckdb.org/docs/

---

## 🛡️ Güvenlik / Etik

- **Read-only:** Özel anahtar yok, imza yok, custody yok
- **`.env` hygiene:** Secrets asla commit etme
- **Testnet-first:** Sepolia ile başla
- **Eğitim amaçlı:** Yatırım tavsiyesi değildir

---

## 🔗 Navigasyon

- **→ Sonraki:** [02 - JSON-RPC 101](02_tahta_rpc_101.md)
- **↑ İndeks:** [W0 Tahta Serisi](README.md)

---

**Tahta 01 — Blockchain'i Okumak: EVM Veri Modeli**  
*Format: Temelden Production-Ready*  
*Süre: 60-75 dk (Geliştirilmiş Versiyon)*  
*Prerequisite: Yok (sıfırdan başlar)*  
*Versiyon: 2.0 (Complete Rewrite)*