# 🧑‍🏫 Tahta 02 — JSON-RPC 101: Node ile Konuşma Sanatı

> **Amaç:** EVM node'uyla **3 temel çağrı** üzerinden profesyonelce konuşmayı öğrenmek: **sağlık**, **zaman**, **olay tarama**.
> **Mod:** Read-only, testnet-first (Sepolia), **yatırım tavsiyesi değildir**.

---

## 🗺️ Plan (tahta)

1. JSON-RPC anatomisi: protokol yapısı ve felsefesi
2. **`eth_blockNumber`** → Sağlık kontrolü + latency ölçümü
3. **`eth_getBlockByNumber`** → Zaman damgası ve blok metadata
4. **`eth_getLogs`** → Event tarama: filtre, pencere, optimizasyon
5. Rate limiting & backoff stratejileri (gerçek dünya)
6. Error handling & troubleshooting senaryoları
7. Performans optimizasyonu: batch vs sequential
8. Laboratuvar: curl, httpie, Python örnekleri
9. Mini quiz & pratik ödevler

---

## 1) JSON-RPC Anatomisi (protokol yapısı)

### Temel Paket Yapısı

```json
{
  "jsonrpc": "2.0",        // Protokol versiyonu (sabit)
  "id": 1,                 // İstek-yanıt eşleştirme (number/string)
  "method": "eth_methodName",  // Çağrılacak fonksiyon
  "params": []             // Parametreler (array)
}
```

**Yanıt (başarılı):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "0x..."      // Sonuç (method'a göre tip değişir)
}
```

**Yanıt (hata):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found"
  }
}
```

### Kritik Kurallar

1. **Hex encoding:** Sayılar genelde `0x` ile başlar (16'lık)
   ```
   0x10    → 16 (decimal)
   0x100   → 256
   0xA     → 10
   ```

2. **ID tracking:** Async durumda yanıtları eşleştirmek için
   ```python
   request_id = 42
   response.id == request_id  # True olmalı
   ```

3. **Timeout:** Her RPC çağrısı için **maksimum süre** belirle
   ```python
   requests.post(url, json=payload, timeout=10)  # 10 saniye
   ```

### Akış Şeması

```
Client                          Node
  |                              |
  |---(1) POST: eth_method------>|
  |                              |
  |    (2) Processing...         |
  |                              |
  |<--(3) Response: result-------|
  |                              |
```

---

## 2) `eth_blockNumber` — Sağlık Kontrolü ve Latency

### Ne İşe Yarar?

✅ **Node canlı mı?** (connectivity check)  
✅ **Güncel blok numarası** (sync durumu)  
✅ **Latency ölçümü** (ms bazında)  
✅ **RPC provider kalitesi** (benchmark)

### İstek Yapısı

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "eth_blockNumber",
  "params": []                    // Parametre yok!
}
```

### Dönüş Örneği

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "0x12CDEF0"          // Hex formatında blok numarası
}
```

**Decimal'e çevir:**
```python
block_hex = "0x12CDEF0"
block_num = int(block_hex, 16)  # 19,718,640
```

### Pratik Örnek: Health Check Script

```python
#!/usr/bin/env python3
import os, time, requests
from dotenv import load_dotenv

load_dotenv()
RPC_URL = os.getenv("RPC_URL")

def health_check():
    """RPC node sağlık kontrolü + latency"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_blockNumber",
        "params": []
    }
    
    # Latency ölçümü
    t0 = time.perf_counter()
    try:
        r = requests.post(RPC_URL, json=payload, timeout=10)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        r.raise_for_status()  # HTTP hata kontrolü
        data = r.json()
        
        if "error" in data:
            print(f"❌ RPC Error: {data['error']}")
            return False
        
        block_hex = data["result"]
        block_num = int(block_hex, 16)
        
        # Latency değerlendirme
        status = "🟢" if latency_ms < 300 else "🟡" if latency_ms < 1000 else "🔴"
        
        print(f"{status} RPC OK | Block: {block_num:,} | Latency: {latency_ms:.1f} ms")
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Timeout: RPC yanıt vermiyor")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")
        return False

if __name__ == "__main__":
    health_check()
```

**Çıktı örnekleri:**
```
🟢 RPC OK | Block: 5,234,567 | Latency: 142.3 ms    # İyi
🟡 RPC OK | Block: 5,234,567 | Latency: 678.9 ms    # Yavaş
🔴 RPC OK | Block: 5,234,567 | Latency: 1,523.1 ms  # Çok yavaş
```

### Latency Benchmark (Referans)

| Provider Type | Expected Latency | Notes |
|--------------|------------------|-------|
| Alchemy Free | 100-300 ms | Normal |
| Infura Free | 150-400 ms | Değişken |
| Public RPC | 500-2000 ms | Güvenilmez |
| Local Node | 5-50 ms | Ideal ama ağır |

---

## 3) `eth_getBlockByNumber` — Zaman ve Metadata

### Ne İşe Yarar?

✅ **Timestamp** (Unix epoch, saniye)  
✅ **Block hash** (kimlik)  
✅ **Parent hash** (zincir doğrulama)  
✅ **Transaction count** (blok yoğunluğu)

### İstek Yapısı

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "eth_getBlockByNumber",
  "params": [
    "0x12CDEF0",    // Block number (hex) veya "latest"
    false           // false = tx hash'leri, true = full tx objesi
  ]
}
```

**Parametre seçenekleri:**
- `"latest"` → En son blok
- `"earliest"` → Genesis blok
- `"pending"` → Henüz mine edilmemiş (çoğu node desteklemez)
- `"0x..."` → Spesifik blok numarası (hex)

### Dönüş Örneği (kırpılmış)

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "number": "0x12cdef0",
    "hash": "0x4e3a3754410177e6937ef1f84bba68ea139e8d1a2258c5f85db9f1cd715a1bdd",
    "parentHash": "0x7501a4c...0e8a",
    "timestamp": "0x66f2a4b1",        // ⭐ Unix epoch (hex)
    "gasUsed": "0x1234567",
    "baseFeePerGas": "0x7",
    "transactions": [                 // false ise sadece hash'ler
      "0xabc...",
      "0xdef..."
    ]
  }
}
```

### Timestamp Çözümleme

```python
import time

def parse_block_time(block_data):
    """Block timestamp'ı insan-okur formata çevir"""
    ts_hex = block_data["timestamp"]
    ts_int = int(ts_hex, 16)  # Unix epoch (saniye)
    
    # UTC formatı
    utc_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts_int))
    
    # Lokal zaman
    local_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_int))
    
    return {
        "unix": ts_int,
        "utc": utc_str,
        "local": local_str
    }

# Örnek kullanım
block_time = parse_block_time({"timestamp": "0x66f2a4b1"})
print(f"Unix: {block_time['unix']}")      # 1727275185
print(f"UTC:  {block_time['utc']}")       # 2025-10-06 08:33:05 UTC
print(f"Local: {block_time['local']}")    # 2025-10-06 11:33:05 (TR)
```

### Pratik Kullanım: Blok Zaman Aralığı Hesaplama

```python
def estimate_block_by_time(target_time, avg_block_time=12):
    """
    Belirli bir zamana denk gelen blok numarasını tahmin et
    
    Args:
        target_time: Unix timestamp (int)
        avg_block_time: Ortalama blok süresi (saniye)
    """
    # En son bloğu al
    latest_block = get_block("latest")
    latest_num = int(latest_block["number"], 16)
    latest_time = int(latest_block["timestamp"], 16)
    
    # Zaman farkı
    time_diff = latest_time - target_time
    
    # Blok farkı tahmini
    block_diff = time_diff // avg_block_time
    
    # Hedef blok
    estimated_block = latest_num - block_diff
    
    return max(0, estimated_block)

# Örnek: 24 saat önceki blok?
target = int(time.time()) - (24 * 3600)
block_24h_ago = estimate_block_by_time(target)
print(f"~24h önce: Block #{block_24h_ago:,}")
```

---

## 4) `eth_getLogs` — Event Tarama (Çekirdek İş)

### Ne İşe Yarar?

✅ **Event log'ları çek** (Transfer, Swap, Mint, vb.)  
✅ **Belirli adres/kontrattan** filtrele  
✅ **Belirli topic (event signature)** ile ara  
✅ **Blok aralığı** belirle

### Filter Objesi Anatomisi

```json
{
  "fromBlock": "0x12CDA00",           // Başlangıç bloğu
  "toBlock": "0x12CDFEF",             // Bitiş bloğu
  "address": "0xTokenAddress",        // Opsiyonel: kontrat adresi
  "topics": [                         // Opsiyonel: event filtreleme
    "0xddf252ad1be2c89b...",          // topic0: event signature
    null,                             // topic1: any (veya spesifik adres)
    "0x000...UserAddress"             // topic2: spesifik değer
  ]
}
```

### ERC-20 Transfer Event Signature

```solidity
// Solidity event tanımı
event Transfer(address indexed from, address indexed to, uint256 value);
```

**Keccak256 hash:**
```
keccak256("Transfer(address,address,uint256)")
= 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef
```

### Topics Yapısı

```
topics[0] = Event signature (sabit)
topics[1] = from (indexed parametre)
topics[2] = to (indexed parametre)
data      = value (non-indexed, 32 byte)
```

**Görsel:**
```
Log Structure:
┌─────────────────────────────────────────┐
│ address: 0xTokenContract                │
│ blockNumber: 0x12CDEF0                  │
│ transactionHash: 0xabc123...            │
│ logIndex: 0x5                           │
├─────────────────────────────────────────┤
│ topics[0]: 0xddf252ad... (Transfer sig) │ ← Event imzası
│ topics[1]: 0x000...Alice                │ ← from (32 byte, son 20 adres)
│ topics[2]: 0x000...Bob                  │ ← to
├─────────────────────────────────────────┤
│ data: 0x00...64 hex chars               │ ← value (256 bit)
└─────────────────────────────────────────┘
```

### Pencere Boyutu Stratejisi (Kritik!)

**Problem:** Geniş aralık → timeout / rate limit

```python
# ❌ Kötü: 100k blok bir seferde
filter_bad = {
    "fromBlock": "0x100000",
    "toBlock": "0x200000"   # 100,000 blok!
}
# Result: 429 Too Many Requests veya timeout

# ✅ İyi: 1-2k blok parçalara böl
STEP = 1500  # Optimal pencere boyutu

def scan_range(start, end):
    for a in range(start, end + 1, STEP):
        b = min(a + STEP - 1, end)
        filter_obj = {
            "fromBlock": hex(a),
            "toBlock": hex(b),
            "topics": [TRANSFER_SIG]
        }
        logs = eth_getLogs(filter_obj)
        yield logs
        time.sleep(0.05)  # Rate limit koruma
```

### Reorg Buffer (Kesinleşme Tamponu)

```python
CONFIRMATIONS = 12  # Mainnet için
# CONFIRMATIONS = 5   # Testnet için

def get_safe_latest(latest_block):
    """
    Son N blok 'pending' sayılır (reorg riski)
    """
    return latest_block - CONFIRMATIONS

# Kullanım
latest = get_block_number()           # 5,234,567
safe = get_safe_latest(latest)        # 5,234,555 (12 blok buffer)
```

**Neden?**
```
Block N-2  Block N-1  Block N  ← Chain tip (reorg olabilir)
   ✅         ✅        ⚠️
              ↑
         Safe point
```

### Pratik Örnek: Transfer Tarama

```python
import os, time, requests
from dotenv import load_dotenv

load_dotenv()
RPC_URL = os.getenv("RPC_URL")
TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

def eth_getLogs(filter_obj, timeout=20):
    """getLogs çağrısı + error handling"""
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "eth_getLogs",
        "params": [filter_obj]
    }
    
    try:
        r = requests.post(RPC_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        
        if "error" in data:
            code = data["error"].get("code")
            msg = data["error"].get("message")
            print(f"⚠️  RPC Error [{code}]: {msg}")
            return []
        
        return data["result"]
    
    except requests.exceptions.Timeout:
        print("⏰ Timeout: Pencereyi küçült")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def scan_transfers(start_block, end_block, step=1500):
    """Blok aralığını parçalara bölüp tara"""
    all_logs = []
    
    for a in range(start_block, end_block + 1, step):
        b = min(a + step - 1, end_block)
        
        filter_obj = {
            "fromBlock": hex(a),
            "toBlock": hex(b),
            "topics": [TRANSFER_SIG]
        }
        
        print(f"📡 Scanning {a:,} → {b:,}...")
        logs = eth_getLogs(filter_obj)
        
        if logs:
            all_logs.extend(logs)
            print(f"   ✅ Found {len(logs)} transfers")
        else:
            print(f"   ⚪ No transfers")
        
        time.sleep(0.1)  # Rate limit koruması
    
    return all_logs

# Kullanım
latest = 5_234_567
safe_latest = latest - 12
start = safe_latest - 5_000

transfers = scan_transfers(start, safe_latest)
print(f"\n📊 Total: {len(transfers)} transfer events")
```

---

## 5) Rate Limiting & Backoff Stratejileri

### Tipik Rate Limitler

| Provider | Free Tier | Paid Tier |
|----------|-----------|-----------|
| Alchemy | 330 req/s | Sınırsız |
| Infura | 100k req/day | 100M req/day |
| Public | Değişken | N/A |

### HTTP 429: Too Many Requests

```python
def eth_getLogs_with_retry(filter_obj, max_retries=3):
    """Exponential backoff ile retry"""
    for attempt in range(max_retries):
        try:
            r = requests.post(RPC_URL, json=payload, timeout=20)
            
            if r.status_code == 429:
                # Rate limit hit
                backoff = 2 ** attempt  # 1, 2, 4, 8...
                print(f"⏳ Rate limit (429), waiting {backoff}s...")
                time.sleep(backoff)
                continue
            
            r.raise_for_status()
            return r.json()["result"]
        
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    return []
```

### Token Bucket Pattern

```python
import time
from collections import deque

class RateLimiter:
    """
    Token bucket rate limiter
    Örnek: 10 req/s max
    """
    def __init__(self, max_rate=10, period=1.0):
        self.max_rate = max_rate
        self.period = period
        self.timestamps = deque()
    
    def acquire(self):
        """Bir token al (gerekirse bekle)"""
        now = time.time()
        
        # Eski timestamp'leri temizle
        while self.timestamps and now - self.timestamps[0] > self.period:
            self.timestamps.popleft()
        
        # Limit doldu mu?
        if len(self.timestamps) >= self.max_rate:
            sleep_time = self.period - (now - self.timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            return self.acquire()  # Recursive
        
        self.timestamps.append(now)

# Kullanım
limiter = RateLimiter(max_rate=5, period=1.0)  # 5 req/s

for i in range(20):
    limiter.acquire()
    print(f"Request {i+1}")
    # ... RPC call
```

---

## 6) Error Handling & Troubleshooting

### Yaygın Hatalar ve Çözümleri

#### A) "Invalid JSON response"

**Neden:** RPC URL yanlış veya node down
```python
# ❌ Hatalı
RPC_URL = "https://eth-mainnet.alchemyapi.io/v2/"  # Key eksik

# ✅ Doğru
RPC_URL = "https://eth-mainnet.alchemyapi.io/v2/YOUR_API_KEY"
```

#### B) "execution reverted"

**Neden:** Hatalı kontrat çağrısı (getLogs için nadır, ama olabilir)
```python
# getLogs: kontrat adresi yanlış yazılmış
"address": "0xINVALID"  # ❌ Checksum hatası

# Çözüm: lowercase kullan
"address": "0xabcd...".lower()  # ✅
```

#### C) "query returned more than 10000 results"

**Neden:** Pencere çok geniş
```python
# ❌ 50k blok
filter_obj = {"fromBlock": "0x0", "toBlock": "0xC350"}

# ✅ 1.5k blok parçalar
for start in range(0, 50_000, 1500):
    ...
```

#### D) Timeout

**Neden:** Yavaş RPC veya ağ problemi
```python
# Timeout değerini artır + pencere küçült
requests.post(url, json=payload, timeout=30)  # 30s

# Veya provider değiştir
```

### Debug Checklist

```python
def debug_rpc_call(method, params):
    """RPC çağrısını debug et"""
    print(f"🔍 Debug: {method}")
    print(f"   URL: {RPC_URL[:50]}...")
    print(f"   Params: {params}")
    
    t0 = time.time()
    try:
        r = requests.post(RPC_URL, json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }, timeout=10)
        
        latency = (time.time() - t0) * 1000
        print(f"   Status: {r.status_code}")
        print(f"   Latency: {latency:.1f} ms")
        print(f"   Response size: {len(r.content)} bytes")
        
        if r.status_code != 200:
            print(f"   ❌ HTTP Error: {r.text[:200]}")
            return None
        
        data = r.json()
        if "error" in data:
            print(f"   ❌ RPC Error: {data['error']}")
            return None
        
        print(f"   ✅ Success")
        return data["result"]
    
    except Exception as e:
        print(f"   💥 Exception: {e}")
        return None
```

---

## 7) Performans Optimizasyonu

### Batch Requests (Gelişmiş)

```python
def batch_getLogs(filters):
    """
    Birden fazla getLogs'u tek HTTP call'da gönder
    Not: Tüm RPC provider'lar desteklemez
    """
    batch_payload = [
        {
            "jsonrpc": "2.0",
            "id": i,
            "method": "eth_getLogs",
            "params": [flt]
        }
        for i, flt in enumerate(filters)
    ]
    
    r = requests.post(RPC_URL, json=batch_payload, timeout=30)
    return r.json()  # List of responses

# Örnek
filters = [
    {"fromBlock": "0x100", "toBlock": "0x200", "topics": [SIG]},
    {"fromBlock": "0x200", "toBlock": "0x300", "topics": [SIG]},
    {"fromBlock": "0x300", "toBlock": "0x400", "topics": [SIG]},
]
results = batch_getLogs(filters)
```

### Concurrent Requests (Threading)

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_scan(block_ranges):
    """Paralel blok tarama (thread pool)"""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(scan_single_range, start, end): (start, end)
            for start, end in block_ranges
        }
        
        all_logs = []
        for future in as_completed(futures):
            start, end = futures[future]
            try:
                logs = future.result()
                all_logs.extend(logs)
                print(f"✅ {start}-{end}: {len(logs)} logs")
            except Exception as e:
                print(f"❌ {start}-{end}: {e}")
        
        return all_logs

# Kullanım
ranges = [
    (1000, 2500),
    (2500, 4000),
    (4000, 5500),
    (5500, 7000),
]
logs = parallel_scan(ranges)
```

---

## 8) Laboratuvar: Pratik Örnekler

### A) curl (Basit)

```bash
# blockNumber
curl -s -X POST $RPC_URL \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}'

# getBlockByNumber (latest)
curl -s -X POST $RPC_URL \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"eth_getBlockByNumber","params":["latest",false]}'

# getLogs (son 100 blok)
curl -s -X POST $RPC_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc":"2.0",
    "id":3,
    "method":"eth_getLogs",
    "params":[{
      "fromBlock":"0x4F0000",
      "toBlock":"0x4F0064",
      "topics":["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"]
    }]
  }' | jq '.result | length'  # Log count
```

### B) httpie (Okunabilir)

```bash
# blockNumber
http POST $RPC_URL \
  jsonrpc=2.0 \
  id:=1 \
  method=eth_blockNumber \
  params:='[]'

# getLogs
http POST $RPC_URL \
  jsonrpc=2.0 \
  id:=3 \
  method=eth_getLogs \
  params:='[{"fromBlock":"0x4F0000","toBlock":"0x4F0064","topics":["0xddf25..."]}]'
```

### C) Python (Production-Ready)

```python
#!/usr/bin/env python3
"""
Production-grade RPC client örneği
"""
import os, time, requests, json
from typing import Optional, Dict, List
from dotenv import load_dotenv

load_dotenv()

class EVMClient:
    """EVM RPC client with error handling & rate limiting"""
    
    def __init__(self, rpc_url: str, timeout: int = 20):
        self.rpc_url = rpc_url
        self.timeout = timeout
        self._req_count = 0
    
    def _call(self, method: str, params: list) -> Optional[Dict]:
        """Base RPC call"""
        payload = {
            "jsonrpc": "2.0",
            "id": self._req_count + 1,
            "method": method,
            "params": params
        }
        
        try:
            r = requests.post(
                self.rpc_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            self._req_count += 1
            
            if r.status_code == 429:
                raise RateLimitError("Too many requests")
            
            r.raise_for_status()
            data = r.json()
            
            if "error" in data:
                raise RPCError(data["error"])
            
            return data["result"]
        
        except requests.exceptions.Timeout:
            raise TimeoutError(f"RPC timeout after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"RPC connection error: {e}")
    
    def block_number(self) -> int:
        """Get latest block number"""
        result = self._call("eth_blockNumber", [])
        return int(result, 16)
    
    def get_block(self, block_id: str = "latest", full_tx: bool = False) -> Dict:
        """Get block by number"""
        return self._call("eth_getBlockByNumber", [block_id, full_tx])
    
    def get_logs(self, filter_obj: Dict) -> List[Dict]:
        """Get logs with filter"""
        return self._call("eth_getLogs", [filter_obj])

# Custom exceptions
class RPCError(Exception):
    pass

class RateLimitError(Exception):
    pass

# Kullanım
if __name__ == "__main__":
    client = EVMClient(os.getenv("RPC_URL"))
    
    # Latest block
    latest = client.block_number()
    print(f"Latest block: {latest:,}")
    
    # Block info
    block = client.get_block("latest")
    print(f"Timestamp: {int(block['timestamp'], 16)}")
    
    # Logs (son 1000 blok)
    logs = client.get_logs({
        "fromBlock": hex(latest - 1000),
        "toBlock": hex(latest),
        "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"]
    })
    print(f"Transfer count: {len(logs)}")
```

---

## 9) Mini Quiz

### Sorular

1. `eth_blockNumber` çağrısı hangi 3 bilgiyi sağlar?
2. `eth_getBlockByNumber`'da 2. parametrenin `false` olması ne demek?
3. `eth_getLogs` için ideal pencere boyutu nedir ve neden?
4. Transfer event'inde `topics[0]` neyi temsil eder?
5. Rate limit (429) aldığında ne yapmalısın?
6. Reorg buffer nedir ve neden gerekli?
7. `value_unit` nasıl hesaplanır?
8. Batch request'in avantajı nedir?

### Cevap Anahtarı

1. **Node canlılığı**, **en son blok numarası**, **latency (ms)**
2. İşlem objelerini değil, sadece **hash'lerini** getir (hafif ve hızlı)
3. **1000-2000 blok** idealdir; timeout ve rate limit riskini dengeler
4. Event signature'ı (`keccak256("Transfer(address,address,uint256)")`)
5. **Exponential backoff** uygula: 1s, 2s, 4s, 8s bekle ve tekrar dene
6. Son N blok **reorg** (chain reorganization) olabilir; **safe_latest** kullan
7. `raw_value / (10 ** decimals)` — örn: `1000000000000000000 / 10^18 = 1.0`
8. **Tek HTTP call**'da birden fazla request gönder → latency azalır

---

## 10) Ödevler (Pratik)

### Ödev 1: Latency Benchmark
```python
# 3 farklı RPC provider'ı test et
# Her biri için 10 kez blockNumber çağır
# Ortalama latency'i karşılaştır
```

### Ödev 2: Block Time Analysis
```python
# Son 1000 bloğun timestamp'lerini al
# Ortalama blok süresini hesapla (saniye)
# Min/max blok sürelerini bul
```

### Ödev 3: getLogs Window Experiment
```python
# Aynı aralığı 3 farklı pencere ile tara:
# - 500 blok
# - 1500 blok
# - 5000 blok
# Süre ve başarı oranını kaydet
```

### Ödev 4: Error Handling
```python
# Kasıtlı hatalı parametreler gönder:
# - Geçersiz hex format
# - Çok geniş pencere
# - Yanlış method adı
# Error response'ları incele
```

---

## 11) Terimler Sözlüğü

* **JSON-RPC:** Remote Procedure Call protokolü (JSON format)
* **Hex:** 16'lık sayı sistemi (`0x` prefix)
* **Latency:** İstek-yanıt geçen süre (ms)
* **Topic:** Event log'unda indexed alan + signature (topic0)
* **Filter object:** getLogs parametreleri (fromBlock, toBlock, address, topics)
* **Reorg:** Chain reorganization (fork, rollback)
* **Buffer:** Güvenlik tamponu (son N blok)
* **Rate limit:** Birim zamandaki maksimum istek sayısı
* **Backoff:** Hata sonrası bekleme stratejisi (exponential)
* **Batch request:** Tek HTTP'de birden fazla RPC call
* **Idempotent:** Tekrar çalıştırılabilir (aynı sonuç)

---

## 12) Best Practices (Özet)

✅ **Timeout kullan** (10-30s)  
✅ **Pencereyi küçük tut** (1-2k blok)  
✅ **Reorg buffer** ekle (5-12 blok)  
✅ **Rate limit** koru (sleep + backoff)  
✅ **Error handle** et (try-except)  
✅ **Hex → int** dönüşümü dikkatli  
✅ **Decimals** unutma (`value / 10^18`)  
✅ **Logging** ekle (debug için)  
✅ **Retry logic** yaz (429, timeout)  
✅ **Test et** (küçük aralıklarla başla)

---

## 🔗 İlgili Kodlar (Repo)

* `crypto/w0_bootstrap/rpc_health.py` → blockNumber + latency ölçümü
* `crypto/w0_bootstrap/capture_transfers*.py` → getLogs + pencere/parçalama
* `crypto/w0_bootstrap/report_json.py` → 24h rapor (JSON)

---

## 🛡️ Güvenlik / Etik

* **Read-only:** Özel anahtar yok, imza yok, custody yok
* **`.env` hygiene:** RPC URL'leri asla commit etme
* **Testnet-first:** Sepolia ile başla, mainnet sonrası
* **Eğitim amaçlı:** Yatırım tavsiyesi değildir

---

## 🔗 Navigasyon

* **← Önceki:** [01 - EVM Veri Modeli](01_tahta_evm_giris.md)
* **→ Sonraki:** [03 - ERC-20 Transfer Anatomisi](03_tahta_transfer_anatomi.md) (Coming)
* **↑ İndeks:** [W0 Tahta Serisi](README.md)

---

**Tahta 02 — JSON-RPC 101**  
*Format: Hoca Tahtası (Detaylı, Production-Ready)*  
*Süre: 40-50 dk*  
*Prerequisite: Tahta 01 (EVM Veri Modeli)*  
*Versiyon: 1.1 (Geliştirilmiş)*
