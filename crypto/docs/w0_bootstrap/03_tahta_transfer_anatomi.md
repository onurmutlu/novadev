# 🧑‍🏫 Tahta 03 — ERC-20 Transfer Olayının Anatomisi: Topics, Data, Decimals

> **Amaç:** Zincirdeki bir **Transfer** log'unu **mikroskop altında** incelemek: `from`, `to`, `value` çıkarımı + **decimals** ile insan birimine çeviri + edge-case'ler.
> **Mod:** Read-only, testnet-first (Sepolia), **yatırım tavsiyesi değildir**.

---

## 🗺️ Plan (tahta)

1. Transfer event anatomy (ABI → blockchain'e yolculuk)
2. Log yapısı deep-dive: topics vs data
3. Hexadecimal parsing (elle + kod)
4. Decimals: 18, 6, ve diğer varyasyonlar
5. Edge-cases: mint, burn, self-transfer, zero-value, proxy
6. Üç gerçek örnek log (adım adım çözümleme)
7. Production-grade parser implementation
8. Debugging & validation tools
9. Quiz, ödevler, ve troubleshooting

---

## 1) Transfer Event Anatomisi (ABI → Blockchain)

### Solidity Event Tanımı

```solidity
// ERC-20 Standard
event Transfer(
    address indexed from,    // indexed = topic'e yazılır
    address indexed to,      // indexed = topic'e yazılır
    uint256 value            // non-indexed = data'ya yazılır
);
```

### Blockchain'e Nasıl Yansır?

**Event Signature (topic0):**
```
keccak256("Transfer(address,address,uint256)")
= 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef
```

**Topic Mapping:**
```
topics[0] = Event signature (sabit, her Transfer için aynı)
topics[1] = from address (indexed parametre)
topics[2] = to address (indexed parametre)
data      = value (non-indexed, 32 byte)
```

### Görsel: ABI → Blockchain Dönüşümü

```
Solidity Event                      Blockchain Log
─────────────────────────          ───────────────────────────────
Transfer(                          Log {
  address indexed from,    ──→       topics[0]: 0xddf252ad... (signature)
  address indexed to,      ──→       topics[1]: 0x000...from_addr
  uint256 value            ──→       topics[2]: 0x000...to_addr
)                                    data: 0x...value (32 bytes)
                                   }
```

---

## 2) Log Yapısı Deep-Dive

### Complete Log Object

```json
{
  "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",  // DAI kontrat
  "blockNumber": "0x12CDEF0",                             // 19,718,640
  "blockHash": "0x4e3a3754...",
  "transactionHash": "0xabc123...",
  "transactionIndex": "0x5",
  "logIndex": "0x12",                                     // ⭐ Idempotency key
  "removed": false,                                       // ⚠️ Reorg indicator
  
  "topics": [
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    "0x000000000000000000000000a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "0x0000000000000000000000009d8e7f6a5b4c3a2e1f9d8e7c6b5a4d3c2b1a0918"
  ],
  
  "data": "0x0000000000000000000000000000000000000000000000056bc75e2d63100000"
}
```

### Topics vs Data: Neden Ayrı?

**Topics (indexed):**
- ✅ **Filtrelenebilir** (getLogs'ta topics parametresi)
- ✅ **Hızlı arama** (node indexer)
- ⚠️ **Maksimum 3 indexed** parametre (+ topic0 = 4 total)
- ⚠️ **32 byte fixed** (küçük değerler padding alır)

**Data (non-indexed):**
- ✅ **Sınırsız boyut** (teoride)
- ✅ **Gaz tasarrufu** (indexed daha pahalı)
- ⚠️ **Filtrelemez** (tüm log'u çekip parse etmen lazım)

### Neden `value` Data'da?

```solidity
// ❌ Eğer indexed olsaydı (pahalı + gereksiz)
event Transfer(
    address indexed from,
    address indexed to,
    uint256 indexed value    // Her değer için ayrı index!
);

// ✅ Standard (optimal)
event Transfer(
    address indexed from,    // Filtre: "bu adresten çıkanlar"
    address indexed to,      // Filtre: "bu adrese girenler"
    uint256 value            // Miktar: parse edince öğren
);
```

---

## 3) Hexadecimal Parsing (Elle + Kod)

### 3.1 Address Extraction (Topics → 20 Byte)

**Rule:** Topics'te adresler **32 byte (64 hex char)**, ama adres **20 byte (40 hex char)**

```
Original topic (32 byte):
0x000000000000000000000000A0B86991c6218b36c1D19D4a2e9Eb0cE3606eB48
  ├──────24 char (12 byte)─────┤├─────40 char (20 byte)──────┤
           Padding (0's)              Actual Address
```

**Manual extraction:**
```python
topic = "0x000000000000000000000000A0B86991c6218b36c1D19D4a2e9Eb0cE3606eB48"
address = "0x" + topic[-40:]  # Son 40 char
# Result: 0xA0B86991c6218b36c1D19D4a2e9Eb0cE3606eB48
```

**Checksum (opsiyonel ama önerilen):**
```python
from web3 import Web3

def to_checksum_address(addr: str) -> str:
    """EIP-55 checksum address"""
    return Web3.toChecksumAddress(addr.lower())

# "0xa0b86991..." → "0xA0b86991..." (mixed case)
```

### 3.2 Value Parsing (Data → uint256)

**32 Byte Hex Anatomy:**
```
Data field:
0x0000000000000000000000000000000000000000000000056bc75e2d63100000
  ├──────────────────60 chars──────────────────┤├─4 chars─┤
              Leading zeros (padding)          Actual value
```

**Step-by-step parsing:**
```python
data = "0x0000000000000000000000000000000000000000000000056bc75e2d63100000"

# Step 1: Remove 0x prefix
hex_str = data.replace("0x", "")
# "0000000000000000000000000000000000000000000000056bc75e2d63100000"

# Step 2: Convert to integer
raw_value = int(hex_str, 16)
# 100000000000000000000 (decimal)

# Step 3: Scientific notation (yardımcı)
# 1e20 = 100 * 10^18

# Step 4: Apply decimals
decimals = 18
value_unit = raw_value / (10 ** decimals)
# 100.0
```

### 3.3 Hex Calculator (Reference Table)

| Hex | Decimal | With 18 Decimals | With 6 Decimals |
|-----|---------|------------------|-----------------|
| `0x0` | 0 | 0.0 | 0.0 |
| `0x1` | 1 | 0.000000000000000001 | 0.000001 |
| `0xF4240` | 1,000,000 | 0.000000000001 | 1.0 |
| `0xDE0B6B3A7640000` | 1,000,000,000,000,000,000 | 1.0 | 1,000,000,000,000.0 |
| `0x8AC7230489E80000` | 10,000,000,000,000,000,000 | 10.0 | 10,000,000,000,000.0 |

---

## 4) Decimals: Token Precision Deep-Dive

### 4.1 Standard Decimals

| Token | Decimals | 1 Unit (Raw) | Example |
|-------|----------|--------------|---------|
| ETH, DAI, USDT | 18 | 1,000,000,000,000,000,000 | `0xDE0B6B3A7640000` |
| USDC, USDT (Tron) | 6 | 1,000,000 | `0xF4240` |
| WBTC | 8 | 100,000,000 | `0x5F5E100` |
| GUSD | 2 | 100 | `0x64` |

### 4.2 Calculation Examples

#### Example 1: DAI Transfer (18 decimals)

```python
# Raw value from blockchain
raw = 0x0000000000000000000000000000000000000000000000056bc75e2d63100000
raw_int = int(raw, 16)  # 100000000000000000000

# Decimals
DECIMALS_DAI = 18

# Human-readable
value_dai = raw_int / (10 ** DECIMALS_DAI)
# 100.0 DAI
```

#### Example 2: USDC Transfer (6 decimals)

```python
# Raw value
raw = 0x000000000000000000000000000000000000000000000000000000003B9ACA00
raw_int = int(raw, 16)  # 1000000000

# Decimals
DECIMALS_USDC = 6

# Human-readable
value_usdc = raw_int / (10 ** DECIMALS_USDC)
# 1000.0 USDC
```

#### Example 3: WBTC Transfer (8 decimals)

```python
# Raw value
raw = 0x0000000000000000000000000000000000000000000000000000000005F5E100
raw_int = int(raw, 16)  # 100000000

# Decimals
DECIMALS_WBTC = 8

# Human-readable
value_btc = raw_int / (10 ** DECIMALS_WBTC)
# 1.0 WBTC
```

### 4.3 Precision Pitfalls

**⚠️ Floating Point Precision:**
```python
# ❌ Dikkatli: float precision loss
value = 123456789012345678901234567890 / 10**18
# Python: 123456789012345680000.0 (precision kaybı!)

# ✅ İyi: Decimal library kullan
from decimal import Decimal

raw = Decimal('123456789012345678901234567890')
decimals = Decimal('18')
value = raw / (Decimal('10') ** decimals)
# Decimal('123456789012.345678901234567890')
```

**Production-grade parser:**
```python
from decimal import Decimal

def parse_value_precise(data_hex: str, decimals: int) -> Decimal:
    """
    Precision-safe value parsing
    Returns Decimal for exact arithmetic
    """
    raw = int(data_hex.replace("0x", ""), 16)
    return Decimal(raw) / (Decimal(10) ** decimals)

# Kullanım
value = parse_value_precise("0xDE0B6B3A7640000", 18)
# Decimal('1.000000000000000000')
```

---

## 5) Edge-Cases: Mint, Burn, ve Özel Durumlar

### 5.1 Mint (Token Basımı)

**Pattern:** `from = 0x0000000000000000000000000000000000000000`

```json
{
  "topics": [
    "0xddf252ad...",
    "0x0000000000000000000000000000000000000000000000000000000000000000",  // ← Zero address (from)
    "0x000000000000000000000000RecipientAddress..."
  ],
  "data": "0x..."  // Basılan miktar
}
```

**Anlamı:**
- ✅ Yeni token yaratıldı
- ✅ Total supply arttı
- ⚠️ Protokol kurallarına bağlı (bazı token'lar mint'e izin vermez)

**Detection code:**
```python
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

def is_mint(log: dict) -> bool:
    """Mint (basım) detection"""
    from_addr = topic_to_addr(log["topics"][1])
    return from_addr.lower() == ZERO_ADDRESS.lower()

# Kullanım
if is_mint(log):
    print(f"🪙 Mint: {value} tokens created")
```

### 5.2 Burn (Token Yakımı)

**Pattern:** `to = 0x0000000000000000000000000000000000000000`

```json
{
  "topics": [
    "0xddf252ad...",
    "0x000000000000000000000000SenderAddress...",
    "0x0000000000000000000000000000000000000000000000000000000000000000"  // ← Zero address (to)
  ],
  "data": "0x..."  // Yakılan miktar
}
```

**Anlamı:**
- ✅ Token yok edildi
- ✅ Total supply azaldı
- ⚠️ Geri döndürülemez (permanen destruction)

**Detection code:**
```python
def is_burn(log: dict) -> bool:
    """Burn (yakım) detection"""
    to_addr = topic_to_addr(log["topics"][2])
    return to_addr.lower() == ZERO_ADDRESS.lower()

if is_burn(log):
    print(f"🔥 Burn: {value} tokens destroyed")
```

### 5.3 Self-Transfer

**Pattern:** `from == to`

```json
{
  "topics": [
    "0xddf252ad...",
    "0x000000000000000000000000SameAddress...",
    "0x000000000000000000000000SameAddress..."  // ← Aynı adres
  ],
  "data": "0x..."
}
```

**Neden olur?**
- ✅ Internal accounting (bazı protokoller)
- ✅ Staking/unstaking mekanizmaları
- ✅ Fee collection
- ⚠️ Bazen hata/bug göstergesi

**Detection:**
```python
def is_self_transfer(log: dict) -> bool:
    """Self-transfer detection"""
    from_addr = topic_to_addr(log["topics"][1])
    to_addr = topic_to_addr(log["topics"][2])
    return from_addr.lower() == to_addr.lower()

if is_self_transfer(log):
    print(f"↔️  Self-transfer: {value} (internal accounting?)")
```

### 5.4 Zero-Value Transfer

**Pattern:** `value = 0`

```python
def is_zero_value(log: dict) -> bool:
    """Zero-value transfer detection"""
    raw_value = int(log["data"].replace("0x", ""), 16)
    return raw_value == 0

if is_zero_value(log):
    print("⚠️  Zero-value transfer (spam/signal?)")
```

**Neden olur?**
- ✅ Notification mechanism (signal gönderme)
- ✅ Spam/airdrop claims
- ⚠️ Gas waste (gereksiz)

### 5.5 Proxy Contracts

**Pattern:** `address` != gerçek logic kontrat

```
User Call
   ↓
Proxy Contract (0xProxy...)  ← Log address
   ↓ delegatecall
Implementation (0xImpl...)   ← Gerçek logic
```

**Tehlike:**
- ⚠️ Implementation değişebilir (upgradeable)
- ⚠️ Davranış farklılaşabilir
- ✅ Event signature aynı kalır (ERC-20 standardı)

**Detection (advanced):**
```python
def is_proxy_contract(address: str, rpc_url: str) -> bool:
    """
    Proxy detection (basit heuristic)
    Gerçek detection için: EIP-1967 storage slot check
    """
    # Check if contract has implementation() or _implementation() method
    # Bu tam bir örnek değil, concept gösterimi
    pass
```

---

## 6) Üç Gerçek Örnek Log (Adım Adım Çözümleme)

### Örnek A: DAI Transfer (18 Decimals)

**Raw Log:**
```json
{
  "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
  "topics": [
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    "0x000000000000000000000000a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "0x0000000000000000000000009d8e7f6a5b4c3a2e1f9d8e7c6b5a4d3c2b1a0918"
  ],
  "data": "0x0000000000000000000000000000000000000000000000056bc75e2d63100000",
  "blockNumber": "0x12CDEF0",
  "transactionHash": "0xabc123...",
  "logIndex": "0x5"
}
```

**Step-by-Step Parsing:**

**1. Verify Transfer:**
```python
assert log["topics"][0] == "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
# ✅ This is a Transfer event
```

**2. Extract FROM:**
```python
from_topic = log["topics"][1]
# "0x000000000000000000000000a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

from_addr = "0x" + from_topic[-40:]
# "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
```

**3. Extract TO:**
```python
to_topic = log["topics"][2]
to_addr = "0x" + to_topic[-40:]
# "0x9d8e7f6a5b4c3a2e1f9d8e7c6b5a4d3c2b1a0918"
```

**4. Parse VALUE:**
```python
data = log["data"]
# "0x0000000000000000000000000000000000000000000000056bc75e2d63100000"

raw_value = int(data.replace("0x", ""), 16)
# 100000000000000000000 (decimal)

# DAI has 18 decimals
value_dai = raw_value / (10 ** 18)
# 100.0 DAI
```

**Final Result:**
```
Token: DAI (0x6B17...)
From:  0xa0b86991... 
To:    0x9d8e7f6a...
Amount: 100.0 DAI
Block: 19,718,640
Tx: 0xabc123...
```

### Örnek B: USDC Transfer (6 Decimals)

**Raw Log:**
```json
{
  "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
  "topics": [
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    "0x000000000000000000000000742d35Cc6634C0532925a3b844Bc9e7595f0bEb7",
    "0x000000000000000000000000Ab5801a7D398351b8bE11C439e05C5B3259aeC9B"
  ],
  "data": "0x000000000000000000000000000000000000000000000000000000003B9ACA00",
  "blockNumber": "0x12CD000",
  "transactionHash": "0xdef456...",
  "logIndex": "0x3"
}
```

**Parsing:**
```python
# FROM
from_addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb7"

# TO
to_addr = "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B"

# VALUE
raw = int("0x3B9ACA00", 16)  # 1000000000
value_usdc = raw / (10 ** 6)  # 1000.0 USDC
```

**Final Result:**
```
Token: USDC (0xA0b8...)
From:  0x742d35Cc...
To:    0xAb5801a7...
Amount: 1000.0 USDC
```

### Örnek C: Mint Event (WBTC, 8 Decimals)

**Raw Log:**
```json
{
  "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
  "topics": [
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    "0x0000000000000000000000000000000000000000000000000000000000000000",  // ← ZERO (mint)
    "0x000000000000000000000000Recipient123456789abcdef..."
  ],
  "data": "0x0000000000000000000000000000000000000000000000000000000005F5E100",
  "blockNumber": "0x12CE000",
  "transactionHash": "0xghi789...",
  "logIndex": "0x0"
}
```

**Parsing:**
```python
# FROM (zero = mint)
from_addr = "0x0000000000000000000000000000000000000000"
is_mint = True

# TO
to_addr = "0xRecipient123456789abcdef..."

# VALUE
raw = int("0x05F5E100", 16)  # 100000000
value_wbtc = raw / (10 ** 8)  # 1.0 WBTC
```

**Final Result:**
```
🪙 MINT Event
Token: WBTC (0x2260...)
To:    0xRecipient...
Amount: 1.0 WBTC (minted)
```

---

## 7) Production-Grade Parser Implementation

### Complete Transfer Parser

```python
#!/usr/bin/env python3
"""
Production-grade ERC-20 Transfer log parser
"""
from decimal import Decimal
from typing import Dict, Optional, Literal
from dataclasses import dataclass

# Constants
TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

@dataclass
class TransferEvent:
    """Parsed Transfer event"""
    token: str
    from_addr: str
    to_addr: str
    value_raw: int
    value_unit: Decimal
    decimals: int
    block_number: int
    tx_hash: str
    log_index: int
    event_type: Literal["transfer", "mint", "burn", "self"]
    
    def __str__(self) -> str:
        emoji = {
            "mint": "🪙",
            "burn": "🔥",
            "self": "↔️",
            "transfer": "→"
        }[self.event_type]
        
        return (
            f"{emoji} {self.event_type.upper()}: "
            f"{self.value_unit} tokens | "
            f"{self.from_addr[:8]}... → {self.to_addr[:8]}... | "
            f"Block {self.block_number:,}"
        )

class TransferParser:
    """ERC-20 Transfer event parser"""
    
    @staticmethod
    def topic_to_address(topic: str) -> str:
        """Extract address from topic (last 20 bytes)"""
        cleaned = topic.lower().replace("0x", "")
        return "0x" + cleaned[-40:]
    
    @staticmethod
    def parse_value(data_hex: str, decimals: int) -> tuple[int, Decimal]:
        """
        Parse value from data field
        Returns (raw_int, decimal_value)
        """
        raw = int(data_hex.replace("0x", ""), 16)
        value_unit = Decimal(raw) / (Decimal(10) ** decimals)
        return raw, value_unit
    
    @staticmethod
    def classify_transfer(from_addr: str, to_addr: str) -> str:
        """Classify transfer type"""
        from_lower = from_addr.lower()
        to_lower = to_addr.lower()
        zero = ZERO_ADDRESS.lower()
        
        if from_lower == zero:
            return "mint"
        elif to_lower == zero:
            return "burn"
        elif from_lower == to_lower:
            return "self"
        else:
            return "transfer"
    
    @classmethod
    def parse_log(cls, log: Dict, decimals: int) -> Optional[TransferEvent]:
        """
        Parse Transfer log with full validation
        
        Args:
            log: Raw log dict from eth_getLogs
            decimals: Token decimals (must be known in advance)
        
        Returns:
            TransferEvent or None (if not a Transfer)
        
        Raises:
            ValueError: If log structure is invalid
        """
        # Validate topics
        topics = log.get("topics", [])
        if len(topics) < 3:
            raise ValueError(f"Invalid topics length: {len(topics)}")
        
        # Verify Transfer signature
        if topics[0].lower() != TRANSFER_SIG.lower():
            return None  # Not a Transfer event
        
        # Extract addresses
        from_addr = cls.topic_to_address(topics[1])
        to_addr = cls.topic_to_address(topics[2])
        
        # Parse value
        data = log.get("data", "0x0")
        raw_value, value_unit = cls.parse_value(data, decimals)
        
        # Classify
        event_type = cls.classify_transfer(from_addr, to_addr)
        
        # Build result
        return TransferEvent(
            token=log["address"].lower(),
            from_addr=from_addr,
            to_addr=to_addr,
            value_raw=raw_value,
            value_unit=value_unit,
            decimals=decimals,
            block_number=int(log["blockNumber"], 16),
            tx_hash=log["transactionHash"].lower(),
            log_index=int(log["logIndex"], 16),
            event_type=event_type
        )

# Usage example
if __name__ == "__main__":
    # Example log (DAI)
    log = {
        "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "blockNumber": "0x12CDEF0",
        "transactionHash": "0xabc123...",
        "logIndex": "0x5",
        "topics": [
            TRANSFER_SIG,
            "0x000000000000000000000000a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "0x0000000000000000000000009d8e7f6a5b4c3a2e1f9d8e7c6b5a4d3c2b1a0918"
        ],
        "data": "0x0000000000000000000000000000000000000000000000056bc75e2d63100000"
    }
    
    parser = TransferParser()
    transfer = parser.parse_log(log, decimals=18)
    
    if transfer:
        print(transfer)
        print(f"Amount: {transfer.value_unit} tokens")
```

---

## 8) Debugging & Validation Tools

### 8.1 Log Validator

```python
def validate_transfer_log(log: Dict) -> list[str]:
    """
    Validate Transfer log structure
    Returns list of issues (empty = valid)
    """
    issues = []
    
    # Check required fields
    required_fields = ["address", "blockNumber", "transactionHash", 
                      "logIndex", "topics", "data"]
    for field in required_fields:
        if field not in log:
            issues.append(f"Missing field: {field}")
    
    # Check topics structure
    topics = log.get("topics", [])
    if len(topics) < 3:
        issues.append(f"Invalid topics count: {len(topics)} (expected 3+)")
    
    # Verify Transfer signature
    if topics and topics[0].lower() != TRANSFER_SIG.lower():
        issues.append(f"Not a Transfer event (topic0: {topics[0][:10]}...)")
    
    # Check data format
    data = log.get("data", "")
    if not data.startswith("0x"):
        issues.append(f"Invalid data format: {data[:10]}")
    
    return issues

# Usage
issues = validate_transfer_log(log)
if issues:
    print("❌ Validation errors:")
    for issue in issues:
        print(f"   • {issue}")
else:
    print("✅ Log structure valid")
```

### 8.2 Hex → Decimal Converter (CLI Tool)

```python
#!/usr/bin/env python3
"""Hex to decimal converter for Transfer values"""
import sys

def hex_to_decimal(hex_str: str, decimals: int = 18):
    """Convert hex value to human-readable decimal"""
    cleaned = hex_str.replace("0x", "")
    raw = int(cleaned, 16)
    value = raw / (10 ** decimals)
    
    print(f"Hex:     {hex_str}")
    print(f"Raw:     {raw:,}")
    print(f"Value:   {value:,.{decimals}f}")
    print(f"Compact: {value:,.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hex_converter.py 0x... [decimals]")
        sys.exit(1)
    
    hex_val = sys.argv[1]
    decimals = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    
    hex_to_decimal(hex_val, decimals)
```

**Usage:**
```bash
$ python hex_converter.py 0xDE0B6B3A7640000
Hex:     0xDE0B6B3A7640000
Raw:     1,000,000,000,000,000,000
Value:   1.000000000000000000
Compact: 1.0000

$ python hex_converter.py 0x3B9ACA00 6
Hex:     0x3B9ACA00
Raw:     1,000,000,000
Value:   1000.000000
Compact: 1000.0000
```

### 8.3 Batch Log Analyzer

```python
def analyze_transfer_batch(logs: list[Dict], decimals: int):
    """Analyze a batch of Transfer logs"""
    parser = TransferParser()
    
    stats = {
        "total": len(logs),
        "mint": 0,
        "burn": 0,
        "self": 0,
        "transfer": 0,
        "zero_value": 0,
        "total_volume": Decimal(0)
    }
    
    for log in logs:
        try:
            transfer = parser.parse_log(log, decimals)
            if not transfer:
                continue
            
            stats[transfer.event_type] += 1
            stats["total_volume"] += transfer.value_unit
            
            if transfer.value_raw == 0:
                stats["zero_value"] += 1
        
        except Exception as e:
            print(f"⚠️  Parse error: {e}")
    
    # Report
    print(f"\n📊 Transfer Batch Analysis")
    print(f"   Total logs: {stats['total']}")
    print(f"   • Transfers: {stats['transfer']}")
    print(f"   • Mints:  {stats['mint']}")
    print(f"   • Burns: {stats['burn']}")
    print(f"   • Self: {stats['self']}")
    print(f"   • Zero-value: {stats['zero_value']}")
    print(f"   Total volume: {stats['total_volume']:,.4f}")
    
    return stats
```

---

## 9) Sık Hatalar & Anti-Patterns

### ❌ Hata 1: Decimals Unutmak
```python
# YANLIŞ
raw_value = int(log["data"], 16)
print(f"Value: {raw_value}")  # 1000000000000000000 (anlamsız!)

# DOĞRU
raw_value = int(log["data"], 16)
value = raw_value / (10 ** 18)
print(f"Value: {value} tokens")  # 1.0 (anlamlı!)
```

### ❌ Hata 2: Hex'i Decimal Sanmak
```python
# YANLIŞ
data = "0x0DE0B6B3A7640000"
value = int(data)  # ValueError: invalid literal for int()

# DOĞRU
data = "0x0DE0B6B3A7640000"
value = int(data, 16)  # Base 16 belirt!
```

### ❌ Hata 3: topic0 Kontrol Etmemek
```python
# YANLIŞ
from_addr = topic_to_addr(log["topics"][1])  # Transfer mi? Approval mi?

# DOĞRU
if log["topics"][0].lower() != TRANSFER_SIG.lower():
    raise ValueError("Not a Transfer event!")
from_addr = topic_to_addr(log["topics"][1])
```

### ❌ Hata 4: logIndex Idempotency Key'den Unutmak
```python
# YANLIŞ (çift kayıt riski)
unique_key = log["transactionHash"]  # Aynı tx'te 2+ transfer olabilir!

# DOĞRU
unique_key = (log["transactionHash"], log["logIndex"])
```

### ❌ Hata 5: Zero Address'i Görmezden Gelmek
```python
# YANLIŞ (mint/burn'ü kaçırırsın)
if from_addr and to_addr:  # Zero address = falsy!
    process_transfer(from_addr, to_addr, value)

# DOĞRU
# Zero address'i explicit kontrol et
```

### ❌ Hata 6: Float Precision Loss
```python
# YANLIŞ (büyük sayılarda precision kaybı)
value = 123456789012345678901234567890 / 10**18  # float overflow!

# DOĞRU (Decimal kullan)
from decimal import Decimal
value = Decimal('123456789012345678901234567890') / Decimal(10**18)
```

---

## 10) Mini Quiz

### Sorular

1. `topics[0]` hangi bilgiyi içerir ve neden sabit?
2. Bir adresi topics'ten çıkarmak için kaç karakteri kullanırız?
3. 18 decimals ile 1.0 token'ın raw değeri nedir?
4. Mint event'i nasıl tespit edilir?
5. Burn event'i nasıl tespit edilir?
6. Self-transfer neden olur (3 neden)?
7. Idempotent key neden `(tx_hash, log_index)` çifti?
8. Float yerine Decimal kullanmanın avantajı nedir?

### Cevap Anahtarı

1. Event signature (`keccak256("Transfer(address,address,uint256)")`); sabit çünkü event tanımı standardize
2. **Son 40 hex karakter** (20 byte = adres boyutu)
3. `1,000,000,000,000,000,000` (decimal) = `0xDE0B6B3A7640000` (hex)
4. `from address == 0x0000...0000` (zero address)
5. `to address == 0x0000...0000` (zero address)
6. (a) Internal accounting, (b) Staking/fee mechanisms, (c) Bazen hata/bug
7. Bir tx'te birden fazla Transfer olabilir; `log_index` tekliği garanti eder
8. **Precision loss yok**, büyük sayılarda kesin aritmetik

---

## 11) Ödevler (Pratik)

### Ödev 1: Manual Parsing

**Verilen log'u elle parse et:**
```json
{
  "topics": [
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
    "0x000000000000000000000000742d35Cc6634C0532925a3b844Bc9e7595f0bEb7",
    "0x000000000000000000000000Ab5801a7D398351b8bE11C439e05C5B3259aeC9B"
  ],
  "data": "0x00000000000000000000000000000000000000000000000000000000773594000"
}
```
**Sorular:**
- from address?
- to address?
- Raw value (hex → decimal)?
- 18 decimals ile human value?
- 6 decimals ile human value?

### Ödev 2: Edge-Case Detection

**Bu log'ların tipini belirle (mint/burn/self/normal):**
```python
logs = [
    {"topics": [SIG, "0x0...0", "0x123..."]},  # ?
    {"topics": [SIG, "0x456...", "0x0...0"]},  # ?
    {"topics": [SIG, "0x789...", "0x789..."]}, # ?
    {"topics": [SIG, "0xabc...", "0xdef..."]}, # ?
]
```

### Ödev 3: Production Parser

**`TransferParser` class'ını kullanarak:**
- 10 log parse et
- Her tip için adet say (mint/burn/self/transfer)
- Toplam volume hesapla
- Sonuçları tablo olarak yazdır

### Ödev 4: Decimal Converter Tool

**CLI tool yaz:**
```bash
$ python converter.py --hex 0xDE0B6B3A7640000 --decimals 18
```
**Output:**
```
Hex:     0xDE0B6B3A7640000
Raw:     1000000000000000000
Value:   1.000000000000000000 (18 decimals)
Compact: 1.0
```

**Bonus:** `--reverse` flag ekle (decimal → hex)

---

## 12) Terimler Sözlüğü

| Terim | Tanım |
|-------|-------|
| **Event Signature** | `keccak256("EventName(types...)")` hash'i |
| **topic0** | Event signature (her Transfer için sabit) |
| **topics[1-3]** | Indexed parametreler (max 3) |
| **data** | Non-indexed parametreler (ABI-encoded) |
| **Decimals** | Token ondalık hassasiyeti (18, 6, 8, vb.) |
| **Raw value** | Blockchain'deki ham integer değer |
| **Unit value** | İnsan-okur değer (`raw / 10^decimals`) |
| **Zero address** | `0x0000...0000` (mint/burn göstergesi) |
| **Mint** | `from = zero` (token basımı) |
| **Burn** | `to = zero` (token yakımı) |
| **Self-transfer** | `from == to` (internal accounting) |
| **Proxy contract** | Logic'i delegate eden kontrat |
| **Idempotent key** | `(tx_hash, log_index)` teklik garantisi |

---

## 🔗 İlgili Kaynaklar

### Repo Kodları
- `crypto/w0_bootstrap/capture_transfers_idempotent.py` → Log parsing implementation
- `crypto/w0_bootstrap/report_json.py` → Aggregate analysis
- `schemas/report_v1.json` → JSON schema

### Previous Lessons
- **← Tahta 01:** [EVM Veri Modeli](01_tahta_evm_giris.md)
- **← Tahta 02:** [JSON-RPC 101](02_tahta_rpc_101.md)
- **→ Tahta 04:** [getLogs Penceresi + Reorg](04_tahta_getlogs_pencere_reorg.md) (Coming)

### External
- **ERC-20 Standard:** https://eips.ethereum.org/EIPS/eip-20
- **Ethereum Log Topics:** https://ethereum.org/en/developers/docs/apis/json-rpc/#eth_getlogs
- **Etherscan:** Example Transfer logs (mainnet)

---

## 🛡️ Güvenlik / Etik

- **Read-only:** Özel anahtar yok, imza yok, custody yok
- **`.env` hygiene:** Secrets asla commit etme
- **Testnet-first:** Sepolia ile başla
- **Eğitim amaçlı:** Yatırım tavsiyesi değildir

---

## 🔗 Navigasyon

- **← Önceki:** [02 - JSON-RPC 101](02_tahta_rpc_101.md)
- **→ Sonraki:** [04 - getLogs Penceresi + Reorg](04_tahta_getlogs_pencere_reorg.md) (Coming)
- **↑ İndeks:** [W0 Tahta Serisi](README.md)

---

**Tahta 03 — ERC-20 Transfer Anatomisi**  
*Format: Production-Ready Parser + Edge-Cases*  
*Süre: 50-60 dk*  
*Prerequisite: Tahta 01-02*  
*Versiyon: 1.1 (Geliştirilmiş)*
