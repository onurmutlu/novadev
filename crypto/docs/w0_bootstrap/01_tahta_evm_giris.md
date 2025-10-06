# 🧑‍🏫 Tahta 01 — Zinciri Okumak: EVM'de Blok → İşlem → Log → Event (Read-Only)

> **Amaç:** "Kripto" deyince ekran görüntüsü değil, **zincirin kendisini okuyup** basit bir **cüzdan özeti** çıkarabilmek.
> **Mod:** Read-only (özel anahtar yok), testnet-first, yatırım tavsiyesi değildir.

---

## 🧱 Neyi Öğreneceğiz? (Tahta Planı)

1. **Blockchain = defter** 👉 "Satır" = **blok**
2. **EVM veri akışı:** **Block → Tx → Receipt → Log → Event**
3. **ERC-20 `Transfer`** olayını "parçalarına ayırma"
4. **JSON-RPC** (3 komutla hayatta kal)
5. **Sağlam ingest hattı:** idempotent + reorg + state
6. **Mini rapor**: "Son 24 saatte giren/çıkan miktar + top 3 karşı taraf"

---

## 1) Zincir nedir? (Basit benzetme)

* Bir **defter** düşün. Her sayfa = **blok**.
* Her sayfaya yazılan satırlar = **işlemler (tx)**.
* Bir işlem çalışınca "not düşer": **log/event** (kanıt niteliğinde).

```
[ Block N ]  ->  [ Block N+1 ]  ->  [ Block N+2 ]
   |   |               |   |
  tx  tx             tx   tx
   |   |              |    |
  log log           log  log   (event = anlamlı log)
```

**Neden önemli?** Çünkü "ne oldu?" sorusunun cevabı **log/event**'lerde saklı.

---

## 2) EVM Veri Akışı (el alıştırması)

```
Block  ──>  Tx (işlem)  ──>  Receipt  ──>  Log(lar)  ──>  Event(ler)
  ↑             ↑
zaman      kim ne çağırdı?
```

* **Block:** zaman damgası, blok numarası.
* **Tx (işlem):** kimin kimi/neyi çağırdığı.
* **Receipt:** "işlem oldu mu?" + gaz bilgisi.
* **Log:** sözleşmenin "yayınladığı" makine-okur kayıt.
* **Event:** log'un anlamlı hâli (ABI ile okunur).

---

## 3) ERC-20 `Transfer` Olayı (tahtada parçalama)

**Tanım:**
`Transfer(address indexed from, address indexed to, uint256 value)`

**Zincirde nasıl görünür?**

* `topic0` = `keccak256("Transfer(address,address,uint256)")` → olayın imzası
* `topic1` = `from` (adres, 32 baytın son 20 baytı)
* `topic2` = `to` (adres)
* `data`   = `value` (miktar, 32 bayt, ondalık **decimals** ile ölçeklenir)

**Tahta şeması:**

```
topics[0] = 0xddf252ad... (sabit: "Transfer" imzası)
topics[1] = 0x000...<40 hex>  -> from
topics[2] = 0x000...<40 hex>  -> to
data      = 0x000...<64 hex>  -> value (ham sayı)
value_unit = value / (10^decimals)
```

---

## 4) JSON-RPC: 3 Komutla Hayatta Kal

**A) Son blok numarası (sağlık kontrolü)**
`eth_blockNumber` → "Node canlı mı? Kaç ms?"

**B) Blok bilgisi (zaman lazım olduğunda)**
`eth_getBlockByNumber` → `timestamp` alırız.

**C) Olay tarama (çekirdek iş)**
`eth_getLogs` → belli blok aralığında, belli `topic0` (Transfer) için log'ları getirir.

> **Pratik kural:** `getLogs` aralığını **küçük** tut (≈1000–2000 blok). Yoksa zaman aşımı / rate-limit.

---

## 5) Sağlam Ingest Hattı (sorunsuz tekrarlanabilirlik)

**İdempotent** (üst üste çalıştırınca **çift kayıt yok**)
**Reorg tamponu** (son N blok kesinleşmeyebilir)
**State** (en son nereye kadar okudun?)

**DuckDB tablo anahtarı:**

```
UNIQUE(tx_hash, log_index)   # aynı log bir daha yazılmasın
```

**Reorg mantığı:**

```
latest = zincirde en son blok
safe_latest = latest - CONFIRMATIONS   # örn. 5-12
```

**State tablosu:** `last_scanned_block` → kaldığın yerden devam.

---

## 6) Mini Rapor (hedefimiz)

**Soru:** "Bu cüzdana son 24 saatte ne girdi/çıktı? En çok kimlerle konuştu?"

**Cevap (JSON):**

```json
{
  "wallet": "0xabc...",
  "window_hours": 24,
  "inbound":  1.2345,
  "outbound": 0.5678,
  "tx_count": 12,
  "top_counterparties": [
    {"address":"0x111...","amount":0.50},
    {"address":"0x222...","amount":0.30},
    {"address":"0x333...","amount":0.25}
  ]
}
```

Bu raporu üretmek için:

1. **Transfer loglarını** çektik
2. **Decimalleri** uyguladık
3. **Zaman penceresi** (24h) ve **gruplama** yaptık

---

## 7) Uygulamalı Adımlar (özet, bugünlük)

> Not: Kodlar zaten repo'da (W0 bootstrap). Burada **neden**'i kavrıyoruz.

1. **RPC sağlık**: "Son blok kaç? Kaç ms?"
2. **İdempotent ingest**: Son 8k blok → DuckDB (çift yok).
3. **24h rapor**: inbound/outbound + top 3.
4. **Şema doğrulama** (JSON Schema v1): Yanıt formatı doğru mu?

---

## 8) En Sık Hatalar (tahta uyarıları)

* `value_unit` yerine ham `value` yazmak → **10^18** çarpanı unutur, rakamlar uçar.
* Geniş `getLogs` aralığı → timeout/429.
* Reorg yok → "hayalet" kayıtlar kalır.
* İdempotent yok → tekrar koşunca **çift kayıt**.

---

## 9) Mini Quiz (kendini ölç)

1. `topic0` niye sabit?
2. `getLogs` penceresini neden küçük tutuyoruz?
3. İdempotency'yi hangi anahtarla sağlıyoruz?
4. `safe_latest` nasıl hesaplanır?
5. `value_unit` nasıl çıkar?

**Cevap anahtarı (kısa):**

1. Olay imzasının keccak256 hash'i = sabit kimlik.
2. Timeout + rate-limit riski; yanıt boyutu.
3. **(tx_hash, log_index)** unique.
4. `latest - CONFIRMATIONS`.
5. `raw_value / 10^decimals`.

---

## 10) Hatırlatmalar (Güvenlik/Etik)

* **Read-only**: Özel anahtar yok, imza yok, custody yok.
* **Testnet-first**: Sepolia gibi test ağında dene.
* **Yatırım tavsiyesi değildir.**

---

# ➕ Ek: Terimler Sözlüğü (kısa)

* **Block:** Zincirdeki sıra numaralı sayfa.
* **Tx (işlem):** "Şu sözleşmeyi şu parametrelerle çağır."
* **Receipt:** İşlemin sonucu ve yan etkileri.
* **Log/Event:** Sözleşmenin "yayınladığı" kayıt; event = anlamlı log.
* **ERC-20:** En yaygın token standardı (Transfer olayı var).
* **JSON-RPC:** Node ile konuşma protokolü (HTTP + JSON).
* **Idempotent:** Aynı işi tekrar çalıştırınca **aynı sonuç**, **çift kayıt yok**.

---

**Bugünlük bitti.** Sonraki derste: **getLogs**'u doğru pencereyle kullanıp **reorg + state**'i pratikte göstereceğiz.

---

## 🔗 Navigasyon

- **Ana İndeks:** [W0 Tahta Serisi](README.md)
- **Sonraki Ders:** [02 - JSON-RPC 101](02_tahta_rpc_101.md) (Coming)
- **Pratik Kod:** [crypto/w0_bootstrap/](../../w0_bootstrap/README.md)

---

**Tahta 01 — Zinciri Okumak**  
*Format: Hoca Tahtası (Lise Seviyesi)*  
*Süre: 25-35 dk*  
*Prerequisite: Yok (temelden başlar)*
