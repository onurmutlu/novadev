# 🧑‍🏫 W0 Tahta Serisi — On-Chain Intelligence Temelleri

**Format:** "Hoca Tahtası" — Lise seviyesinde, sade, uygulamalı  
**Mod:** Read-only, testnet-first, yatırım tavsiyesi değildir

---

## 🎯 Amaç

Bu seri, **blockchain'i okuyarak** anlamlı bilgi çıkarmayı öğretir:
- Özel anahtar yok (read-only)
- Zincirle konuşma (JSON-RPC)
- Event yakalama (ERC-20 Transfer)
- Sağlam ingest (idempotent + reorg)
- Mini rapor (wallet summary)

**Hedef:** W0 sonunda "cüzdan 24h özeti" üretebiliyorsun, **neden/nasıl** biliyorsun.

---

## 📚 Ders Listesi (W0 Bootstrap)

| No | Ders | Başlık | Süre | Durum |
|----|------|--------|------|-------|
| 01 | [Zinciri Okumak: EVM Veri Modeli](01_tahta_evm_giris.md) | Block→Tx→Log→Event akışı, ERC-20 Transfer anatomisi | 25-35 dk | ✅ |
| 02 | [JSON-RPC 101](02_tahta_rpc_101.md) | `eth_blockNumber`, `getBlock`, `getLogs` — 3 kritik komut | 25-35 dk | 📝 Coming |
| 03 | [ERC-20 Transfer Anatomisi](03_tahta_transfer_anatomi.md) | topics/data parçalama, decimals, 3 örnek log çözümleme | 30-40 dk | 📝 Coming |
| 04 | [getLogs Penceresi + Reorg](04_tahta_getlogs_pencere_reorg.md) | Neden küçük aralık? `safe_latest` mantığı, reorg simülasyonu | 30-40 dk | 📝 Coming |
| 05 | [DuckDB + İdempotent Yazma](05_tahta_duckdb_idempotent.md) | `UNIQUE(tx_hash,log_index)`, anti-join pattern | 30-40 dk | 📝 Coming |
| 06 | [State Takibi & Resume](06_tahta_state_resume.md) | `scan_state` tablosu, `last_scanned_block` ile devam | 25-35 dk | 📝 Coming |
| 07 | [24h Cüzdan Raporu + JSON Schema](07_tahta_rapor_json_schema.md) | Rapor formatı, schema validation, `report_v1.json` | 30-40 dk | 📝 Coming |
| 08 | [Mini FastAPI Servisi](08_tahta_fastapi_mini.md) | `/wallet/{addr}/report` endpoint, p95 ölçümü | 30-40 dk | 📝 Coming |
| 09 | [Kalite & CI](09_tahta_kalite_ci.md) | Makefile, CI workflows, schema check | 25-35 dk | 📝 Coming |
| 10 | [Sorun Giderme](10_tahta_troubleshoot.md) | En sık 10 hata ve çözümü, kontrol listeleri | 20-30 dk | 📝 Coming |

**Toplam:** ~280-350 dakika (4.5-6 saat teori)

---

## 🗺️ Öğrenme Yolu

### Faz 1: Kavramlar (Ders 01-03)
```
01. EVM Veri Modeli     → Blockchain = defter, Block→Tx→Log
02. JSON-RPC 101        → 3 komut ile hayatta kal
03. Transfer Anatomisi  → topics/data nasıl okunur?
```

### Faz 2: Sağlam İngest (Ders 04-06)
```
04. getLogs + Reorg     → Pencere boyutu, reorg tamponu
05. İdempotent Yazma    → Çift kayıt yok (UNIQUE key)
06. State Takibi        → Kaldığın yerden devam
```

### Faz 3: Rapor & Servis (Ders 07-08)
```
07. 24h Rapor + Schema  → JSON formatı sabitle
08. FastAPI Servisi     → Endpoint sun, p95 ölç
```

### Faz 4: Kalite & Sorun Giderme (Ders 09-10)
```
09. Kalite & CI         → Otomatik kontroller
10. Troubleshoot        → Sık hatalar ve çözümleri
```

---

## 🎓 Pedagojik İlkeler

### "Hoca Tahtası" Formatı
- **Lise seviyesi:** Önceden blockchain bilgisi gerektirmez
- **Sade dil:** Jargon minimum, her terim açıklanır
- **Görsel:** ASCII şemalar, akış diyagramları
- **Pratik:** Her derste mini quiz + ödev

### Ritim
```
10-15 dk: Kavram (tahta anlatım)
10-15 dk: Örnek/Alıştırma
5-10 dk:  Sık hatalar + Quiz
```

### Eksen
- **Neden?** → Sorunu anla
- **Nasıl?** → Çözümü gör
- **Ne zaman?** → Hangi durumda kullan

---

## 🔧 Pratik Kod (Paralel)

Bu tahta serisi **kavramları** anlatır. Kod ise:
- **[crypto/w0_bootstrap/](../../w0_bootstrap/README.md)** — Çalışan scriptler
  - `rpc_health.py`
  - `capture_transfers_idempotent.py`
  - `report_json.py`
  - `validate_report.py`

**Tavsiye edilen akış:**
1. Dersi oku (30 dk)
2. İlgili script'i çalıştır (10 dk)
3. Quiz'i çöz (5 dk)

---

## 📊 Başarı Kriterleri (W0 Tahta Serisi)

### Kavramsal (Teori)
- [ ] Block→Tx→Log→Event akışını anlatabiliyorum
- [ ] ERC-20 Transfer'in topics/data yapısını parçalayabiliyorum
- [ ] `eth_getLogs` parametrelerini doğru seçebiliyorum
- [ ] İdempotency'nin neden gerekli olduğunu açıklayabiliyorum
- [ ] Reorg tamponu mantığını kavradım

### Uygulamalı (Kod)
- [ ] RPC health check çalıştırabiliyorum
- [ ] Idempotent capture script'ini test ettim
- [ ] 24h rapor oluşturabiliyorum
- [ ] JSON schema validation yapabiliyorum
- [ ] FastAPI endpoint'ini local'de test ettim

### Quiz Skorları
- [ ] Her dersin quiz'inde 4/5 veya üzeri

---

## 🛠️ Gereksinimler

### Yazılım
```bash
# Python 3.11+
python3 --version

# Dependencies
pip install -e ".[crypto]"
# Includes: web3, requests, duckdb, python-dotenv, jsonschema
```

### Hesaplar
- **RPC Provider:** Alchemy/Infura (ücretsiz tier yeterli)
- **Testnet:** Sepolia (mainnet değil!)

### Zaman
- **Teori:** 4.5-6 saat (10 ders)
- **Pratik:** 2-3 saat (scriptler + test)
- **Toplam:** ~8 saat (W0 bootstrap tam)

---

## 🎯 Ne Yapacağız? (Özet)

**Input:**
```
RPC URL (Sepolia testnet)
Wallet address (0x...)
Time window (24 hours)
```

**Process:**
```
1. RPC'ye bağlan
2. Transfer event'lerini çek (getLogs)
3. DuckDB'ye yaz (idempotent)
4. Rapor üret (inbound/outbound/top3)
5. JSON schema ile validate et
6. FastAPI endpoint'ten sun
```

**Output:**
```json
{
  "wallet": "0xd8dA6BF...",
  "window_hours": 24,
  "inbound": 1.2345,
  "outbound": 0.5678,
  "tx_count": 12,
  "top_counterparties": [...]
}
```

---

## 📖 Okuma Sırası (Önerilen)

### Yeni Başlayanlar (Sıfırdan)
```
1. 01_tahta_evm_giris.md         (Temel kavramlar)
2. 02_tahta_rpc_101.md            (RPC komutları)
3. 03_tahta_transfer_anatomi.md   (Event anatomy)
→ Ara: crypto/w0_bootstrap/rpc_health.py çalıştır
4. 04_tahta_getlogs_pencere_reorg.md
5. 05_tahta_duckdb_idempotent.md
→ Ara: capture_transfers_idempotent.py test et
6. 06_tahta_state_resume.md
7. 07_tahta_rapor_json_schema.md
→ Ara: report_json.py | validate_report.py
8. 08_tahta_fastapi_mini.md
→ Ara: uvicorn crypto.service.app:app
9. 09_tahta_kalite_ci.md
10. 10_tahta_troubleshoot.md
```

### Deneyimliler (Hızlı Geçiş)
```
1. 01 (skim)
2. 04 (reorg detayı)
3. 05 (idempotent pattern)
4. 07 (schema contract)
5. 09 (CI setup)
→ Direkt scriptlere geç
```

---

## 🔗 İlgili Kaynaklar

### NovaDev Docs
- **[Program Overview](../../../docs/program_overview.md)** — Tam syllabus
- **[Crypto Overview](../../../docs/crypto_overview.md)** — 8 haftalık plan
- **[Crypto README](../../README.md)** — Setup guide

### Pratik Kod
- **[W0 Bootstrap](../../w0_bootstrap/README.md)** — Scriptler + setup
- **[Service](../../service/app.py)** — FastAPI app
- **[Schema](../../../schemas/report_v1.json)** — JSON contract

### External
- **Ethereum JSON-RPC:** https://ethereum.org/en/developers/docs/apis/json-rpc/
- **ERC-20 Standard:** https://eips.ethereum.org/EIPS/eip-20
- **DuckDB Docs:** https://duckdb.org/docs/

---

## 🎬 Hızlı Başlangıç (5 Dakika)

```bash
# 1. Repo'yu klonla (zaten yaptıysan atla)
cd /Users/onur/code/novadev-protocol

# 2. İlk dersi oku
cat crypto/docs/w0_bootstrap/01_tahta_evm_giris.md

# 3. Quiz'i çöz (ders sonunda)

# 4. RPC health çalıştır
cd crypto/w0_bootstrap
cp .env.example .env
# vim .env → RPC_URL ekle
python rpc_health.py

# 5. Sonraki derse geç
cat ../docs/w0_bootstrap/02_tahta_rpc_101.md  (Coming)
```

---

## ❓ SSS

**S: "Blockchain bilgim yok, anlayabilir miyim?"**  
C: Evet! Seri lise seviyesinde, sıfırdan başlar. Tek şart: temel programlama (Python).

**S: "Mainnet'te test edebilir miyim?"**  
C: Yapabilirsin ama **testnet-first** öneriyoruz (Sepolia). Mainnet için RPC rate-limit daha sıkı.

**S: "Tüm seriyi bitirmem gerekiyor mu?"**  
C: Hayır. Ders 01-03 + pratik script'ler bile W0 bootstrap için yeterli. Derinlik istersen 04-10'a geç.

**S: "Kodlar nerede?"**  
C: **[crypto/w0_bootstrap/](../../w0_bootstrap/)** klasöründe. Tahta serisi **neden**'i anlatır, kod **nasıl**'ı gösterir.

**S: "02. dersi ne zaman eklersiniz?"**  
C: İstek üzerine. "02'yi yaz" de, hazırlayalım.

---

## 📝 Geri Bildirim

Bu seri sürekli gelişir. Öneriler:
- **Issue aç:** GitHub Issues
- **PR gönder:** Typo, clarity, yeni örnekler
- **Tartış:** GitHub Discussions

---

**W0 Tahta Serisi — On-Chain Intelligence Temelleri**

*Format: Hoca Tahtası (Lise Seviyesi)*  
*Toplam: 10 ders, ~280-350 dakika*  
*Durum: Ders 01 hazır, 02-10 coming*  
*Versiyon: 1.1.0*
