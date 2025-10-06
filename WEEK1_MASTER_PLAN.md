# 🚀 WEEK 1 — MASTER PLAN (AI + CRYPTO, Paralel Sprint)

> **Hedef:** 5 günde hem AI hem Crypto hattında **çalışan, ölçülen, raporlanan** bir paket çıkarmak.  
> **Ritim:** T → P → X (Teori → Pratik → Ürün).  
> **Prensip:** Read-only, testnet-first, idempotent, schema-driven, p95<1s.

---

## 📋 İçindekiler

1. [Haftalık Hedefler (DoD)](#haftalık-hedefler-definition-of-done)
2. [Günlük Detaylı Plan](#günlük-detaylı-plan)
3. [Test & Kalite Kriterleri](#test--kalite-w1)
4. [Komut Kartı](#komut-kartı-cheatsheet)
5. [Çıktı Klasörleri](#çıktı-klasörleri)
6. [Risk Yönetimi](#risk--önlem)
7. [Metrik Takibi](#skor-tahtası-takip-edilecek-metrikler)
8. [PR/Issue Şablonları](#prissue-şablonları)
9. [Stretch Goals](#stretch-goals-opsiyonel)
10. [Kapanış Kriteri](#kapanış-kriteri)

---

## 🎯 Haftalık Hedefler (Definition of Done)

### AI Hattı: Linear Regression from Scratch

#### Teori (T)
- [ ] Gradient Descent manuel implementasyon anlaşıldı
- [ ] `nn.Module` API'si kullanımı öğrenildi
- [ ] Train/Val split mantığı oturdu
- [ ] Early Stopping + L2 regularization prensipleri kavrandı

#### Pratik (P)
- [ ] Manuel GD + `nn.Module` implementasyonu tamamlandı
- [ ] Train/Val split uygulandı (80/20)
- [ ] **Early Stopping** + **L2** regularization entegre edildi
- [ ] **Val MSE ≤ 0.50** (sentetik veri üzerinde, 1000 sample)
- [ ] Loss eğrileri: `outputs/ai/w1_loss_curve.png` oluşturuldu
- [ ] Ablation study: LR sweep + L2 on/off + Early Stopping on/off

#### Ürün (X)
- [ ] **Tests:** `pytest tests/test_linreg.py -q` yeşil (coverage ≥ 70%)
- [ ] **Model checkpoint:** `outputs/ai/best_model.pt` kaydedildi
- [ ] **Özet rapor:** `reports/ai/w1_summary.md` (sonuç + ablation + grafikler)
- [ ] Kod temiz (ruff check geçti)

**Başarı Metrikleri:**
- Val MSE: **≤ 0.50** (hedef: 0.40-0.45)
- Convergence: **< 200 epoch**
- Test coverage: **≥ 70%**

---

### Crypto Hattı: Collector Loop + API Performance

#### Teori (T)
- [ ] Collector loop architecture (30s polling)
- [ ] Idempotent ingest + tail rescan patterns
- [ ] Cache optimization strategies (LRU+TTL)
- [ ] API performance targets (p95 < 1s)

#### Pratik (P)
- [ ] **Collector loop**: 30s polling, idempotent + tail rescan çalışıyor
- [ ] **/wallet/{addr}/report** p95 **< 1s** (warm cache)
- [ ] **Schema v1** kontratına %100 uyum (validator yeşil)
- [ ] **Load test:** 100 req, c=10, error=0, p95<1s
- [ ] 5 test cüzdanı için rapor üretildi

#### Ürün (X)
- [ ] **Collector service:** Systemd/supervisor ile daemonize
- [ ] **API metrics:** p50/p95/p99 + cache hit ratio logged
- [ ] **Özet rapor:** `reports/crypto/w1_metrics.md` (perf + cache + notes)
- [ ] **Wallet reports:** `reports/crypto/wallets/<addr>.json` (5 cüzdan)

**Başarı Metrikleri:**
- API p95 (warm cache): **< 1s** (hedef: 0.5-0.8s)
- Cache hit ratio: **> 70%** (hedef: 80-85%)
- Error rate: **0%**
- Collector lag: **< 100 blocks**

---

## 🗓️ Günlük Detaylı Plan

### **D1 — Pazartesi: Kickoff & Baz Çizgisi**

**Hedef:** İlk koşu + baseline metrics + environment setup

#### AI (2-3 saat)

**Sabah (09:00-11:00):**
```bash
# 1. Workspace setup
cd week1_tensors
mkdir -p outputs/ai reports/ai

# 2. Sentetik veri oluştur
python -c "
import torch
torch.manual_seed(42)
X = torch.randn(1000, 5)
w_true = torch.tensor([1.5, -2.0, 0.5, 3.0, -1.0])
b_true = 0.5
y = X @ w_true + b_true + torch.randn(1000) * 0.1
torch.save({'X': X, 'y': y}, 'data/synthetic_data.pt')
print('✅ Sentetik veri hazır:', X.shape, y.shape)
"

# 3. İlk training run
python linreg_manual.py --lr 0.01 --epochs 100 --seed 42

# 4. Loss curve kontrolü
ls -lh outputs/ai/w1_loss_curve.png
```

**Çıktılar:**
- `data/synthetic_data.pt` - sentetik dataset
- `outputs/ai/w1_loss_curve.png` - ilk loss eğrisi
- `reports/day1_ai.txt` - initial MSE değeri

**Başarı Kriteri:**
- [ ] Training tamamlandı (hata vermedi)
- [ ] Loss curve düşüyor (overfitting yok)
- [ ] Final train MSE < 1.0

---

#### Crypto (2-3 saat)

**Öğleden Sonra (14:00-17:00):**
```bash
# 1. Environment check
make c.health
# Expected: RPC latency < 300ms, chain tip sync'd

# 2. Test wallets listesi oluştur
cat > data/wallets_w1.txt <<EOF
0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
0x0000000000000000000000000000000000000000
0x1111111111111111111111111111111111111111
0x2222222222222222222222222222222222222222
0x3333333333333333333333333333333333333333
EOF

# 3. Backfill last 5k blocks (idempotent)
make c.capture.idem BACKFILL=5000
# Expected: ~10-15 minutes, depends on RPC

# 4. DuckDB verification
duckdb onchain.duckdb -c "
SELECT 
  COUNT(*) AS total_transfers,
  MIN(block_number) AS min_block,
  MAX(block_number) AS max_block,
  COUNT(DISTINCT tx_hash) AS unique_txs
FROM transfers;
"

# 5. API start
make crypto.api
# Expected: Uvicorn starts on :8000

# 6. Baseline p95 measurement (cold cache)
WALLET=$(head -1 data/wallets_w1.txt)
hey -n 50 -c 5 "http://localhost:8000/wallet/$WALLET/report?hours=24" \
  | tee reports/day1_baseline_p95.txt

# 7. Warm cache test
hey -n 50 -c 5 "http://localhost:8000/wallet/$WALLET/report?hours=24" \
  | tee reports/day1_warm_p95.txt
```

**Çıktılar:**
- `data/wallets_w1.txt` - test wallet list
- `onchain.duckdb` - populated database (5k blocks)
- `reports/day1_baseline_p95.txt` - cold cache performance
- `reports/day1_warm_p95.txt` - warm cache performance

**Başarı Kriteri:**
- [ ] Ingest completed without errors
- [ ] Database has > 0 transfers
- [ ] API returns valid JSON
- [ ] Baseline p95 measured (no target yet)

---

**Günlük Özet Rapor:**
```bash
# reports/day1.md
cat > reports/day1.md <<EOF
# Day 1 — Kickoff & Baseline

## AI
- ✅ Sentetik veri: 1000 samples, 5 features
- ✅ İlk training run: Train MSE = X.XX
- ✅ Loss curve: outputs/ai/w1_loss_curve.png
- 📊 Baseline: Train MSE = X.XX (hedef: < 1.0)

## Crypto
- ✅ RPC health: latency = XXX ms
- ✅ Ingest: 5k blocks, XXX transfers
- ✅ API start: localhost:8000
- 📊 Baseline p95 (cold): XXX ms
- 📊 Baseline p95 (warm): XXX ms
- Cache hit ratio: X%

## Notlar
- [ ] AI: LR sweep yapılacak (D2)
- [ ] Crypto: Cache tuning başlayacak (D2)

## Riskler
- None (kickoff günü)
EOF
```

---

### **D2 — Salı: Stabilizasyon & Modülerlik**

**Hedef:** Modüler kod + Val split + Cache optimization başlangıcı

#### AI (3-4 saat)

```bash
# 1. nn.Module implementation
python linreg_module.py \
  --lr 0.01 \
  --l2 0.001 \
  --early_stop_patience 10 \
  --epochs 200 \
  --val_split 0.2 \
  --seed 42

# 2. Model checkpoint verification
python -c "
import torch
model = torch.load('outputs/ai/best_model.pt')
print('Model params:', model)
"

# 3. Train/Val MSE report
python scripts/eval_model.py \
  --model outputs/ai/best_model.pt \
  --data data/synthetic_data.pt \
  > reports/day2_ai_metrics.txt
```

**Çıktılar:**
- `week1_tensors/linreg_module.py` - nn.Module implementation
- `outputs/ai/best_model.pt` - checkpointed model
- `outputs/ai/w1_train_val_curve.png` - train/val split curves
- `reports/day2_ai_metrics.txt` - MSE summary

**Başarı Kriteri:**
- [ ] Val MSE < 0.6 (improvement from baseline)
- [ ] Early stopping triggered (patience=10)
- [ ] Model checkpoint loads successfully

---

#### Crypto (3-4 saat)

```bash
# 1. Cache configuration tuning
export NOVA_CACHE_TTL=120  # 2 minutes
export NOVA_CACHE_CAPACITY=4096

# Restart API
pkill -f uvicorn
make crypto.api

# 2. Test all 5 wallets
while read wallet; do
  echo "Testing $wallet..."
  curl -s "http://localhost:8000/wallet/$wallet/report?hours=24" \
    | jq '.meta.generated_at, .tx_count' \
    | tee -a reports/day2_wallet_results.txt
  sleep 2
done < data/wallets_w1.txt

# 3. Cache hit ratio measurement
for i in {1..20}; do
  WALLET=$(shuf -n 1 data/wallets_w1.txt)
  curl -s "http://localhost:8000/wallet/$WALLET/report?hours=24" > /dev/null
  sleep 1
done

# Check health endpoint for cache stats
curl -s localhost:8000/healthz | jq '{
  cache_size,
  cache_hit_rate,
  uptime_s
}' | tee reports/day2_cache_stats.txt

# 4. Schema validation for all wallets
while read wallet; do
  echo "Validating $wallet..."
  python crypto/w0_bootstrap/report_json.py \
    --wallet $wallet \
    --hours 24 \
    --validate \
    || echo "VALIDATION FAILED: $wallet"
done < data/wallets_w1.txt
```

**Çıktılar:**
- `reports/day2_wallet_results.txt` - 5 wallet outputs
- `reports/day2_cache_stats.txt` - cache metrics
- Cache hit ratio measurement

**Başarı Kriteri:**
- [ ] All 5 wallets return valid JSON
- [ ] Schema validation passes for all
- [ ] Cache hit ratio > 50%
- [ ] No 5xx errors

---

**Günlük Özet Rapor:**
```bash
cat > reports/day2.md <<EOF
# Day 2 — Stabilization & Modularity

## AI
- ✅ nn.Module implementation: linreg_module.py
- ✅ Train/Val split: 800/200
- ✅ Early Stopping: triggered at epoch XXX
- 📊 Train MSE = X.XX, Val MSE = X.XX
- ✅ Model checkpoint: best_model.pt

## Crypto
- ✅ Cache tuning: TTL=120s, capacity=4096
- ✅ 5 wallets tested: all valid JSON
- ✅ Schema validation: 5/5 pass
- 📊 Cache hit ratio: XX%
- 📊 p95 (sample): XXX ms

## Notlar
- [ ] AI: LR sweep tomorrow (D3)
- [ ] Crypto: Load test tomorrow (D3)

## Riskler
- None
EOF
```

---

### **D3 — Çarşamba: Performans & Ablation**

**Hedef:** Comprehensive testing + ablation study + performance optimization

#### AI (3-4 saat)

```bash
# 1. LR sweep
for lr in 0.1 0.05 0.01 0.005 0.001; do
  echo "Testing LR=$lr..."
  python linreg_module.py \
    --lr $lr \
    --l2 0.001 \
    --early_stop_patience 10 \
    --epochs 200 \
    --val_split 0.2 \
    --seed 42 \
    --output_prefix "lr_${lr}"
done

# 2. Ablation: L2 on/off
python linreg_module.py --lr 0.01 --l2 0.0 --output_prefix "no_l2"
python linreg_module.py --lr 0.01 --l2 0.001 --output_prefix "with_l2"

# 3. Ablation: Early Stopping on/off
python linreg_module.py --lr 0.01 --l2 0.001 --early_stop_patience 0 --output_prefix "no_early"
python linreg_module.py --lr 0.01 --l2 0.001 --early_stop_patience 10 --output_prefix "with_early"

# 4. Generate ablation report
python scripts/ablation_report.py \
  --results_dir outputs/ai \
  --output reports/ai/w1_ablation.md
```

**Çıktılar:**
- `outputs/ai/lr_*_loss_curve.png` - LR sweep curves
- `outputs/ai/no_l2_vs_with_l2.png` - L2 comparison
- `outputs/ai/no_early_vs_with_early.png` - Early stopping comparison
- `reports/ai/w1_ablation.md` - comprehensive ablation report

**Başarı Kriteri:**
- [ ] Best LR identified (lowest Val MSE)
- [ ] L2 impact quantified
- [ ] Early stopping benefit demonstrated
- [ ] Val MSE < 0.55

---

#### Crypto (3-4 saat)

```bash
# 1. Load test (cold cache)
pkill -f uvicorn
rm -f /tmp/cache.pkl  # If file-based cache
make crypto.api
sleep 5

WALLET=$(head -1 data/wallets_w1.txt)
hey -n 200 -c 20 "http://localhost:8000/wallet/$WALLET/report?hours=24" \
  | tee reports/day3_cold_load.txt

# 2. Load test (warm cache)
hey -n 200 -c 20 "http://localhost:8000/wallet/$WALLET/report?hours=24" \
  | tee reports/day3_warm_load.txt

# 3. Concurrent wallet load
seq 1 50 | xargs -P 10 -I {} sh -c '
  WALLET=$(shuf -n 1 data/wallets_w1.txt)
  curl -s "http://localhost:8000/wallet/$WALLET/report?hours=24" > /dev/null
'

# Check cache stats
curl -s localhost:8000/healthz | jq

# 4. TTL/Capacity tuning experiment
for TTL in 60 120 300; do
  for CAP in 1024 2048 4096; do
    echo "Testing TTL=$TTL, CAP=$CAP..."
    export NOVA_CACHE_TTL=$TTL
    export NOVA_CACHE_CAPACITY=$CAP
    pkill -f uvicorn
    make crypto.api
    sleep 5
    
    hey -n 100 -c 10 "http://localhost:8000/wallet/$WALLET/report?hours=24" \
      | grep "95%" \
      | tee -a reports/day3_cache_tuning.txt
  done
done

# 5. Identify optimal config
cat reports/day3_cache_tuning.txt | sort -k2 -n | head -1
```

**Çıktılar:**
- `reports/day3_cold_load.txt` - cold cache load test
- `reports/day3_warm_load.txt` - warm cache load test
- `reports/day3_cache_tuning.txt` - TTL/capacity experiment
- `reports/crypto/w1_perf_summary.md` - performance analysis

**Başarı Kriteri:**
- [ ] p95 (warm cache) < 1s ✅
- [ ] Error rate = 0%
- [ ] Cache hit ratio > 70%
- [ ] Optimal TTL/capacity identified

---

**Günlük Özet Rapor:**
```bash
cat > reports/day3.md <<EOF
# Day 3 — Performance & Ablation

## AI
- ✅ LR sweep: tested {0.1, 0.05, 0.01, 0.005, 0.001}
- ✅ Best LR: X.XX (Val MSE = X.XX)
- ✅ L2 ablation: improvement = X.XX MSE
- ✅ Early stopping ablation: saves XX epochs
- 📊 Best Val MSE = X.XX (target: < 0.55)

## Crypto
- ✅ Load test (cold): p95 = XXX ms
- ✅ Load test (warm): p95 = XXX ms ✅ (< 1s)
- ✅ Cache hit ratio: XX% (> 70%)
- ✅ Error rate: 0%
- ✅ Optimal config: TTL=XXX, CAP=XXXX

## Notlar
- [ ] AI: Final cleanup tomorrow (D4)
- [ ] Crypto: Error scenarios tomorrow (D4)

## Riskler
- None
EOF
```

---

### **D4 — Perşembe: Sertleştirme & Kenar Durumlar**

**Hedef:** Error handling + edge cases + code quality

#### AI (2-3 saat)

```bash
# 1. Code cleanup
ruff check week1_tensors/ --fix
ruff format week1_tensors/

# 2. Test suite
pytest tests/test_linreg.py -v --cov=week1_tensors --cov-report=term-missing

# 3. Edge cases
python scripts/test_edge_cases.py

# 4. Documentation
python scripts/generate_ai_summary.py > reports/ai/w1_summary.md
```

**Çıktılar:**
- Clean code (ruff pass)
- Test coverage report
- `reports/ai/w1_summary.md` draft

**Başarı Kriteri:**
- [ ] Ruff: 0 errors
- [ ] Tests: all pass
- [ ] Coverage ≥ 70%

---

#### Crypto (3-4 saat)

```bash
# 1. Error scenario: RPC 429
# Simulate by reducing rate limits (if possible) or manual trigger
python crypto/tests/simulate_429.py
# Follow Runbook R1 to recover
# Log: reports/day4_error_429.txt

# 2. Error scenario: Reorg
# Simulate by clearing last 20 blocks and re-ingesting
duckdb onchain.duckdb -c "
DELETE FROM transfers WHERE block_number > (SELECT MAX(block_number) - 20 FROM transfers);
UPDATE scan_state SET last_scanned_block = last_scanned_block - 20;
"
make c.capture.idem BACKFILL=30
# Verify: no duplicates, no gaps
./crypto/scripts/data_quality_check.sh onchain.duckdb
# Log: reports/day4_error_reorg.txt

# 3. Error scenario: Bad window (10k limit)
# Test with very large window
curl -s "http://localhost:8000/wallet/0x0000000000000000000000000000000000000000/report?hours=720"
# Should handle gracefully or return error
# Log: reports/day4_error_window.txt

# 4. Contract tests
pytest tests/contract/test_report_schema.py -v

# 5. Schema validation CI
python crypto/w0_bootstrap/report_json.py \
  --wallet $(head -1 data/wallets_w1.txt) \
  --hours 24 \
  --validate \
  || exit 1
```

**Çıktılar:**
- `reports/day4_error_*.txt` - error scenario logs
- Contract test results
- Data quality check pass

**Başarı Kriteri:**
- [ ] All error scenarios recovered successfully
- [ ] Contract tests pass
- [ ] Data quality checks pass
- [ ] No regressions introduced

---

**Günlük Özet Rapor:**
```bash
cat > reports/day4.md <<EOF
# Day 4 — Hardening & Edge Cases

## AI
- ✅ Code cleanup: ruff pass
- ✅ Tests: XX/XX pass (coverage = XX%)
- ✅ Edge cases tested
- ✅ w1_summary.md draft ready

## Crypto
- ✅ Error scenario 1: RPC 429 → recovered via Runbook R1
- ✅ Error scenario 2: Reorg → tail rescan successful
- ✅ Error scenario 3: Large window → handled gracefully
- ✅ Contract tests: 12/12 pass
- ✅ Data quality: all checks pass

## Notlar
- [ ] Tomorrow (D5): Final run + ship

## Riskler
- None
EOF
```

---

### **D5 — Cuma: Ship Day**

**Hedef:** Final run + documentation + release

#### AI (2 saat)

```bash
# 1. Final training run with best params
python linreg_module.py \
  --lr 0.01 \
  --l2 0.001 \
  --early_stop_patience 10 \
  --epochs 200 \
  --val_split 0.2 \
  --seed 42 \
  --output_prefix "final"

# 2. Verify Val MSE ≤ 0.50
python -c "
import torch
checkpoint = torch.load('outputs/ai/final_best_model.pt')
print(f'Val MSE: {checkpoint[\"val_mse\"]:.4f}')
assert checkpoint['val_mse'] <= 0.50, 'Val MSE target not met!'
print('✅ Val MSE target met!')
"

# 3. Finalize summary report
python scripts/generate_ai_summary.py \
  --include_graphs \
  --include_ablation \
  > reports/ai/w1_summary.md

# 4. Archive outputs
tar -czf outputs/ai/w1_final.tar.gz outputs/ai/*.png outputs/ai/*.pt
```

**Çıktılar:**
- `outputs/ai/final_best_model.pt` - final model checkpoint
- `reports/ai/w1_summary.md` - complete summary with graphs
- `outputs/ai/w1_final.tar.gz` - archived outputs

**Başarı Kriteri:**
- [x] Val MSE ≤ 0.50 ✅
- [x] Summary report complete
- [x] All graphs included

---

#### Crypto (3 saat)

```bash
# 1. Generate reports for all 5 wallets
mkdir -p reports/crypto/wallets
while read wallet; do
  echo "Generating report for $wallet..."
  python crypto/w0_bootstrap/report_json.py \
    --wallet $wallet \
    --hours 24 \
    --output "reports/crypto/wallets/${wallet}.json" \
    --validate
done < data/wallets_w1.txt

# 2. Final performance measurement
WALLET=$(head -1 data/wallets_w1.txt)
hey -n 200 -c 20 "http://localhost:8000/wallet/$WALLET/report?hours=24" \
  > reports/day5_final_perf.txt

# Extract metrics
cat reports/day5_final_perf.txt | grep -E "(p50|p95|p99)"

# 3. Generate w1_metrics.md
python scripts/generate_crypto_summary.py \
  --perf_file reports/day5_final_perf.txt \
  --cache_stats "$(curl -s localhost:8000/healthz)" \
  > reports/crypto/w1_metrics.md

# 4. Data quality final check
./crypto/scripts/data_quality_check.sh onchain.duckdb \
  | tee reports/day5_data_quality.txt

# 5. Archive
tar -czf reports/crypto/w1_wallets.tar.gz reports/crypto/wallets/
```

**Çıktılar:**
- `reports/crypto/wallets/<addr>.json` - 5 wallet reports
- `reports/crypto/w1_metrics.md` - performance summary
- `reports/day5_final_perf.txt` - final load test results
- `reports/day5_data_quality.txt` - quality check results

**Başarı Kriteri:**
- [x] 5 wallets reports generated ✅
- [x] p95 < 1s ✅
- [x] Cache hit ratio > 70% ✅
- [x] Error rate = 0% ✅
- [x] w1_metrics.md complete

---

#### Release (1 saat)

```bash
# 1. Create WEEK1_CLOSEOUT.md
cat > reports/WEEK1_CLOSEOUT.md <<EOF
# Week 1 Closeout — AI + Crypto Sprint

## 🎯 Hedefler & Başarı

### AI: Linear Regression from Scratch
- ✅ Val MSE: **X.XX** (hedef: ≤ 0.50)
- ✅ Convergence: XX epochs (hedef: < 200)
- ✅ Tests: XX/XX pass (coverage: XX%)
- ✅ Ablation: LR=X.XX, L2=X.XXX optimal

### Crypto: Collector + API Performance
- ✅ API p95 (warm): **XXX ms** (hedef: < 1s)
- ✅ Cache hit ratio: **XX%** (hedef: > 70%)
- ✅ Error rate: **0%**
- ✅ Load test: 200 req, 0 errors

## 📊 Metrikler

| Kategori   | Metrik          | Hedef     | Gerçekleşen | Durum |
| ---------- | --------------- | --------- | ----------- | ----- |
| AI         | Val MSE         | ≤ 0.50    | X.XX        | ✅    |
| AI         | Convergence     | < 200 ep  | XX          | ✅    |
| Crypto     | API p95 (warm)  | < 1s      | XXX ms      | ✅    |
| Crypto     | Cache hit ratio | > 70%     | XX%         | ✅    |
| Crypto     | Error rate      | 0%        | 0%          | ✅    |

## 📦 Deliverables

### AI
- ✅ Code: \`week1_tensors/linreg_module.py\`
- ✅ Model: \`outputs/ai/final_best_model.pt\`
- ✅ Report: \`reports/ai/w1_summary.md\`
- ✅ Graphs: \`outputs/ai/*.png\`

### Crypto
- ✅ Collector: \`crypto/collector/\`
- ✅ API: \`crypto/service/\`
- ✅ Reports: \`reports/crypto/wallets/*.json\`
- ✅ Metrics: \`reports/crypto/w1_metrics.md\`

## 🔗 Links

- [AI Summary](./ai/w1_summary.md)
- [Crypto Metrics](./crypto/w1_metrics.md)
- [Wallet Reports](./crypto/wallets/)
- [Daily Reports](./day1.md) ... [day5.md](./day5.md)

## 🚀 Next Steps (Week 2)

### AI
- [ ] Logistic Regression (binary classification)
- [ ] Multi-class classification
- [ ] Cross-entropy loss

### Crypto
- [ ] Price integration (CoinGecko API)
- [ ] USD value calculation
- [ ] Historical snapshots

## 📝 Lessons Learned

### What Went Well
- Clear daily goals
- Parallel AI + Crypto tracks
- Automated testing
- Performance targets met

### What Could Improve
- [Add specific improvements]

### Surprises
- [Add surprising findings]
EOF

# 2. Update CHANGELOG
cat >> CHANGELOG.md <<EOF

## [1.2.0-w1] - $(date +%Y-%m-%d)

### AI Track
- Linear Regression from scratch (manual GD + nn.Module)
- Train/Val split + Early Stopping + L2 regularization
- Val MSE achieved: X.XX (target: ≤ 0.50)

### Crypto Track
- Collector loop (30s polling, idempotent + tail rescan)
- API performance optimized (p95 < 1s warm cache)
- 5 wallet reports generated
- Cache hit ratio: XX%

### Infrastructure
- Test coverage improved (≥ 70%)
- Performance testing automated
- Error handling hardened
EOF

# 3. Git tag
git add -A
git commit -m "feat(w1): Complete Week 1 Sprint - AI LR (MSE=X.XX) + Crypto API (p95=XXXms)

AI Track:
- Linear Regression from scratch
- Val MSE: X.XX (target: ≤ 0.50) ✅
- Convergence: XX epochs
- Ablation study complete

Crypto Track:
- API p95 (warm): XXX ms (< 1s) ✅
- Cache hit ratio: XX% (> 70%) ✅
- Error rate: 0%
- 5 wallet reports generated

Deliverables:
- reports/WEEK1_CLOSEOUT.md
- reports/ai/w1_summary.md
- reports/crypto/w1_metrics.md
- outputs/ai/final_best_model.pt
- reports/crypto/wallets/*.json
"

git tag -a v1.2.0-w1 -m "Week 1 Release: AI + Crypto Sprint Complete"
git push && git push --tags

echo "🎉 Week 1 Complete! Tag: v1.2.0-w1"
```

---

## 🧪 Test & Kalite (W1)

### AI Tests
```bash
# Unit tests
pytest tests/test_linreg.py -v

# Coverage
pytest tests/test_linreg.py --cov=week1_tensors --cov-report=html

# Linting
ruff check week1_tensors/
ruff format week1_tensors/ --check
```

**Target:**
- Tests: **all pass**
- Coverage: **≥ 70%**
- Ruff: **0 errors**

### Crypto Tests
```bash
# Unit tests
pytest tests/unit/ -v

# Contract tests (schema)
pytest tests/contract/ -v

# Integration tests
pytest tests/integration/ -v

# Load test
hey -n 200 -c 20 "http://localhost:8000/wallet/$W/report?hours=24"
```

**Target:**
- Tests: **all pass**
- Schema validation: **100%**
- Load test error rate: **0%**
- p95 (warm cache): **< 1s**

---

## 🧰 Komut Kartı (Cheatsheet)

### AI Quick Commands
```bash
# Training
python linreg_module.py --lr 0.01 --l2 0.001 --early_stop_patience 10

# Evaluation
python scripts/eval_model.py --model outputs/ai/best_model.pt

# Tests
pytest tests/test_linreg.py -q

# Lint
ruff check week1_tensors/ --fix
```

### Crypto Quick Commands
```bash
# Health check
make c.health
curl -s localhost:8000/healthz | jq

# Ingest (backfill 5k blocks)
make c.capture.idem BACKFILL=5000

# API start
make crypto.api

# Single wallet report
WALLET=0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
curl -s "http://localhost:8000/wallet/$WALLET/report?hours=24" | jq

# Load test
hey -n 200 -c 20 "http://localhost:8000/wallet/$WALLET/report?hours=24"

# Schema validation
python crypto/w0_bootstrap/report_json.py --wallet $WALLET --hours 24 --validate

# Data quality
./crypto/scripts/data_quality_check.sh onchain.duckdb
```

### Combined Workflow
```bash
# Morning routine
make c.health                    # Check RPC
make c.capture.idem BACKFILL=100 # Catch up
make crypto.api                  # Start API
pytest tests/ -q                 # Run all tests

# Evening routine
git add -A
git commit -m "day X: progress update"
tar -czf backups/day$(date +%d).tar.gz outputs/ reports/
```

---

## 📦 Çıktı Klasörleri

```
novadev-protocol/
├── outputs/
│   └── ai/
│       ├── w1_loss_curve.png
│       ├── w1_train_val_curve.png
│       ├── w1_ablation.png
│       ├── final_best_model.pt
│       └── w1_final.tar.gz
│
├── reports/
│   ├── day1.md
│   ├── day2.md
│   ├── day3.md
│   ├── day4.md
│   ├── day5.md
│   ├── WEEK1_CLOSEOUT.md
│   ├── ai/
│   │   ├── w1_summary.md
│   │   └── w1_ablation.md
│   └── crypto/
│       ├── w1_metrics.md
│       ├── wallets/
│       │   ├── 0xd8dA....json
│       │   ├── 0x0000....json
│       │   └── ... (5 total)
│       └── w1_wallets.tar.gz
│
├── data/
│   ├── synthetic_data.pt
│   └── wallets_w1.txt
│
└── logs/
    └── api/
        └── YYYY-MM-DD.jsonl
```

---

## ⚠️ Risk → Önlem

### AI Risks

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **Overfitting** | Train MSE ↓, Val MSE ↑ | Increase L2, reduce epochs, early stopping |
| **Underfitting** | Both MSE high | Increase LR, more epochs, reduce L2 |
| **Divergence** | Loss → NaN or ∞ | Reduce LR, check data normalization |
| **Slow convergence** | Loss plateau | Increase LR, check gradients |

### Crypto Risks

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **RPC 429** | Rate limit errors | AIMD window ↓, token bucket ↑, backoff+jitter |
| **RPC Timeout** | Slow responses | Retry logic, backup RPC, reduce window size |
| **DB Lock** | Write errors | API read-only, single writer process |
| **Cache Stampede** | p95 spike after restart | Warmup script, single-flight pattern |
| **Schema Drift** | Validation fails | Schema-check CI, version strategy |
| **Memory Leak** | RSS grows | Profile, fix leaks, restart schedule |

### General Risks

| Risk | Symptom | Mitigation |
|------|---------|------------|
| **Time Crunch** | Tasks not finished | Prioritize DoD items, defer stretch goals |
| **Scope Creep** | Too many features | Stick to master plan, defer to W2 |
| **Integration Issues** | Tests fail | Incremental integration, CI checks |

---

## 📊 Skor Tahtası (Takip Edilecek Metrikler)

### AI Metrics

| Metrik | Baseline (D1) | Target (D5) | Actual (D5) | Status |
|--------|---------------|-------------|-------------|--------|
| Train MSE | 0.80 | < 0.45 | X.XX | ⏳ |
| Val MSE | 0.85 | ≤ 0.50 | X.XX | ⏳ |
| Convergence (epochs) | 150 | < 200 | XX | ⏳ |
| Test Coverage | 50% | ≥ 70% | XX% | ⏳ |

### Crypto Metrics

| Metrik | Baseline (D1) | Target (D5) | Actual (D5) | Status |
|--------|---------------|-------------|-------------|--------|
| API p95 (cold cache) | 2.5s | N/A | XXX ms | ⏳ |
| API p95 (warm cache) | 1.2s | < 1s | XXX ms | ⏳ |
| Cache hit ratio | 30% | > 70% | XX% | ⏳ |
| Error rate | 2% | 0% | X% | ⏳ |
| Collector lag (blocks) | 200 | < 100 | XX | ⏳ |

### Quality Metrics

| Metrik | Target | Actual | Status |
|--------|--------|--------|--------|
| Tests Pass | 100% | XX% | ⏳ |
| Ruff Errors | 0 | X | ⏳ |
| Schema Validation | 100% | XX% | ⏳ |
| Data Quality Checks | All pass | X/5 | ⏳ |

---

## 🧩 PR/Issue Şablonları

### Pull Request Template

```markdown
# Week 1 Sprint: [AI/Crypto] [Feature]

## 🎯 Objective
[Brief description of what this PR accomplishes]

## 📊 Metrics
- **AI:** Val MSE = X.XX (target: ≤ 0.50)
- **Crypto:** p95 = XXX ms (target: < 1s)
- **Tests:** XX/XX pass
- **Coverage:** XX%

## 🔄 Changes
- [ ] Feature 1
- [ ] Feature 2
- [ ] Tests added/updated
- [ ] Documentation updated

## 🧪 Testing
```bash
# AI
pytest tests/test_linreg.py -v

# Crypto
pytest tests/contract/ tests/integration/ -v
hey -n 100 -c 10 "http://localhost:8000/wallet/$W/report?hours=24"
```

## 📸 Screenshots
[Attach graphs, performance charts, etc.]

## ⚠️ Risks
- [Any risks or concerns]

## 🔗 Related
- Closes #X
- Related to #Y
```

### Issue Templates

#### AI Issue
```markdown
# [W1-AI] Feature/Bug Name

## Description
[Detailed description]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Steps to Reproduce
1. Step 1
2. Step 2

## Metrics
- Train MSE: X.XX
- Val MSE: X.XX

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
```

#### Crypto Issue
```markdown
# [W1-CRYPTO] Feature/Bug Name

## Description
[Detailed description]

## Current Behavior
[What's happening now]

## Desired Behavior
[What should happen]

## Metrics
- p95: XXX ms
- Cache hit ratio: XX%
- Error rate: X%

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
```

---

## 🌱 Stretch Goals (Opsiyonel)

### AI Stretch Goals
- [ ] Implement batch gradient descent
- [ ] Add momentum optimizer
- [ ] Learning rate scheduler
- [ ] Feature normalization
- [ ] Cross-validation (k-fold)

### Crypto Stretch Goals
- [ ] Prometheus `/metrics` endpoint
- [ ] Grafana dashboard
- [ ] Canary deployment (A/B test)
- [ ] Dockerfile + docker-compose
- [ ] Horizontal scaling (load balancer)
- [ ] Price API integration (CoinGecko)
- [ ] USD value in reports

### Infrastructure Stretch Goals
- [ ] Pre-commit hooks
- [ ] GitHub Actions CI/CD
- [ ] Code coverage badges
- [ ] Automated deployment script
- [ ] Monitoring alerts (PagerDuty/Slack)

---

## ✅ Kapanış Kriteri (Cuma 18:00 TR Time)

### AI Track DoD
- [ ] Val MSE ≤ 0.50 ✅
- [ ] Tests all pass (coverage ≥ 70%)
- [ ] Model checkpoint saved (`outputs/ai/final_best_model.pt`)
- [ ] Summary report complete (`reports/ai/w1_summary.md`)
- [ ] Graphs generated (loss curves, ablation)
- [ ] Code clean (ruff pass)

### Crypto Track DoD
- [ ] API p95 < 1s (warm cache) ✅
- [ ] Load test: 100 req, error=0
- [ ] Cache hit ratio > 70%
- [ ] 5 wallet reports generated (`reports/crypto/wallets/*.json`)
- [ ] Metrics report complete (`reports/crypto/w1_metrics.md`)
- [ ] Schema validation: 100% pass
- [ ] Data quality checks: all pass

### Release DoD
- [ ] `WEEK1_CLOSEOUT.md` complete
- [ ] CHANGELOG updated
- [ ] Git tag created: `v1.2.0-w1`
- [ ] All commits pushed
- [ ] Tag pushed

### Verification Commands
```bash
# AI
python -c "
import torch
checkpoint = torch.load('outputs/ai/final_best_model.pt')
assert checkpoint['val_mse'] <= 0.50, f'Val MSE {checkpoint[\"val_mse\"]:.4f} > 0.50'
print('✅ AI DoD met: Val MSE =', checkpoint['val_mse'])
"

# Crypto
WALLET=$(head -1 data/wallets_w1.txt)
P95=$(hey -n 100 -c 10 "http://localhost:8000/wallet/$WALLET/report?hours=24" 2>&1 | grep "95%" | awk '{print $2}')
echo "p95: $P95"
if (( $(echo "$P95 < 1.0" | bc -l) )); then
  echo "✅ Crypto DoD met: p95 < 1s"
else
  echo "❌ Crypto DoD NOT met: p95 >= 1s"
fi

# Release
git tag | grep v1.2.0-w1 && echo "✅ Release DoD met: tag exists"
```

---

## 📝 Notlar

### Timezone
- Tüm saatler **TR (Europe/Istanbul)** lokal planlandı
- Veride **UTC** kullanılır (code standardı)
- Timestamp format: ISO 8601 with `Z` suffix

### Security
- **Private keys:** NEVER in code/config
- **Read-only:** API ve analysis only
- **Testnet:** Sepolia öncelikli
- **Environment:** `.env` files in `.gitignore`

### Documentation
- Günlük raporlar: `reports/dayX.md`
- Weekly closeout: `reports/WEEK1_CLOSEOUT.md`
- API docs: `crypto/API_GUIDE.md`
- Troubleshooting: `crypto/docs/w0_bootstrap/10_tahta_troubleshoot_runbooks.md`

### Communication
- Daily standup: 09:00 TR (optional)
- End-of-day summary: 18:00 TR
- Blockers: immediate Slack/Discord ping
- Wins: celebrate in team channel!

---

**Version:** 1.0  
**Last Updated:** 2025-10-06  
**Status:** Active (Week 1)  
**Next:** Week 2 Planning (Logistic Regression + Price Integration)

---

🎯 **Let's ship it!** Week 1 begins Monday 09:00 TR. Ready? 🚀

