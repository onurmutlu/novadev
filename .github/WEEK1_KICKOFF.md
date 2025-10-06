# Week 1 Kickoff: Linear Regression + Collector Loop

**Week Start:** [Set Date]  
**Week End:** [Set Date]

---

## 🎯 Week 1 Goals

### AI Week 1: Linear Regression

**Target Metric:** Val MSE < 0.5

**Core Concepts:**
- Tensors, autograd, optimizers
- Train/val split
- Loss curves & early stopping
- Hyperparameter sweep (LR)

**Deliverables:**
- ✅ `data_synth.py` (synthetic data, seed controlled)
- ✅ `linreg_manual.py` (manual GD loop)
- ✅ `linreg_module.py` (nn.Module + early stop)
- ✅ `train.py` (LR sweep: 1e-2, 5e-3, 1e-3)
- ✅ `outputs/w1_loss_curve.png`
- ✅ `reports/w1_summary.md` (5 bullet insights)

### Crypto Week 1: Collector Loop + Report Optimization

**Target Metric:** `/report` endpoint p95 < 1s

**Core Concepts:**
- Idempotent state tracking
- Polling loop (30s interval)
- Price feeds (CoinGecko stub)
- Query optimization (DuckDB indexes)

**Deliverables:**
- ✅ `crypto/w1_ingest/collector_loop.py` (cron/launchd config)
- ✅ `crypto/w1_ingest/price_cache.py` (CoinGecko + cache)
- ✅ `/wallet/{addr}/report` p95 < 1s (local benchmark)
- ✅ `make benchmark` target
- ✅ `crypto/w1_ingest/README.md` (setup + cron example)

---

## 📅 Daily Plan

### Monday: Setup & Baseline

**AI (60-90 min):**
- [ ] Review Week 0 theory (linear models, GD)
- [ ] `data_synth.py`: Generate synthetic data (N=1000, D=5, seed=42)
- [ ] Plot data: scatter matrix
- [ ] **Commit:** `W1D1: data_synth + scatter plot`

**Crypto (45-60 min):**
- [ ] Review W0: `capture_transfers_idempotent.py`
- [ ] Design collector loop (pseudo-code)
- [ ] Research CoinGecko API (rate limits, caching)
- [ ] **Commit:** `W1D1: crypto collector design`

**Evening:**
- [ ] Log: `exp_log.csv` (baseline plan)
- [ ] Tomorrow's goal: AI manual GD, Crypto loop prototype

---

### Tuesday: Manual GD + Loop Prototype

**AI (60-90 min):**
- [ ] `linreg_manual.py`: Manual gradient descent
  - Forward: y_pred = X @ w + b
  - Loss: MSE
  - Backward: manual grad computation
  - Update: w -= lr * grad_w
- [ ] Train/val split (80/20)
- [ ] LR = 1e-2, 100 epochs
- [ ] **Target:** Val MSE < 1.0 (baseline)
- [ ] **Commit:** `W1D2: manual GD, val MSE=X.XX`

**Crypto (45-60 min):**
- [ ] `collector_loop.py`: Prototype
  - Read last state from DB
  - Fetch latest block
  - Capture events (idempotent)
  - Sleep 30s
  - Loop
- [ ] Test: run for 5 minutes
- [ ] **Commit:** `W1D2: collector loop prototype`

**Evening:**
- [ ] Log: Manual GD insights (LR effects)
- [ ] Tomorrow: nn.Module + price feeds

---

### Wednesday: nn.Module + Price Feeds

**AI (60-90 min):**
- [ ] `linreg_module.py`: PyTorch nn.Module
  - `nn.Linear(D, 1)`
  - Adam optimizer
  - Early stopping (patience=3)
- [ ] LR = 5e-3
- [ ] **Target:** Val MSE < 0.7
- [ ] Save model checkpoint
- [ ] **Commit:** `W1D3: nn.Module, val MSE=X.XX, early stop`

**Crypto (45-60 min):**
- [ ] `price_cache.py`: CoinGecko stub
  - Fetch ETH/USDT price
  - Cache (5 min TTL)
  - Fallback to last known price
- [ ] Integrate into `/wallet/{addr}/report`
- [ ] Add `price_usd` field to JSON
- [ ] **Commit:** `W1D3: price cache + /report price field`

**Evening:**
- [ ] Log: Early stopping worked? Price cache hit rate?
- [ ] Tomorrow: LR sweep + benchmarking

---

### Thursday: LR Sweep + Benchmark

**AI (60-90 min):**
- [ ] `train.py`: LR sweep
  - LRs: [1e-2, 5e-3, 1e-3]
  - Log all to `exp_log.csv`
  - Pick best model (val MSE)
- [ ] **Target:** Val MSE < 0.5 ✅
- [ ] Plot loss curves (all LRs)
- [ ] **Commit:** `W1D4: LR sweep, best MSE=X.XX`

**Crypto (45-60 min):**
- [ ] Benchmark `/wallet/{addr}/report`
  - Tool: `ab` (ApacheBench) or `wrk`
  - Requests: 100
  - Concurrency: 10
  - Measure: p50, p95, p99
- [ ] **Target:** p95 < 1s
- [ ] Add DuckDB index if needed: `CREATE INDEX idx_block ON transfers(block_number)`
- [ ] **Commit:** `W1D4: /report p95=XXXms`

**Evening:**
- [ ] Log: Best LR? What helped p95?
- [ ] Tomorrow: Final polish + reports

---

### Friday: Reports & Demo

**AI (60-90 min):**
- [ ] Generate `outputs/w1_loss_curve.png`
  - 3 curves (LR sweep)
  - Mark best model
- [ ] Write `reports/w1_summary.md`:
  1. Data: N=1000, D=5, synthetic
  2. Best LR: X.XXX (val MSE=X.XX)
  3. Early stopping: stopped at epoch Y
  4. Manual vs nn.Module: insights
  5. Next: MLP (W2)
- [ ] **Commit:** `W1D5: AI report + loss curve`

**Crypto (45-60 min):**
- [ ] Write `crypto/w1_ingest/README.md`:
  - Setup instructions
  - Cron example: `*/30 * * * * cd /path && python collector_loop.py`
  - Benchmark results
  - Schema updates
- [ ] Test full pipeline:
  1. `make c.capture.idem` (manual)
  2. Start collector loop
  3. `make c.api`
  4. `curl /wallet/.../report`
- [ ] **Commit:** `W1D5: crypto W1 complete, p95=XXXms ✅`

**Evening:**
- [ ] **Weekly commit:**
  ```bash
  git add -A
  git commit -m "Week 1 Complete: AI MSE=X.XX ✅, Crypto /report p95=XXXms ✅"
  git tag w1-complete
  git push origin master --tags
  ```
- [ ] Update `CHANGELOG.md`: Week 1 section
- [ ] Plan Week 2: MLP + Telegram bot

---

## 📊 Success Criteria (DoD)

### AI Week 1
- [x] Val MSE < 0.5
- [ ] Loss curve generated
- [ ] 5-bullet summary written
- [ ] Code in `week1_tensors/`
- [ ] Tests pass: `pytest tests/test_week1.py`

### Crypto Week 1
- [ ] `/report` p95 < 1s
- [ ] Collector loop runs (30s interval)
- [ ] Price cache working (CoinGecko stub)
- [ ] Benchmark results documented
- [ ] Code in `crypto/w1_ingest/`

---

## 🛠️ Tools & Commands

### AI

```bash
# Data generation
python week1_tensors/data_synth.py

# Manual GD
python week1_tensors/linreg_manual.py

# nn.Module
python week1_tensors/linreg_module.py

# LR sweep
python week1_tensors/train.py --lr_sweep

# Tests
pytest tests/test_week1.py -v
```

### Crypto

```bash
# Collector loop (foreground test)
python crypto/w1_ingest/collector_loop.py

# Collector loop (background)
nohup python crypto/w1_ingest/collector_loop.py > collector.log 2>&1 &

# API
make c.api

# Benchmark
ab -n 100 -c 10 http://localhost:8000/wallet/0x.../report

# Check logs
tail -f collector.log
```

---

## 📝 Notes & Blockers

### Blockers

- [ ] RPC rate limit? → Solution: local caching
- [ ] MPS memory? → Solution: smaller batch size
- [ ] CoinGecko API key? → Solution: use public endpoint (rate limited)

### Learnings

<!-- Fill as you go -->

- AI: 
- Crypto:

---

## 📚 Reading

### AI
- Week 0 theory review: `theory_foundations.md`
- PyTorch docs: `torch.nn.Linear`, `torch.optim.Adam`

### Crypto
- DuckDB docs: Indexes, query optimization
- CoinGecko API: https://www.coingecko.com/en/api/documentation

---

## 🔗 Related

- Week 0 Summary: `week0_setup/theory_closure.md`
- Crypto W0: `crypto/w0_bootstrap/README.md`
- Program Overview: `docs/program_overview.md`

---

**Week 1 Kickoff — Let's Ship! 🚀**

*AI: MSE < 0.5*  
*Crypto: p95 < 1s*  
*Discipline: 1 commit/day*
