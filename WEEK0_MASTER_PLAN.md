# 🚀 NovaDev — WEEK 0 MASTER PLAN

> **Hafta:** 0 — Setup + Theory + Bootstrap  
> **Süre:** 5 gün (15-20 saat)  
> **Mod:** Read-only (crypto), testnet-first, **yatırım tavsiyesi değildir**  
> **Tarih:** Esnek — *Europe/Istanbul*

---

## 📋 İçindekiler

- [Amaç & KPI'lar](#-amaç--kpilar)
- [Ön Koşullar](#-ön-koşullar)
- [Günlük Plan (5 Gün)](#-günlük-plan-5-gün)
- [Teslimatlar (DoD)](#-teslimatlar-dod)
- [Test & Ölçüm](#-test--ölçüm-rehberi)
- [Komut Kartı](#-komut-hızlı-kartı)
- [Riskler & Önlemler](#-riskler--önlemler)
- [Kapanış Checklist](#-hafta-sonu-kapanış-checklist)
- [PR/Issue Şablonları](#-prissue-şablonları)

---

## 🎯 Amaç & KPI'lar

### Week 0 Vision

```
╔════════════════════════════════════════════════════════════╗
║                   WEEK 0: FOUNDATION                       ║
╠════════════════════════════════════════════════════════════╣
║  AI Track:    Setup + Theory (Tensors → LR)              ║
║  Crypto Track: RPC + Idempotent Ingest + Report          ║
║  Quality:     CI/CD + Docs + Schema Validation           ║
║                                                            ║
║  Output: Paralel program için sağlam temel               ║
╚════════════════════════════════════════════════════════════╝
```

### 🤖 AI Track (Week 0)

**Hedef:** Geliştirme ortamı + PyTorch MPS + Linter/Test + Teori çekirdeği

**KPI'lar:**

| Metrik | Target | Nasıl Ölçülür |
|--------|--------|---------------|
| **MPS aktif** | ✅ | `hello_tensor.py` → "MPS is available!" |
| **Tests green** | 100% | `pytest -q` exit code 0 |
| **Lint clean** | 0 error | `ruff check .` exit code 0 |
| **Theory coverage** | 100% | `theory_closure.md` all checkboxes ✓ |

**Artefaktlar:**
- `week0_setup/nova-setup.md` (kurulum log)
- `reports/w0_ai_theory_closure.md` (self-assessment)
- `week0_setup/hello_tensor.py` (MPS validation)

---

### 🪙 Crypto Track (Week 0)

**Hedef:** RPC health + Idempotent ingest + State tracking + Wallet report + Schema + Mini API

**KPI'lar:**

| Metrik | Target | Nasıl Ölçülür |
|--------|--------|---------------|
| **RPC latency** | < 300ms | `rpc_health.py` p95 |
| **Duplicate logs** | 0 | SQL: `COUNT(*) GROUP BY (tx,log_idx) HAVING >1` |
| **Schema valid** | ✅ | `validate_report.py` exit 0 |
| **API health** | 200 OK | `curl /healthz` |
| **Reorg safe** | ✅ | `CONFIRMATIONS ≥ 5` + tail re-scan |

**Artefaktlar:**
- `onchain.duckdb` (state + logs)
- `reports/w0_crypto_summary.md` (metrics)
- `crypto/w0_bootstrap/.env` (config)

---

### 📚 Docs & Quality Track

**Hedef:** CI green + Docs structure + Tahta lessons + Schema

**KPI'lar:**

| Metrik | Target | Nasıl Ölçülür |
|--------|--------|---------------|
| **Docs CI** | ✅ | `make docs.check` green |
| **Python CI** | ✅ | `make py.ci` green |
| **Tahta lessons** | ≥ 2 | 01-02 complete (03-04 bonus) |
| **README hierarchy** | ✅ | Links working, structure clear |

**Artefaktlar:**
- `crypto/docs/w0_bootstrap/01_tahta_evm_giris.md` (1,277 lines)
- `crypto/docs/w0_bootstrap/02_tahta_rpc_101.md` (1,012 lines)
- `crypto/docs/w0_bootstrap/03_tahta_transfer_anatomi.md` (1,094 lines) *bonus*
- `crypto/docs/w0_bootstrap/04_tahta_getlogs_pencere_reorg.md` (2,266 lines) *bonus*
- `schemas/report_v1.json`
- `CHANGELOG.md`, `COMMANDS.md`

---

## ✅ Ön Koşullar

### Donanım
- macOS Apple Silicon (M1/M2/M3) 🧠
- 16GB+ RAM (önerilen)
- 50GB+ disk (DuckDB + models)

### Yazılım
```bash
# Python
python --version  # 3.11+ required

# Git
git --version

# Make
make --version

# (Optional) Homebrew
brew --version
```

### Kurulum
```bash
# 1. Clone repo
git clone <repo-url>
cd novadev-protocol

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -e ".[dev,crypto]"

# 4. Verify
python week0_setup/hello_tensor.py  # Should see "MPS is available!"
pytest -q                            # Tests should pass
ruff check .                         # Should be clean
```

### Crypto Specific
```bash
# 1. RPC Provider (pick one)
# - Alchemy: https://dashboard.alchemy.com
# - Infura: https://infura.io
# → Create Sepolia testnet app

# 2. Configure .env
cd crypto/w0_bootstrap
cp .env.example .env
# vim .env → paste your RPC_URL

# 3. Test
python rpc_health.py
# Expected: ✅ RPC OK | latest block: XXXXX | <300ms
```

---

## 🗓️ Günlük Plan (5 Gün)

### Day 1: Environment Setup + Health Checks

**Time:** 2-3 hours

#### AI Block (60-90 min)
```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Verify
python week0_setup/hello_tensor.py
# Expected: "MPS is available! ✓"

# Quality checks
ruff check .
pytest -q

# Document
echo "## W0D1 Setup Log" > week0_setup/nova-setup.md
# Log: Python version, MPS status, dependencies
```

#### Crypto Block (45-60 min)
```bash
cd crypto/w0_bootstrap

# Configure
cp .env.example .env
# Edit: RPC_URL, CHAIN_ID=11155111 (Sepolia)

# Health check
python rpc_health.py
# Target: latency < 300ms, no errors

# Log results
echo "RPC: $(date) | latency_ms: XXX" >> ../../reports/daily_log.md
```

#### DoD (Day 1)
- [ ] MPS verified ✓
- [ ] pytest + ruff green
- [ ] RPC healthy (< 300ms)
- [ ] `nova-setup.md` created

---

### Day 2: Theory Foundation + Tahta Lessons

**Time:** 2-3 hours

#### AI Block (60-90 min)
```bash
# Read theory (focus on core)
cat week0_setup/theory_intro.md           # 15 min
cat week0_setup/theory_core_concepts.md   # 30 min
cat week0_setup/theory_foundations.md     # 20 min

# Self-assessment
cp week0_setup/theory_closure.md reports/w0_ai_theory_closure.md
# Fill in: Tensors, Loss, Optimizer, LR, Overfit/Underfit
```

**Key Questions (Answer in closure doc):**
1. What is a tensor's shape/dtype/device?
2. MSE vs MAE vs Cross-Entropy — when to use?
3. LR too high → ? LR too low → ?
4. Overfit symptoms? Underfit symptoms?
5. Train/Val/Test split purpose?

#### Crypto Block (45-60 min)
```bash
cd crypto/docs/w0_bootstrap

# Tahta 01: EVM Data Model (1,277 lines, 60-75 min)
cat 01_tahta_evm_giris.md
# Focus: Block → Tx → Receipt → Log → Event
# Quiz: 8 questions

# Tahta 02: JSON-RPC 101 (1,012 lines, 40-50 min)
cat 02_tahta_rpc_101.md
# Focus: blockNumber, getBlock, getLogs
# Production patterns: Rate limiting, error handling
```

#### DoD (Day 2)
- [ ] AI theory closure ≥ 80% complete
- [ ] Tahta 01 quiz solved
- [ ] Tahta 02 mini-exercises done
- [ ] Notes in `daily_log.md`

---

### Day 3: Ingest (Backfill) + Report v0

**Time:** 2-3 hours

#### Crypto Block (90 min)
```bash
cd crypto/w0_bootstrap

# Basic backfill (first 5000 blocks)
python capture_transfers.py --start 5000000 --blocks 5000
# Expected: logs inserted into onchain.duckdb

# Verify
duckdb onchain.duckdb
# > SELECT COUNT(*) FROM transfers;
# > SELECT MAX(block_number), MAX(block_time) FROM transfers;
# > .quit

# Generate report v0 (CLI pretty print)
python report_v0.py --wallet 0xYourWallet --hours 24
# Expected: Inbound/outbound/top counterparties table
```

**Metrics to Log:**
```
- Total logs ingested: XXX
- Time taken: XXX seconds
- Logs/second: XXX
- Latest block: XXX
- Earliest block: XXX
```

#### AI Block (30-45 min)
```bash
# Light reading: LR/MSE intuition
cat week0_setup/theory_mathematical.md | grep -A 20 "Linear Regression"
# Focus on: y = θx + b intuition
```

#### DoD (Day 3)
- [ ] 5000+ blocks ingested
- [ ] DuckDB queries working
- [ ] Report v0 generated
- [ ] Metrics logged

---

### Day 4: Idempotent + State + Schema

**Time:** 2-3 hours

#### Crypto Block (90-120 min)
```bash
cd crypto/w0_bootstrap

# Idempotent ingest (8000 blocks with state tracking)
python capture_transfers_idempotent.py --backfill 8000
# Expected: State tracking, reorg protection

# Re-run same range (idempotency test)
python capture_transfers_idempotent.py --backfill 8000
# Expected: No duplicate logs!

# Verify no duplicates
duckdb onchain.duckdb <<SQL
SELECT tx_hash, log_index, COUNT(*) as cnt
FROM transfers
GROUP BY tx_hash, log_index
HAVING cnt > 1;
-- Should return empty (0 duplicates)
SQL

# JSON report + schema validation
python report_json.py --wallet 0xYourWallet --hours 24 > /tmp/report.json
cat /tmp/report.json | python validate_report.py
# Expected: ✅ report_v1 schema valid

# Makefile shortcuts
make c.capture.idem
make report.schema W=0xYourWallet
```

**Key Files to Inspect:**
```bash
# State table
duckdb onchain.duckdb "SELECT * FROM scan_state;"
# Expected: key='transfers_v1', last_scanned_block=XXXX

# Schema
cat ../../schemas/report_v1.json | jq '.required'
# Expected: wallet, window_hours, inbound, outbound, tx_count, top_counterparties
```

#### AI Block (30-45 min)
```bash
# Week 1 preview
cat week1_tensors/README.md
# Understand: What's coming (Linear regression hands-on)
```

#### DoD (Day 4)
- [ ] Idempotent ingest verified (0 duplicates)
- [ ] State tracking working
- [ ] JSON schema validation ✓
- [ ] Makefile shortcuts tested

---

### Day 5: API Service + CI + Documentation

**Time:** 2-3 hours

#### Crypto Block (60-90 min)
```bash
# Start mini API
make c.api
# Runs: uvicorn crypto.service.app:app --reload
# Terminal will show: Uvicorn running on http://127.0.0.1:8000

# In another terminal, test:
curl http://localhost:8000/healthz
# Expected: {"status":"healthy"}

curl "http://localhost:8000/wallet/0xYourWallet/report?hours=24"
# Expected: JSON with inbound/outbound/tx_count/top_counterparties

# Light load test (optional)
ab -n 20 -c 2 http://localhost:8000/healthz
# Check: All 20 requests succeed, p95 < 100ms
```

#### Docs/CI Block (45-60 min)
```bash
# Run CI checks
make docs.check
# Expected: ✅ Markdown links valid, no broken links

make py.ci
# Expected: ✅ Ruff clean, pytest passes

# Update documentation
# 1. Check README hierarchy
cat README.md | grep "docs/"
# Verify: Links to program_overview.md, crypto_overview.md work

# 2. Update CHANGELOG
echo "## [1.1.0] - $(date +%Y-%m-%d)" >> CHANGELOG.md
echo "- Week 0 complete: Setup + Ingest + Report + API" >> CHANGELOG.md

# 3. Verify COMMANDS.md
make help  # Should match COMMANDS.md content
```

#### Final Reports
```bash
# Crypto summary
cat > reports/w0_crypto_summary.md <<EOF
# Week 0 Crypto Summary

## Metrics
- RPC latency (p95): XXX ms
- Total logs ingested: XXX
- Duplicate logs: 0 ✓
- Schema validation: PASS ✓
- API health: 200 OK ✓

## State
- Last scanned block: XXX
- DB size: XXX MB
- Confirmations: 12

## Notes
- Optimal chunk size: 1500 blocks
- No reorg detected
- STEP parameter stable at 1500

## Next (Week 1)
- Continuous ingest loop
- Price cache integration
- Telegram alerts
EOF

# AI closure
# Complete: reports/w0_ai_theory_closure.md
# Ensure all checkboxes filled
```

#### DoD (Day 5)
- [ ] API `/healthz` working
- [ ] API `/wallet/{addr}/report` working
- [ ] CI green (docs + py)
- [ ] CHANGELOG updated
- [ ] Summary reports created

---

## 📦 Teslimatlar (DoD)

### AI Track Checklist

```
Setup:
  ✓ Python 3.11+ installed
  ✓ Virtual environment created
  ✓ Dependencies installed (dev)
  ✓ MPS available & verified

Quality:
  ✓ pytest -q → all green
  ✓ ruff check . → 0 errors

Theory:
  ✓ theory_closure.md complete
  ✓ Key concepts understood:
    - Tensors (shape/dtype/device)
    - Loss functions (MSE/MAE/CE)
    - Optimizer (SGD/Adam/LR)
    - Overfit/Underfit symptoms
    - Train/Val/Test split

Artifacts:
  ✓ week0_setup/nova-setup.md
  ✓ reports/w0_ai_theory_closure.md
  ✓ week0_setup/hello_tensor.py (working)
```

### Crypto Track Checklist

```
Setup:
  ✓ .env configured (RPC_URL)
  ✓ RPC health check < 300ms
  ✓ DuckDB schema initialized

Ingest:
  ✓ Basic capture working (5000+ blocks)
  ✓ Idempotent ingest verified (0 duplicates)
  ✓ State tracking (scan_state table)
  ✓ Reorg protection (CONFIRMATIONS ≥ 5)

Reports:
  ✓ report_v0.py (CLI pretty)
  ✓ report_json.py (API-ready)
  ✓ Schema validation passing

API:
  ✓ FastAPI service running
  ✓ /healthz endpoint (200 OK)
  ✓ /wallet/{addr}/report working

Artifacts:
  ✓ onchain.duckdb (with data)
  ✓ reports/w0_crypto_summary.md
  ✓ crypto/w0_bootstrap/.env
```

### Docs & CI Checklist

```
Documentation:
  ✓ Tahta 01: EVM Veri Modeli (read)
  ✓ Tahta 02: JSON-RPC 101 (read)
  ✓ Tahta 03: Transfer Anatomisi (bonus)
  ✓ Tahta 04: getLogs + Reorg (bonus)
  ✓ README hierarchy clear
  ✓ CHANGELOG updated

CI/CD:
  ✓ .github/workflows/docs-link-check.yml
  ✓ .github/workflows/python-ci.yml
  ✓ make docs.check → green
  ✓ make py.ci → green

Schema:
  ✓ schemas/report_v1.json defined
  ✓ validate_report.py working

Artifacts:
  ✓ COMMANDS.md (shortcuts documented)
  ✓ CHANGELOG.md (v1.1.0 noted)
```

---

## 🧪 Test & Ölçüm Rehberi

### AI Testing

```bash
# MPS verification
python week0_setup/hello_tensor.py
# Expected output:
# ✓ MPS is available!
# ✓ Created tensor on mps device
# ✓ Performed computation on MPS

# Unit tests
pytest -q
# Expected: X passed in Y.Ys

# Linting
ruff check .
# Expected: All checks passed!

# (Optional) Type checking
mypy week0_setup/ --ignore-missing-imports
```

### Crypto Testing

```bash
# 1. RPC Health
python crypto/w0_bootstrap/rpc_health.py
# Metrics: latency_ms (target < 300), block_number

# 2. Ingest Verification
python crypto/w0_bootstrap/capture_transfers_idempotent.py --backfill 1000

# Check state
duckdb onchain.duckdb <<SQL
SELECT 
  COUNT(*) as total_logs,
  MIN(block_number) as first_block,
  MAX(block_number) as last_block,
  COUNT(DISTINCT token) as unique_tokens
FROM transfers;
SQL

# 3. Duplicate Check (critical!)
duckdb onchain.duckdb <<SQL
SELECT tx_hash, log_index, COUNT(*) as cnt
FROM transfers
GROUP BY tx_hash, log_index
HAVING cnt > 1;
-- Should return 0 rows
SQL

# 4. Schema Validation
python crypto/w0_bootstrap/report_json.py \
  --wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 \
  --hours 24 | \
python crypto/w0_bootstrap/validate_report.py
# Expected: ✅ report_v1 schema valid

# 5. API Tests
curl http://localhost:8000/healthz | jq '.'
# Expected: {"status": "healthy"}

curl "http://localhost:8000/wallet/0xYourWallet/report?hours=24" | jq '.'
# Expected: Valid JSON with all required fields

# 6. Load Test (light)
ab -n 50 -c 5 http://localhost:8000/healthz
# Metrics: 100% success rate, p95 < 200ms
```

### CI/Docs Testing

```bash
# Docs validation
make docs.check
# Runs: lychee (link checker) + markdownlint

# Python CI
make py.ci
# Runs: ruff check + pytest

# Manual link check
find . -name "*.md" -exec grep -H "](docs/" {} \;
# Verify all relative links exist

# Schema validation
cat schemas/report_v1.json | jq '.'
# Ensure valid JSON
```

### Metrics to Log

**Daily Log Format (`reports/daily_log.md`):**
```markdown
# W0D1 (2025-10-06)

## AI
- MPS: ✓ available
- pytest: 15/15 passed
- ruff: 0 errors
- Theory: 40% (intro + core concepts)

## Crypto
- RPC: 145ms (p95)
- Logs ingested: 0 (setup day)
- Duplicates: N/A
- Schema: N/A

## Issues
- None

## Tomorrow
- [ ] Complete theory reading (60%)
- [ ] Start Tahta 01 (EVM model)
- [ ] Backfill 5000 blocks
```

---

## 🧰 Komut Hızlı Kartı

### Setup & Verification
```bash
# Install
pip install -e ".[dev,crypto]"

# Verify AI
python week0_setup/hello_tensor.py
pytest -q
ruff check .

# Verify Crypto
python crypto/w0_bootstrap/rpc_health.py
```

### Daily Workflow
```bash
# AI
pytest -q && ruff check .

# Crypto - Health
make c.health

# Crypto - Ingest (idempotent)
make c.capture.idem

# Crypto - Report
make report.schema W=0xYourWalletAddress

# API
make c.api  # Start server
curl http://localhost:8000/healthz
```

### Quality Checks
```bash
# All CI
make docs.check && make py.ci

# Docs only
make docs.check

# Python only
make py.ci
```

### Debugging
```bash
# DuckDB inspect
duckdb onchain.duckdb
# > .tables
# > SELECT COUNT(*) FROM transfers;
# > SELECT * FROM scan_state;

# RPC detailed check
python crypto/w0_bootstrap/rpc_health.py --verbose

# Log parsing
tail -f logs/ingest.log  # If you set up logging
```

### Shortcuts (Makefile)
```bash
make help              # All available commands

# Crypto aliases
c.health               # = crypto.health
c.capture              # = crypto.capture (basic)
c.capture.idem         # = crypto.capture.idem (production)
c.api                  # = crypto.api (start FastAPI)

# Reports
report.json W=0x...    # JSON output
report.schema W=0x...  # JSON + validation
```

---

## ⚠️ Riskler & Önlemler

### Risk 1: RPC Timeout / Rate Limiting

**Belirtiler:**
- `TimeoutError` or `429 Too Many Requests`
- Logs: "Request took > 5s"

**Önlemler:**
```python
# 1. Pencere boyutunu küçült
CHUNK_SIZE = 1000  # 1500'den düşür

# 2. Backoff ekle
import time
for attempt in range(3):
    try:
        logs = fetch_logs(...)
        break
    except (TimeoutError, RateLimitError):
        time.sleep(2 ** attempt)  # 1s, 2s, 4s

# 3. CONFIRMATIONS artır
CONFIRMATIONS = 12  # 5'ten yükselt (daha az frequent request)
```

---

### Risk 2: Reorg (Blockchain Reorganization)

**Belirtiler:**
- Log'lar DB'de var ama chain'de yok
- Block hash mismatch

**Önlemler:**
```python
# 1. Safe range kullan
safe_latest = latest - CONFIRMATIONS  # Don't scan unsafe tip

# 2. Tail re-scan
start = max(0, last_scanned - CONFIRMATIONS)  # Overlap window

# 3. Reorg detection
stored_hash = db.get_block_hash(block_num)
current_hash = chain.get_block_hash(block_num)
if stored_hash != current_hash:
    # Delete affected logs, re-scan
    pass
```

---

### Risk 3: Duplicate Logs

**Belirtiler:**
```sql
SELECT tx_hash, log_index, COUNT(*)
FROM transfers
GROUP BY tx_hash, log_index
HAVING COUNT(*) > 1;
-- Returns rows (bad!)
```

**Önlemler:**
```sql
-- 1. UNIQUE constraint
CREATE TABLE transfers (
    ...
    UNIQUE(tx_hash, log_index)  -- Prevents duplicates at DB level
);

-- 2. Anti-join pattern (application level)
INSERT INTO transfers
SELECT s.*
FROM staging s
LEFT JOIN transfers t ON t.tx_hash = s.tx_hash AND t.log_index = s.log_index
WHERE t.tx_hash IS NULL;  -- Only insert new
```

---

### Risk 4: Decimals Error

**Belirtiler:**
- Amounts look huge (e.g., 1000000000000000000)
- Or tiny (e.g., 0.000000000000000001)

**Önlem:**
```python
# Always apply decimals!
raw_value = int(log["data"], 16)  # e.g., 1500000000000000000
decimals = 18  # Standard for most tokens

value_unit = raw_value / (10 ** decimals)  # 1.5

# Store both
db.insert({
    "raw_value": raw_value,      # For precision
    "value_unit": value_unit,    # For display
    "decimals": decimals         # For verification
})
```

---

### Risk 5: CI Breaking

**Belirtiler:**
- GitHub Actions red ❌
- `make docs.check` fails
- `make py.ci` fails

**Önlemler:**
```bash
# 1. Fix lint errors
ruff check . --fix

# 2. Fix broken links
# Find: grep -r "](docs/" *.md
# Verify each link exists

# 3. Run locally before push
make docs.check && make py.ci
# Only push if green

# 4. Check CI logs
# GitHub Actions → Click failed job → Read error
```

---

## 🔚 Hafta Sonu Kapanış Checklist

### Pre-Flight (Before marking W0 complete)

```
## AI Track
✓ MPS verified and documented
✓ pytest: all tests green
✓ ruff: 0 errors
✓ Theory closure: 100% checkboxes filled
✓ Week 1 README read (preview)

## Crypto Track
✓ RPC health: < 300ms, documented
✓ Ingest: 5000+ blocks, 0 duplicates
✓ State tracking: scan_state table functional
✓ Report: JSON + schema validation ✓
✓ API: /healthz + /wallet/{addr}/report working

## Docs & Quality
✓ CI: docs.check + py.ci both green
✓ Tahta lessons: At least 01-02 read
✓ CHANGELOG: v1.1.0 noted
✓ COMMANDS.md: up to date

## Reports
✓ reports/w0_ai_theory_closure.md complete
✓ reports/w0_crypto_summary.md with metrics
✓ reports/daily_log.md for all 5 days

## Week 1 Prep
✓ WEEK1_MASTER_PLAN.md reviewed (if exists)
✓ week1_tensors/README.md read
✓ Understand next goals: Linear regression hands-on
```

### Final Commit

```bash
# Verify everything clean
git status

# Add all Week 0 work
git add .

# Commit
git commit -m "Week 0 Complete: Setup + Theory + Ingest + Report + API

AI Track:
- MPS verified ✓
- pytest + ruff green ✓
- Theory closure 100% ✓

Crypto Track:
- RPC health < 300ms ✓
- Idempotent ingest (0 duplicates) ✓
- Report JSON + schema ✓
- Mini API working ✓

Docs:
- CI green ✓
- Tahta 01-04 available
- CHANGELOG + COMMANDS updated

Ready for Week 1: Linear Regression"

# (Optional) Tag
git tag -a v1.1.0-w0 -m "Week 0 Complete"

# Push
git push origin main
git push origin v1.1.0-w0
```

---

## 📝 PR/Issue Şablonları

### GitHub Issue: Week 0 Tracker

```markdown
# Week 0: Setup + Theory + Bootstrap

## Overview
- **Duration:** 5 days (Oct 6-10, 2025)
- **Tracks:** AI + Crypto + Docs/CI
- **Owner:** @onur

## Checklist

### Day 1: Setup
- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] Dependencies: `pip install -e ".[dev,crypto]"`
- [ ] MPS verified (`hello_tensor.py`)
- [ ] RPC health check < 300ms

### Day 2: Theory + Tahta
- [ ] AI theory reading (intro + core + foundations)
- [ ] Theory closure ≥ 80%
- [ ] Tahta 01: EVM Veri Modeli (read + quiz)
- [ ] Tahta 02: JSON-RPC 101 (read + exercises)

### Day 3: Ingest
- [ ] Backfill 5000+ blocks
- [ ] DuckDB queries working
- [ ] Report v0 generated

### Day 4: Idempotent + Schema
- [ ] Idempotent ingest verified (0 duplicates)
- [ ] State tracking functional
- [ ] JSON schema validation ✓
- [ ] Makefile shortcuts tested

### Day 5: API + CI
- [ ] FastAPI `/healthz` working
- [ ] FastAPI `/wallet/{addr}/report` working
- [ ] CI green (docs + py)
- [ ] CHANGELOG updated
- [ ] Summary reports written

### Final DoD
- [ ] All checklists above complete
- [ ] `reports/w0_ai_theory_closure.md`
- [ ] `reports/w0_crypto_summary.md`
- [ ] Week 1 preview read

## Metrics (Fill at end)
- RPC latency (p95): ___ms
- Total logs: ___
- Duplicates: 0 ✓
- Schema: PASS ✓
- API health: 200 OK ✓

## Blockers
(None yet)

## Notes
(Add as you go)
```

---

### Pull Request Template

```markdown
# Week 0 Complete: Setup + Theory + Ingest + Report + API

## Summary
This PR completes Week 0 of the NovaDev v1.1 program, establishing the foundation for both AI and Crypto tracks.

## Changes

### AI Track
- ✅ Python 3.11+ environment with MPS support
- ✅ pytest + ruff integration (CI green)
- ✅ Theory foundation (tensors → LR)
- ✅ Self-assessment complete

**Files:**
- `week0_setup/hello_tensor.py` - MPS validation
- `week0_setup/nova-setup.md` - Setup log
- `reports/w0_ai_theory_closure.md` - Theory checkpoint

### Crypto Track
- ✅ RPC health check (< 300ms)
- ✅ Idempotent ingest pipeline (0 duplicates)
- ✅ State tracking (scan_state table)
- ✅ Wallet report (JSON + schema validation)
- ✅ Mini FastAPI service

**Files:**
- `crypto/w0_bootstrap/` - Complete bootstrap scripts
- `crypto/service/app.py` - FastAPI service
- `schemas/report_v1.json` - API contract
- `onchain.duckdb` - State database

### Docs & CI
- ✅ Tahta lessons (01-04, ~6K lines)
- ✅ CI workflows (docs + python)
- ✅ README hierarchy updated
- ✅ CHANGELOG + COMMANDS

**Files:**
- `crypto/docs/w0_bootstrap/01-04_tahta_*.md`
- `.github/workflows/` - CI/CD
- `CHANGELOG.md`, `COMMANDS.md`

## Testing

```bash
# AI
pytest -q                                    # ✅ 15/15 passed
ruff check .                                 # ✅ 0 errors

# Crypto
make c.health                                # ✅ 142ms
make c.capture.idem                          # ✅ 5234 logs, 0 dup
make report.schema W=0x...                   # ✅ schema valid

# CI
make docs.check && make py.ci                # ✅ All green
```

## Metrics
- **RPC latency:** 142ms (p95)
- **Logs ingested:** 5,234
- **Duplicates:** 0 ✓
- **Schema validation:** PASS ✓
- **API health:** 200 OK ✓

## DoD Verification
- [x] AI: MPS + tests + theory closure
- [x] Crypto: RPC + ingest + report + API
- [x] Docs: CI + Tahta + README
- [x] Reports: closure + summary + daily logs

## Next Steps (Week 1)
- AI: Linear regression implementation
- Crypto: Continuous ingest loop + price cache
- Integration: Combined metrics dashboard

## Screenshots
(Optional: Add API response, DuckDB query results, etc.)

---

**Ready for review!** 🚀
```

---

## 📚 Referanslar

### Repo İçi Dökümanlar

**Genel:**
- `README.md` - Program overview
- `docs/program_overview.md` - ⭐ Complete syllabus
- `docs/crypto_overview.md` - Crypto track details
- `CHANGELOG.md` - Version history
- `COMMANDS.md` - Makefile reference

**AI Track:**
- `week0_setup/README.md` - Setup guide
- `week0_setup/theory_*.md` - Theory lessons (7 docs, 7K+ lines)
- `week1_tensors/README.md` - Week 1 preview

**Crypto Track:**
- `crypto/README.md` - Crypto setup guide
- `crypto/docs/w0_bootstrap/README.md` - Tahta series index
- `crypto/docs/w0_bootstrap/01_tahta_evm_giris.md` - (1,277 lines)
- `crypto/docs/w0_bootstrap/02_tahta_rpc_101.md` - (1,012 lines)
- `crypto/docs/w0_bootstrap/03_tahta_transfer_anatomi.md` - (1,094 lines)
- `crypto/docs/w0_bootstrap/04_tahta_getlogs_pencere_reorg.md` - (2,266 lines)

**Schemas:**
- `schemas/report_v1.json` - Wallet report contract

---

## 🧩 Stretch Goals (Opsiyonel)

Eğer Week 0'ı erken bitirirsen:

### Bonus 1: Postman/HTTPie Collection
```bash
# HTTPie examples
http GET http://localhost:8000/healthz

http GET "http://localhost:8000/wallet/0xYOUR_WALLET/report?hours=24"

# Export collection
# → `crypto/postman_collection.json`
```

### Bonus 2: Price Cache (v1_ext)
```python
# Add USD estimate to report
{
    "wallet": "0x...",
    "inbound": 1.5,
    "inbound_usd": 3975.00,  # ← New!
    ...
}

# Use CoinGecko/CoinMarketCap API (free tier)
```

### Bonus 3: Telegram Notification (Local-Only)
```python
# Send daily summary to personal Telegram
# No secrets in repo! Use env vars

import os
import requests

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_summary(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": text})
```

### Bonus 4: Performance Dashboard
```python
# Streamlit mini dashboard
# streamlit run crypto/dashboard.py

import streamlit as st
import duckdb

conn = duckdb.connect("onchain.duckdb")

# Metrics
total_logs = conn.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
st.metric("Total Logs", f"{total_logs:,}")

# Chart: Logs over time
# ...
```

### Bonus 5: Tahta 03-04 (Already Done! 🎉)
- `03_tahta_transfer_anatomi.md` - Transfer event deep-dive
- `04_tahta_getlogs_pencere_reorg.md` - Production patterns

---

## 📞 Destek & Troubleshooting

### Stuck? Check These First

1. **Environment Issues:**
   ```bash
   python --version  # Must be 3.11+
   which python      # Should be in .venv
   pip list | grep torch  # Verify PyTorch installed
   ```

2. **MPS Not Available:**
   - macOS version < 12.3? MPS requires 12.3+
   - Not Apple Silicon? MPS only on M1/M2/M3
   - Fallback: PyTorch will use CPU (slower but works)

3. **RPC Errors:**
   - Check `.env`: Is `RPC_URL` valid?
   - Test in browser: Paste `RPC_URL` → should show "Method not allowed"
   - Rate limit? Wait 1 min, try again
   - Check provider dashboard for quota

4. **DuckDB Issues:**
   - Permission denied? `chmod +w onchain.duckdb`
   - Corrupted? Delete and re-ingest
   - Large size? Normal (1GB+ after 50K blocks)

5. **API Not Starting:**
   - Port 8000 in use? Kill other process: `lsof -ti:8000 | xargs kill`
   - Import errors? `pip install -e ".[crypto]"`
   - Check logs: `uvicorn ... --log-level debug`

### Getting Help

- **Docs:** Search `docs/` and `crypto/docs/` first
- **Code Examples:** Check `crypto/w0_bootstrap/*.py`
- **Tahta Lessons:** Detailed explanations in 01-04
- **This Plan:** Re-read relevant section

---

## 🏁 Success Criteria

You've **crushed Week 0** if:

```
✅ AI: MPS working + tests green + theory solid
✅ Crypto: RPC healthy + ingest idempotent + report valid + API live
✅ Docs: CI green + Tahta lessons read + structure clear
✅ Artifacts: All reports written, metrics logged
✅ Mindset: Confident to start Week 1 (Linear regression)
```

**Week 0 is about building a solid foundation, not speed.**

Take your time, understand the concepts, and ensure everything works before moving to Week 1.

---

## 🚀 Week 1 Preview

### AI: Linear Regression (Week 1)
- Implement LR from scratch (manual)
- PyTorch `nn.Linear` + optimizer
- Train/val split, loss curve plotting
- Hyperparameter tuning (LR, L2)

### Crypto: Continuous Ingest (Week 1)
- Loop: scan → wait → scan (cron/systemd)
- Price cache integration
- Basic alerting (threshold-based)

### Integration:
- Shared `outputs/` for metrics
- Combined dashboard (Streamlit/Plotly)

---

**Week 0 Master Plan — v1.1.0**  
*Prepared: 2025-10-06*  
*Owner: NovaDev Protocol Team*  
*Status: Production-Ready*

