# NovaDev — Command Reference (Cheatsheet)

Quick reference for all common commands. Use `make help` for full list.

---

## 🚀 Quick Start

```bash
# Clone & setup
git clone <repo-url>
cd novadev-protocol
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,crypto]"

# Verify
python week0_setup/hello_tensor.py  # MPS test
pytest -q                            # Tests
ruff check .                         # Lint
```

---

## 🤖 AI Commands

### Week 0: Setup
```bash
# Theory check
cat week0_setup/theory_closure.md

# MPS test
python week0_setup/hello_tensor.py
```

### Week 1: Linear Regression
```bash
# Data generation
python week1_tensors/data_synth.py

# Manual GD
python week1_tensors/linreg_manual.py

# nn.Module
python week1_tensors/linreg_module.py

# Full training
make ai.week1
# or: python week1_tensors/train.py
```

### Testing & Linting
```bash
make ai.test     # pytest
make ai.lint     # ruff check
make py.ci       # lint + test
```

---

## 🪙 Crypto Commands

### Week 0: Bootstrap

```bash
# Setup .env
cd crypto/w0_bootstrap
cp .env.example .env
# Edit .env: RPC_URL=https://...

# Quick checks (Makefile shortcuts)
make c.health              # RPC health
make c.capture.idem        # Idempotent capture (production)
make c.report W=0x...      # CLI pretty report
make c.report.json W=0x... # JSON report
make c.api                 # FastAPI service

# Full commands
python crypto/w0_bootstrap/rpc_health.py
python crypto/w0_bootstrap/capture_transfers_idempotent.py --backfill 5000
python crypto/w0_bootstrap/report_v0.py --wallet 0x... --hours 24
python crypto/w0_bootstrap/report_json.py --wallet 0x... --hours 24

# JSON schema validation
make report.schema W=0x...
# or: python crypto/w0_bootstrap/report_json.py --wallet 0x... | \
#     python crypto/w0_bootstrap/validate_report.py
```

### FastAPI Service

```bash
# Start server
make c.api
# or: uvicorn crypto.service.app:app --reload

# Test endpoints
curl http://localhost:8000/healthz
curl "http://localhost:8000/wallet/0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045/report?hours=24"

# OpenAPI docs
open http://localhost:8000/docs       # Swagger UI
open http://localhost:8000/redoc      # ReDoc
```

### Week 1: Collector Loop (Coming)

```bash
# Start collector (foreground)
python crypto/w1_ingest/collector_loop.py

# Start collector (background)
nohup python crypto/w1_ingest/collector_loop.py > collector.log 2>&1 &

# Stop collector
pkill -f collector_loop.py

# Check logs
tail -f collector.log
```

---

## 📚 Documentation

```bash
# Main docs
cat docs/program_overview.md    # ⭐⭐⭐ TAM SYLLABUS
cat docs/overview.md             # AI details
cat docs/crypto_overview.md      # Crypto details

# Week guides
cat week0_setup/README.md
cat crypto/README.md
cat crypto/w0_bootstrap/README.md

# Meta
cat CHANGELOG.md                 # Version history
cat PR_TEMPLATE.md               # PR guidelines
```

---

## 🔍 Quality & CI

```bash
# Docs validation
make docs.check
# or: lychee --no-progress --exclude-mail "**/*.md"

# Python CI (lint + test)
make py.ci
# or: ruff check . && pytest -q

# Schema validation
make report.schema W=0x...
```

---

## 📊 Development Workflow

### Daily Routine

```bash
# Morning: Pull latest
git pull origin master

# Work: AI + Crypto
# ... (code, test, iterate)

# Evening: Commit
git add -A
git status
git commit -m "W1D2: AI MSE=X.XX, Crypto /report p95=XXms"
git push origin master
```

### Weekly Routine

```bash
# Monday: Plan
# Read week README
cat weekX_*/README.md
cat crypto/wX_*/README.md

# Tuesday-Thursday: Execute
# (Daily commits)

# Friday: Report
# Generate outputs
python weekX_*/train.py
make c.api

# Write report
vim reports/wX_summary.md

# Tag
git tag wX-complete
git push origin master --tags

# Update CHANGELOG
vim CHANGELOG.md
git commit -m "Week X complete"
```

---

## 🧪 Testing

### Unit Tests

```bash
# All tests
pytest

# Specific week
pytest tests/test_week1.py -v

# With coverage
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### Integration Tests (Crypto)

```bash
# Full pipeline test
cd crypto/w0_bootstrap

# 1. Health
python rpc_health.py

# 2. Capture (small)
python capture_transfers_idempotent.py --backfill 100 --max_batches 3

# 3. Report
python report_json.py --wallet 0x... | python validate_report.py

# 4. Service
uvicorn crypto.service.app:app &
sleep 2
curl http://localhost:8000/healthz
curl http://localhost:8000/wallet/0x.../report
pkill uvicorn
```

---

## 🛠️ Maintenance

### Dependencies

```bash
# Update dependencies
pip install --upgrade pip
pip install -e ".[dev,crypto]"

# Check outdated
pip list --outdated

# Freeze (for reference)
pip freeze > requirements-frozen.txt
```

### Database

```bash
# Check DuckDB
cd crypto/w0_bootstrap
du -h onchain.duckdb

# Query manually
python - <<'PY'
import duckdb
con = duckdb.connect("onchain.duckdb")
print(con.execute("SELECT COUNT(*) FROM transfers").fetchone())
print(con.execute("SELECT * FROM scan_state").fetchall())
PY
```

### Cleanup

```bash
# Remove cached data
rm -rf crypto/**/cache/
rm -rf crypto/**/*.duckdb

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove outputs (be careful!)
rm -rf outputs/*.png
rm -rf outputs/*.csv
```

---

## 🔧 Troubleshooting

### RPC Issues

```bash
# Test RPC directly
curl -X POST https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

# Check .env
cat crypto/w0_bootstrap/.env
```

### MPS Issues (Apple Silicon)

```bash
# Test MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, check PyTorch version
pip show torch

# Reinstall if needed
pip install --upgrade torch torchvision
```

### Import Errors

```bash
# Verify install
pip show novadev-protocol

# Reinstall in editable mode
pip install -e ".[dev,crypto]"

# Check PYTHONPATH
echo $PYTHONPATH
```

---

## 🎯 Makefile Targets (Full List)

```bash
# Show all targets
make help

# AI
make ai.test
make ai.lint
make ai.week1

# Crypto (full names)
make crypto.health
make crypto.capture
make crypto.capture.idem
make crypto.report W=0x...
make crypto.report.json W=0x...
make crypto.api

# Crypto (shortcuts)
make c.health
make c.capture
make c.capture.idem
make c.report W=0x...
make c.report.json W=0x...
make c.api

# Quality
make docs.check
make py.ci
make report.schema W=0x...
```

---

## 📦 Release Commands

### Tagging

```bash
# Create tag
git tag -a v1.x.0 -m "Release v1.x - Description"

# Push tag
git push origin v1.x.0

# List tags
git tag -l

# Delete tag (if mistake)
git tag -d v1.x.0
git push origin :refs/tags/v1.x.0
```

### GitHub Release

```bash
# 1. Create tag (above)
# 2. Go to GitHub → Releases → Draft New Release
# 3. Select tag: v1.x.0
# 4. Copy from: .github/RELEASE_NOTES_v1.x.md
# 5. Publish
```

---

## 🔗 Useful Links

- **Main Docs:** `docs/program_overview.md`
- **AI Track:** `docs/overview.md`
- **Crypto Track:** `docs/crypto_overview.md`
- **Week 0:** `week0_setup/README.md`, `crypto/w0_bootstrap/README.md`
- **Changelog:** `CHANGELOG.md`

---

## 💡 Tips

### Productivity

```bash
# Alias in .bashrc / .zshrc
alias nv='cd ~/code/novadev-protocol && source .venv/bin/activate'
alias nvai='nv && make ai.week1'
alias nvcrypto='nv && make c.api'
alias nvtest='nv && make py.ci'
```

### Git

```bash
# Quick commit
git add -A && git commit -m "W1D2: progress" && git push

# Amend last commit
git add -A && git commit --amend --no-edit

# View history
git log --oneline --graph --all -10
```

### Jupyter (Optional)

```bash
# Install
pip install jupyter

# Start
jupyter notebook

# Open week notebook
# (if you create .ipynb files)
```

---

**NovaDev Command Reference — v1.1**

*Quick access to all tools and workflows*
