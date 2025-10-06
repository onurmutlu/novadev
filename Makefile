# NovaDev Makefile
# Quick shortcuts for common commands

.PHONY: help

help:
	@echo "NovaDev Makefile"
	@echo ""
	@echo "AI Commands:"
	@echo "  make ai.test           - Run pytest"
	@echo "  make ai.lint           - Run ruff check"
	@echo "  make ai.week1          - Run Week 1 training"
	@echo ""
	@echo "Crypto Commands:"
	@echo "  make crypto.health     - RPC health check"
	@echo "  make crypto.capture    - Capture transfers (5000 blocks)"
	@echo "  make crypto.capture.idem - Idempotent capture"
	@echo "  make crypto.report     - Wallet report (W=0x...)"
	@echo "  make crypto.report.json - JSON wallet report (W=0x...)"
	@echo "  make crypto.api        - Start FastAPI service"
	@echo ""
	@echo "General:"
	@echo "  make install           - Install dependencies"
	@echo "  make install.crypto    - Install crypto dependencies"

# ===================================
# General
# ===================================

install:
	pip install -e ".[dev]"

install.crypto:
	pip install -e ".[crypto]"

# ===================================
# AI Commands
# ===================================

ai.test:
	pytest -q

ai.lint:
	ruff check .

ai.week1:
	python week1_tensors/train.py

# ===================================
# Crypto Commands
# ===================================

crypto.health:
	python crypto/w0_bootstrap/rpc_health.py

crypto.capture:
	python crypto/w0_bootstrap/capture_transfers.py --blocks 5000

crypto.capture.idem:
	python crypto/w0_bootstrap/capture_transfers_idempotent.py --backfill 8000 --max_batches 20

crypto.report:
	@test -n "$(W)" || (echo "Usage: make crypto.report W=0x..." && exit 1)
	python crypto/w0_bootstrap/report_v0.py --wallet $(W) --hours 24

crypto.report.json:
	@test -n "$(W)" || (echo "Usage: make crypto.report.json W=0x..." && exit 1)
	python crypto/w0_bootstrap/report_json.py --wallet $(W) --hours 24 | jq

crypto.api:
	uvicorn crypto.service.app:app --reload --host 0.0.0.0 --port 8000

# ===================================
# Shortcuts (aliases)
# ===================================

c.health: crypto.health
c.capture: crypto.capture
c.capture.idem: crypto.capture.idem
c.report: crypto.report
c.report.json: crypto.report.json
c.api: crypto.api

# ===================================
# Quality & CI
# ===================================

docs.check:
	@echo "Checking markdown links..."
	@which lychee > /dev/null || (echo "Install: brew install lychee" && exit 1)
	lychee --no-progress --exclude-mail "**/*.md" || true

py.ci:
	@echo "Running Python CI..."
	ruff check . && pytest -q || true

report.schema:
	@test -n "$(W)" || (echo "Usage: make report.schema W=0x..." && exit 1)
	@echo "Validating report JSON schema..."
	python crypto/w0_bootstrap/report_json.py --wallet $(W) --hours 24 | \
	python crypto/w0_bootstrap/validate_report.py
