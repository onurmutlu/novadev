# Changelog

All notable changes to NovaDev Protocol will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-10-06

### Added

#### Program & Documentation
- **AI + Crypto Paralel Program** (`docs/program_overview.md`) - Tam syllabus
  - T→P→X ritmi (Teori/Pratik/Ürün) detaylı açıklama
  - 8 haftalık plan (AI + Crypto paralel)
  - Her hafta DoD/KPI net tanımlar
  - Güvenlik ilkeleri (read-only, no custody)
  - Haftalık rapor disiplini & şablonlar
  
#### Crypto Infrastructure
- **W0 Bootstrap** (`crypto/w0_bootstrap/`)
  - `rpc_health.py` - RPC connection check
  - `capture_transfers.py` - Basic event capture
  - `capture_transfers_idempotent.py` - Production-ready idempotent ingest
    - State tracking (resume from last block)
    - Reorg protection (CONFIRMATIONS buffer)
    - Anti-join pattern (no duplicates)
  - `report_v0.py` - CLI wallet report (pretty print)
  - `report_json.py` - JSON wallet report (API-ready)
  - `validate_report.py` - JSON schema validator

- **FastAPI Service** (`crypto/service/app.py`)
  - `GET /healthz` - Health check
  - `GET /wallet/{addr}/report?hours=24` - Wallet report endpoint
  - OpenAPI/Swagger documentation (`/docs`)
  - Pydantic models with validation

- **JSON Schema** (`schemas/report_v1.json`)
  - WalletReportV1 contract definition
  - Validation rules for API responses

#### Developer Tools
- **Makefile** - Command shortcuts
  - AI commands: `ai.test`, `ai.lint`, `ai.week1`
  - Crypto commands: `crypto.health`, `crypto.capture.idem`, `crypto.api`
  - Shortcuts: `c.health`, `c.capture.idem`, `c.api`
  
- **GitHub Actions CI**
  - `docs-link-check.yml` - Markdown link validation & linting
  - `python-ci.yml` - Ruff lint + pytest + smoke tests

- **Markdownlint** (`.markdownlint.json`)
  - Consistent markdown formatting rules

### Changed

#### Documentation Updates
- **README.md** - AI + Crypto paralel vurgu
  - v1.1 başlık: "Öğrenirken İki Gemi Yap"
  - İki sistem çıktı vurgusu (AI + Crypto)
  - Roadmap tablosu: AI + Crypto paralel kolonlar
  - Yasal uyarı eklendi (Crypto için)
  - `program_overview.md`'ye ⭐⭐⭐ referans

- **crypto/README.md** - Dökümantasyon hiyerarşisi
  - 4 seviye dokümantasyon yapısı
  - 8 haftalık özet tablo
  - Tech stack detayları
  - Troubleshooting & SSS genişletildi

- **pyproject.toml** - Crypto dependencies
  - `[crypto]` extras: web3, duckdb, python-dotenv, jsonschema
  - Service dependencies: fastapi, uvicorn

- **.gitignore** - Crypto-specific rules
  - `crypto/**/.env` (secrets)
  - `crypto/**/*.duckdb` (databases)
  - `crypto/**/cache/` (temporary data)

### Notes

#### Security & Compliance
- **Varsayılan mod: READ-ONLY**
  - Private key yok
  - Custody yok
  - Auto-execute yok
- **Testnet-first** (Sepolia)
- **Paper trading / simulation only**
- **Yasal Uyarı**: Bu sistem bilgilendirme amaçlıdır, finansal tavsiye değildir. DYOR.

#### Week 0 Status
- ✅ **AI**: 7061 satır teori, self-assessment, MPS test
- ✅ **Crypto**: RPC health, idempotent ingest, JSON API, FastAPI service
- ✅ **DoD**: W0 complete, W1 ready

---

## [1.0.0] - 2025-10-05

### Added
- Initial project structure
- Week 0-8 skeleton directories
- Basic AI theory notes (7061 lines)
- Common utilities
- Testing infrastructure (pytest, ruff)

---

**Legend:**
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Now removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

---

**NovaDev Protocol** — AI + Crypto Paralel Program
*Versiyon: 1.1.0*
