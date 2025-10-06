"""
Shared pytest fixtures for all tests

This file provides common fixtures used across unit, integration, contract, and API tests.
"""

import json
import tempfile
import pathlib
import duckdb
import pytest
from datetime import datetime, timezone
from typing import Generator


@pytest.fixture(scope="session")
def schema_v1():
    """Load report_v1 JSON schema"""
    schema_path = pathlib.Path("schemas/report_v1.json")
    if not schema_path.exists():
        pytest.skip(f"Schema file not found: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


@pytest.fixture
def tmp_db_path(tmp_path) -> pathlib.Path:
    """Create temporary database path"""
    return tmp_path / "test.duckdb"


@pytest.fixture
def tmp_db_empty(tmp_db_path) -> Generator[pathlib.Path, None, None]:
    """
    Create empty DuckDB with transfers schema
    
    Yields:
        Path to temporary database file
    """
    conn = duckdb.connect(str(tmp_db_path))
    
    # Create transfers table (minimal schema from Tahta 05)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transfers (
            tx_hash TEXT NOT NULL,
            log_index INTEGER NOT NULL,
            block_number BIGINT NOT NULL,
            block_time TIMESTAMP NOT NULL,
            token TEXT NOT NULL,
            symbol TEXT NOT NULL,
            decimals INTEGER NOT NULL,
            from_addr TEXT NOT NULL,
            to_addr TEXT NOT NULL,
            value_unit DOUBLE NOT NULL,
            PRIMARY KEY (tx_hash, log_index)
        )
    """)
    
    conn.close()
    
    yield tmp_db_path
    
    # Cleanup
    if tmp_db_path.exists():
        tmp_db_path.unlink()


@pytest.fixture
def seeded_db(tmp_db_path) -> Generator[pathlib.Path, None, None]:
    """
    Create DuckDB with sample transfer data
    
    Sample data:
    - 0xtest wallet (target)
    - 2 transactions (1 inbound, 1 outbound)
    - USDC token
    
    Yields:
        Path to temporary database file
    """
    conn = duckdb.connect(str(tmp_db_path))
    
    # Create schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transfers (
            tx_hash TEXT NOT NULL,
            log_index INTEGER NOT NULL,
            block_number BIGINT NOT NULL,
            block_time TIMESTAMP NOT NULL,
            token TEXT NOT NULL,
            symbol TEXT NOT NULL,
            decimals INTEGER NOT NULL,
            from_addr TEXT NOT NULL,
            to_addr TEXT NOT NULL,
            value_unit DOUBLE NOT NULL,
            PRIMARY KEY (tx_hash, log_index)
        )
    """)
    
    # Seed sample data
    now = datetime.now(timezone.utc)
    test_wallet = "0xtest"
    
    conn.execute("""
        INSERT INTO transfers VALUES
            ('0xhash1', 0, 100, ?, '0xusdc', 'USDC', 6, '0xa', ?, 100.0),
            ('0xhash2', 0, 101, ?, '0xusdc', 'USDC', 6, ?, '0xb', 50.0)
    """, [now, test_wallet, now, test_wallet])
    
    conn.close()
    
    yield tmp_db_path
    
    # Cleanup
    if tmp_db_path.exists():
        tmp_db_path.unlink()


@pytest.fixture
def tmp_db_with_data(seeded_db):
    """Alias for seeded_db (for backward compatibility)"""
    return seeded_db


@pytest.fixture
def tmp_db_with_missing_metadata(tmp_db_path) -> Generator[pathlib.Path, None, None]:
    """
    Create DuckDB with transfers that have missing/null metadata
    
    Tests fallback behavior for missing token symbol/decimals
    """
    conn = duckdb.connect(str(tmp_db_path))
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transfers (
            tx_hash TEXT NOT NULL,
            log_index INTEGER NOT NULL,
            block_number BIGINT NOT NULL,
            block_time TIMESTAMP NOT NULL,
            token TEXT NOT NULL,
            symbol TEXT,  -- Can be NULL
            decimals INTEGER,  -- Can be NULL
            from_addr TEXT NOT NULL,
            to_addr TEXT NOT NULL,
            value_unit DOUBLE NOT NULL,
            PRIMARY KEY (tx_hash, log_index)
        )
    """)
    
    # Insert transfer with missing metadata
    now = datetime.now(timezone.utc)
    conn.execute("""
        INSERT INTO transfers VALUES
            ('0xhash_missing', 0, 200, ?, '0xtoken_unknown', NULL, NULL, '0xa', '0xtest', 10.0)
    """, [now])
    
    conn.close()
    
    yield tmp_db_path
    
    # Cleanup
    if tmp_db_path.exists():
        tmp_db_path.unlink()


# API test fixtures will be added when we test FastAPI service
# These fixtures depend on the service being properly configured

# Example:
# @pytest.fixture
# def client(seeded_db, monkeypatch):
#     """Create FastAPI TestClient with seeded database"""
#     monkeypatch.setenv("NOVA_DB_PATH", str(seeded_db))
#     from crypto.service.app import app
#     from fastapi.testclient import TestClient
#     return TestClient(app)

