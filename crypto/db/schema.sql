-- ===============================================
-- NovaDev Crypto Database Schema (DuckDB)
-- ===============================================
-- Version: 1.0 (Week 0)
-- Purpose: On-chain event storage & analytics
-- ===============================================

-- -----------------------------------------------
-- 1. TRANSFERS (ERC20 & Native)
-- -----------------------------------------------
-- Transfer events from all tokens (including ETH)
CREATE TABLE IF NOT EXISTS transfers (
    id BIGINT PRIMARY KEY,
    
    -- Block info
    block_number BIGINT NOT NULL,
    block_timestamp TIMESTAMP NOT NULL,
    
    -- Transaction
    tx_hash VARCHAR NOT NULL,
    tx_index INTEGER NOT NULL,
    log_index INTEGER NOT NULL,
    
    -- Transfer details
    from_addr VARCHAR(42) NOT NULL,
    to_addr VARCHAR(42) NOT NULL,
    value DECIMAL(78, 0) NOT NULL,  -- Wei/smallest unit (supports up to 2^256)
    
    -- Token info
    token_addr VARCHAR(42) NOT NULL,  -- 0x000...000 for native ETH
    token_symbol VARCHAR(20),
    token_decimals INTEGER,
    
    -- Computed fields (Week 1+)
    value_usd DECIMAL(18, 2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_transfers_from ON transfers(from_addr, block_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transfers_to ON transfers(to_addr, block_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transfers_token ON transfers(token_addr, block_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transfers_block ON transfers(block_number);
CREATE INDEX IF NOT EXISTS idx_transfers_tx ON transfers(tx_hash);

-- -----------------------------------------------
-- 2. SWAPS (DEX Events)
-- -----------------------------------------------
-- Swap events from Uniswap, Curve, etc.
CREATE TABLE IF NOT EXISTS swaps (
    id BIGINT PRIMARY KEY,
    
    -- Block info
    block_number BIGINT NOT NULL,
    block_timestamp TIMESTAMP NOT NULL,
    
    -- Transaction
    tx_hash VARCHAR NOT NULL,
    log_index INTEGER NOT NULL,
    
    -- DEX info
    dex_name VARCHAR(50),  -- 'Uniswap V2', 'Uniswap V3', 'Curve', etc.
    pool_addr VARCHAR(42) NOT NULL,
    
    -- Swap details
    sender VARCHAR(42) NOT NULL,
    recipient VARCHAR(42),
    
    -- Token in
    token_in VARCHAR(42) NOT NULL,
    amount_in DECIMAL(78, 0) NOT NULL,
    
    -- Token out
    token_out VARCHAR(42) NOT NULL,
    amount_out DECIMAL(78, 0) NOT NULL,
    
    -- Computed (Week 1+)
    price_impact DECIMAL(10, 4),  -- %
    amount_in_usd DECIMAL(18, 2),
    amount_out_usd DECIMAL(18, 2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_swaps_sender ON swaps(sender, block_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_swaps_pool ON swaps(pool_addr, block_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_swaps_token_in ON swaps(token_in);
CREATE INDEX IF NOT EXISTS idx_swaps_token_out ON swaps(token_out);

-- -----------------------------------------------
-- 3. BALANCES (Snapshot)
-- -----------------------------------------------
-- Wallet balances at specific timestamps
CREATE TABLE IF NOT EXISTS balances (
    id BIGINT PRIMARY KEY,
    
    -- Wallet
    wallet_addr VARCHAR(42) NOT NULL,
    
    -- Token
    token_addr VARCHAR(42) NOT NULL,
    token_symbol VARCHAR(20),
    
    -- Balance
    balance DECIMAL(78, 0) NOT NULL,  -- Wei/smallest unit
    balance_usd DECIMAL(18, 2),
    
    -- Snapshot time
    snapshot_timestamp TIMESTAMP NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_balances_wallet ON balances(wallet_addr, snapshot_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_balances_token ON balances(token_addr);
CREATE UNIQUE INDEX IF NOT EXISTS idx_balances_unique 
    ON balances(wallet_addr, token_addr, snapshot_timestamp);

-- -----------------------------------------------
-- 4. PRICES (Off-chain)
-- -----------------------------------------------
-- Token prices from CoinGecko, Binance, etc.
CREATE TABLE IF NOT EXISTS prices (
    id BIGINT PRIMARY KEY,
    
    -- Token
    token_addr VARCHAR(42) NOT NULL,
    token_symbol VARCHAR(20) NOT NULL,
    
    -- Price
    price_usd DECIMAL(18, 8) NOT NULL,
    price_source VARCHAR(50),  -- 'coingecko', 'binance', etc.
    
    -- Market data (optional)
    market_cap_usd DECIMAL(20, 2),
    volume_24h_usd DECIMAL(20, 2),
    
    -- Timestamp
    price_timestamp TIMESTAMP NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prices_token ON prices(token_addr, price_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(token_symbol, price_timestamp DESC);

-- -----------------------------------------------
-- 5. ALERTS (Week 2+)
-- -----------------------------------------------
-- Generated alerts
CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT PRIMARY KEY,
    
    -- Alert type
    alert_type VARCHAR(50) NOT NULL,  -- 'volume_spike', 'large_transfer', etc.
    severity VARCHAR(20) NOT NULL,    -- 'low', 'medium', 'high', 'critical'
    
    -- Target
    wallet_addr VARCHAR(42),
    token_addr VARCHAR(42),
    
    -- Context
    title VARCHAR(200) NOT NULL,
    description TEXT,
    
    -- Data
    metadata JSON,  -- Alert-specific data
    
    -- Status
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'sent', 'dismissed'
    sent_at TIMESTAMP,
    
    -- Deduplication
    dedup_key VARCHAR(100),  -- Hash for dedup
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_wallet ON alerts(wallet_addr, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_dedup ON alerts(dedup_key);

-- -----------------------------------------------
-- 6. EVENT_CLASSIFICATIONS (Week 3+)
-- -----------------------------------------------
-- ML-classified events
CREATE TABLE IF NOT EXISTS event_classifications (
    id BIGINT PRIMARY KEY,
    
    -- Event reference
    event_type VARCHAR(20) NOT NULL,  -- 'transfer', 'swap', etc.
    event_id BIGINT NOT NULL,  -- Reference to transfers/swaps table
    tx_hash VARCHAR NOT NULL,
    
    -- Classification
    predicted_class VARCHAR(50) NOT NULL,  -- 'Swap', 'Mint', 'Bridge', etc.
    confidence DECIMAL(5, 4),  -- 0.0 - 1.0
    
    -- Summary (NLP-generated)
    summary_tr TEXT,  -- Türkçe özet
    summary_en TEXT,
    
    -- Metadata
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_classifications_event ON event_classifications(event_type, event_id);
CREATE INDEX IF NOT EXISTS idx_classifications_tx ON event_classifications(tx_hash);

-- -----------------------------------------------
-- VIEWS (Analytics)
-- -----------------------------------------------

-- Daily transfer volume per wallet
CREATE OR REPLACE VIEW v_daily_transfer_volume AS
SELECT
    DATE_TRUNC('day', block_timestamp) AS day,
    from_addr AS wallet,
    token_addr,
    COUNT(*) AS transfer_count,
    SUM(value_usd) AS total_volume_usd
FROM transfers
WHERE value_usd IS NOT NULL
GROUP BY day, from_addr, token_addr;

-- Top wallets by volume (last 7 days)
CREATE OR REPLACE VIEW v_top_wallets_7d AS
SELECT
    wallet,
    SUM(total_volume_usd) AS volume_7d_usd,
    SUM(transfer_count) AS transfers_7d
FROM v_daily_transfer_volume
WHERE day >= CURRENT_DATE - INTERVAL 7 DAYS
GROUP BY wallet
ORDER BY volume_7d_usd DESC;

-- Token price latest
CREATE OR REPLACE VIEW v_latest_prices AS
SELECT DISTINCT ON (token_addr)
    token_addr,
    token_symbol,
    price_usd,
    price_timestamp
FROM prices
ORDER BY token_addr, price_timestamp DESC;

-- -----------------------------------------------
-- SEQUENCES (Auto-increment IDs)
-- -----------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_transfers START 1;
CREATE SEQUENCE IF NOT EXISTS seq_swaps START 1;
CREATE SEQUENCE IF NOT EXISTS seq_balances START 1;
CREATE SEQUENCE IF NOT EXISTS seq_prices START 1;
CREATE SEQUENCE IF NOT EXISTS seq_alerts START 1;
CREATE SEQUENCE IF NOT EXISTS seq_classifications START 1;

-- -----------------------------------------------
-- SEED DATA (Test - Week 0)
-- -----------------------------------------------
-- Örnek fiyat (ETH)
INSERT INTO prices (
    id, token_addr, token_symbol, price_usd, price_source, price_timestamp
) VALUES (
    NEXTVAL('seq_prices'),
    '0x0000000000000000000000000000000000000000',
    'ETH',
    2450.50,
    'test',
    CURRENT_TIMESTAMP
) ON CONFLICT DO NOTHING;

-- Örnek fiyat (USDT)
INSERT INTO prices (
    id, token_addr, token_symbol, price_usd, price_source, price_timestamp
) VALUES (
    NEXTVAL('seq_prices'),
    '0xdac17f958d2ee523a2206206994597c13d831ec7',
    'USDT',
    1.0,
    'test',
    CURRENT_TIMESTAMP
) ON CONFLICT DO NOTHING;

-- ===============================================
-- Schema Complete! ✓
-- ===============================================
-- Week 0: transfers, swaps, balances, prices
-- Week 2+: alerts
-- Week 3+: event_classifications
-- ===============================================
