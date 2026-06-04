-- RetailEdge Ledger Schema
-- Urutan CREATE TABLE ini tidak boleh dibalik.
-- Foreign key dependencies mengikuti urutan di bawah.

-- 1. Decision Bus queue
CREATE TABLE IF NOT EXISTS system_actions (
    action_id TEXT PRIMARY KEY,
    action_type TEXT NOT NULL,
    pair TEXT,
    trade_id TEXT,
    model_id TEXT,
    reason TEXT,
    severity TEXT,
    status TEXT NOT NULL DEFAULT 'PENDING',
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_ts TEXT,
    created_ts TEXT NOT NULL,
    updated_ts TEXT
);

-- 2. All orders placed by Freqtrade
CREATE TABLE IF NOT EXISTS execution_orders (
    order_id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL,
    pair TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    price REAL,
    amount REAL,
    status TEXT NOT NULL,
    created_ts TEXT NOT NULL,
    updated_ts TEXT
);

-- 3. Fill events — deduplicated by order_id + fill_id
CREATE TABLE IF NOT EXISTS execution_fills (
    fill_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    trade_id TEXT NOT NULL,
    pair TEXT NOT NULL,
    side TEXT NOT NULL,
    fill_price REAL NOT NULL,
    fill_amount REAL NOT NULL,
    fee_quote REAL,
    created_ts TEXT NOT NULL,
    PRIMARY KEY (order_id, fill_id)
);

-- 4. Reserved funds per open order
CREATE TABLE IF NOT EXISTS reserved_funds (
    order_id TEXT PRIMARY KEY,
    pair TEXT NOT NULL,
    reserved_quote REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_ts TEXT NOT NULL,
    updated_ts TEXT
);

-- 5. Startup reserved funds reconciliation audit log
CREATE TABLE IF NOT EXISTS reserved_funds_reconciliation (
    run_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    pair TEXT,
    exchange_reserved_quote REAL,
    local_reserved_quote REAL,
    diff_quote REAL,
    status TEXT NOT NULL,
    created_ts TEXT NOT NULL,
    PRIMARY KEY (run_id, order_id)
);

-- 6. Per-trade safety flags
CREATE TABLE IF NOT EXISTS trade_state_flags (
    trade_id TEXT PRIMARY KEY,
    pair TEXT NOT NULL,
    stop_confirmed INTEGER NOT NULL DEFAULT 0,
    stop_unconfirmed_count INTEGER NOT NULL DEFAULT 0,
    emergency_exit_attempted INTEGER NOT NULL DEFAULT 0,
    reserved_mismatch INTEGER NOT NULL DEFAULT 0,
    created_ts TEXT NOT NULL,
    updated_ts TEXT
);

-- 7. Strategy candidate evidence store
CREATE TABLE IF NOT EXISTS strategy_memory (
    candidate_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_hash TEXT,
    feature_hash TEXT,
    oos_trades INTEGER,
    oos_expectancy REAL,
    oos_sharpe REAL,
    pbo REAL,
    dsr REAL,
    cost_floor_pct REAL,
    stage_gate TEXT NOT NULL DEFAULT 'CANDIDATE',
    approved_by TEXT,
    approved_at TEXT,
    created_ts TEXT NOT NULL,
    updated_ts TEXT
);

-- 8. Champion promotion history
CREATE TABLE IF NOT EXISTS deployment_audit (
    audit_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    promoted_from TEXT,
    rollback_model_id TEXT,
    approved_by TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    promotion_source TEXT NOT NULL,
    config_compile_pass INTEGER NOT NULL DEFAULT 0,
    dry_replay_pass INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    created_ts TEXT NOT NULL
);

-- Indexes untuk query yang sering dipakai Guardian dan Reconciler
CREATE INDEX IF NOT EXISTS idx_system_actions_status
    ON system_actions (status, created_ts);

CREATE INDEX IF NOT EXISTS idx_execution_orders_trade
    ON execution_orders (trade_id, status);

CREATE INDEX IF NOT EXISTS idx_execution_fills_order
    ON execution_fills (order_id);

CREATE INDEX IF NOT EXISTS idx_reserved_funds_status
    ON reserved_funds (status);

CREATE INDEX IF NOT EXISTS idx_strategy_memory_stage
    ON strategy_memory (stage_gate, created_ts);