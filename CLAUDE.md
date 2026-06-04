# RetailEdge — Claude Code Context

## What this project is

RetailEdge is a crypto trading bot built on Freqtrade.
Architecture: MVP_TradingBot_Freqtrade_Guarded_EdgeLearning v1.9.
Scope: CEX spot only. No leverage. No futures. No DEX.
Target exchange: Binance Spot (USDT pairs).
Deployment: VPS medium, 4 vCPU, 8GB RAM, Docker Compose.

The system is NOT a simple Freqtrade wrapper.
It has a multi-component sidecar architecture with safety guarantees.
Read this file before touching any component.

---

## Hard constraints — never violate these

1. Runtime loads exactly ONE active champion model at all times.
2. Challenger models run in research mode only. They cannot touch live config.
3. No real-time model switching. Promotion requires operator approval + config recompile + controlled restart.
4. `emergency_exit_enabled` defaults to false. Enable only after drill test passes.
5. Entry is blocked if reserved funds reconciliation fails on startup.
6. ATR percentile requires minimum 5000 candles. No trade if baseline is missing.
7. Every Decision Bus action must have: producer, consumer, actuator, success check, retry, and test.
8. Config compiler must fail closed. Any capability mismatch = build failure.

---

## Repo structure

```
retailedge/
├── CLAUDE.md                          # This file. Read first.
├── docker-compose.yml                 # Freqtrade + sidecar services
├── .env.example                       # Required env vars (never commit .env)
│
├── freqtrade/
│   ├── config.generated.json          # Auto-generated. Do not hand-edit.
│   ├── config.base.json               # Human-edited base config template
│   ├── user_data/
│   │   ├── strategies/
│   │   │   └── RetailEdgeStrategy.py  # Main Freqtrade strategy file
│   │   ├── models/                    # FreqAI model artifacts (gitignored)
│   │   └── logs/
│
├── sidecar/
│   ├── guardian/
│   │   ├── guardian.py                # Guardian main loop
│   │   ├── decision_bus.py            # Decision Bus producer/consumer
│   │   ├── emergency_exit.py          # Emergency exit cascade
│   │   └── alerting.py
│   ├── reconciler/
│   │   ├── reconciler.py              # Order state machine + fill events
│   │   ├── reserved_funds.py          # Reserved funds ledger
│   │   └── startup_reconcile.py      # Startup recovery from exchange open orders
│   ├── health_monitor/
│   │   └── monitor.py                 # Net expectancy, drawdown, slippage, fill rate
│   └── stoploss_auditor/
│       └── auditor.py                 # Periodic stoploss verification
│
├── research/
│   ├── worker.py                      # Research Worker entry point
│   ├── atr_percentile_service.py      # ATR percentile baseline (5000 candle min)
│   ├── edge_discovery/
│   │   ├── bucket_baseline.py         # Bucket strategy baseline
│   │   └── freqai_challenger.py       # FreqAI challenger (research only)
│   └── validation/
│       ├── cpcv.py                    # CPCV / CSCV
│       ├── pbo.py                     # Probability of Backtest Overfitting
│       └── dsr.py                     # Deflated Sharpe Ratio
│
├── deployment/
│   ├── champion_controller.py         # Champion Deployment Controller
│   ├── config_compiler.py             # Config generator + capability validator
│   ├── capability_matrix.json         # Exchange capability contract
│   ├── venue_costs.json               # Fee model
│   └── active_model_manifest.json     # Single source of truth for active model
│
├── ledger/
│   ├── schema.sql                     # SQLite schema (all tables)
│   └── ledger.py                      # Ledger read/write interface
│
└── tests/
    ├── test_champion_deployment.py
    ├── test_emergency_exit.py
    ├── test_reserved_funds.py
    ├── test_atr_percentile.py
    ├── test_capability_matrix.py
    └── test_decision_bus.py
```

---

## Component responsibilities — one-line each

| Component | What it owns |
|---|---|
| Freqtrade Execution Engine | Entry/exit order placement, stoploss_on_exchange, callbacks |
| Champion Deployment Controller | Selects 1 champion, compiles config, writes manifest, controls restart |
| Guardian + Decision Bus | Consumes PAUSE/EXIT/ALERT actions, verifies actuator success, retries |
| Reconciler Worker | Order state machine, fill deduplication, reserved funds post-fill |
| Startup Reconciler | Recomputes reserved funds from exchange open orders on every boot |
| Reserved Funds Ledger | Source of truth for projected available balance per entry |
| Pre-Trade Gate | Blocks entry if balance, exposure, regime, or notional check fails |
| Stoploss Auditor | Periodically verifies exchange-side stop exists; triggers fallback if not |
| Emergency Exit Cascade | market -> aggressive limit -> OPERATOR_REQUIRED (disabled by default) |
| ATR Percentile Service | Computes volatility percentile against 5000-candle baseline, refreshes 6h |
| Adaptive Regime Policy | Adjusts position multiplier based on detected regime performance |
| Health Monitor | Tracks net expectancy, drawdown, losing streak, slippage, fill rate |
| Research Worker | Runs bucket baseline + FreqAI challenger, enforces OOS gate, stores evidence |
| Strategy Memory Store | Stores strategy versions, evidence, promotion decisions, lessons learned |
| Config Compiler | Generates config.generated.json, fails closed on any capability mismatch |

---

## Decision Bus action types

These are the only valid action types. Do not invent new ones without adding to schema.sql.

```
PAUSE_REQUIRED
RESUME_ENTRY
EMERGENCY_EXIT
REJECT_ENTRY
STOP_UNCONFIRMED
RESERVED_MISMATCH_ON_STARTUP
SCHEDULED_MODEL_PROMOTION
OPERATOR_REQUIRED
EMERGENCY_EXIT_FAILED
```

Every action must have: action_id, action_type, severity, status, retry_count, created_ts.
Mark DONE only on confirmed 2xx from Freqtrade REST. Never assume success.

---

## Active model manifest — single source of truth

File: `deployment/active_model_manifest.json`

```json
{
  "strategy_id": "string",
  "model_id": "string — unique per training run",
  "model_type": "bucket_baseline | freqai_challenger",
  "model_hash": "sha256:...",
  "feature_hash": "sha256:...",
  "cost_model_hash": "sha256:...",
  "regime_policy_hash": "sha256:...",
  "approved_by": "operator",
  "approved_at": "ISO8601",
  "promotion_source": "scheduled_review | manual",
  "rollback_model_id": "string"
}
```

Rules enforced by config_compiler.py:
- If `multiple active_model_id detected` -> fail build.
- If `challenger_live == true` -> fail build.
- If `model_hash != deployed model file hash` -> fail build.

---

## Exchange capability contract

File: `deployment/capability_matrix.json`

Current target: `binance_spot`

Required fields that must be true for config to compile:
- `post_only_supported: true` (maker_first mode)
- `stoploss_on_exchange_supported: true`
- `stoploss_order_types_supported: ["stop_limit"]`
- `market_order_supported: true` (required only if emergency_exit_enabled)
- `freqtrade_min_version: "2026.1"`

If any field fails: `config_compiler.py` raises ValueError and build stops.

---

## Ledger schema summary

File: `ledger/schema.sql`

Tables:
- `system_actions` — Decision Bus queue
- `execution_orders` — All orders placed
- `execution_fills` — All fill events (deduplicated by order_id + fill_id)
- `reserved_funds` — Per-order reserved quote amounts
- `reserved_funds_reconciliation` — Startup reconciliation audit log
- `trade_state_flags` — Per-trade safety flags
- `strategy_memory` — Candidate evidence store
- `deployment_audit` — Champion promotion history

---

## Config compiler rules — fail conditions

```
venue.maker_first == true AND capability.post_only_supported != true       -> FAIL
venue.stoploss_on_exchange_supported == true AND config.stoploss_on_exchange != true -> FAIL
venue.stoploss_order_type not in capability.stoploss_order_types_supported -> FAIL
venue.emergency_exit_enabled == true AND capability.market_order_supported != true -> FAIL
manifest.model_hash != sha256(deployed model file)                         -> FAIL
multiple active model_id detected in manifest                              -> FAIL
freqtrade_version < capability.freqtrade_min_version                       -> FAIL
```

---

## Test gate mapping

| Test | Gate required before |
|---|---|
| test_champion_only_one_active_model | Dry-run |
| test_challenger_cannot_auto_promote | Dry-run |
| test_scheduled_model_promotion_manifest_hash | Dry-run |
| test_reload_config_or_restart_window | Dry-run |
| test_reserved_funds_startup_reconcile_match | Dry-run |
| test_reserved_funds_startup_mismatch_blocks_entry | Dry-run |
| test_atr_percentile_min_lookback | Dry-run |
| test_stoploss_interval_uses_atr_baseline | Dry-run |
| test_exchange_capability_matrix_compile_fail_post_only | Dry-run |
| test_exchange_capability_matrix_compile_fail_stoploss_type | Dry-run |
| test_exchange_capability_matrix_compile_fail_market_exit | Dry-run |
| test_emergency_exit_market_success | Micro-live B1 |
| test_emergency_exit_market_fail_limit_fallback | Micro-live B1 |
| test_emergency_exit_all_fail_operator_alert | Micro-live B1 |
| test_no_orphan_decision_contract | Always |

---

## What Claude Code should do in each session

**Before writing any code:**
1. Read this file.
2. Identify which component and which layer (Data / Logic / UI / Edge case).
3. Check which test gate covers this component.
4. Confirm the work does not touch more than one component per session.

**When generating config files:**
Always route through `config_compiler.py`. Never hand-edit `config.generated.json`.

**When generating a new Decision Bus action:**
Add to `schema.sql` first. Then implement producer. Then consumer. Then test.

**When touching reserved funds logic:**
Run `test_reserved_funds_startup_reconcile_match` and `test_reserved_funds_startup_mismatch_blocks_entry` before committing.

**When touching champion deployment:**
Verify `active_model_manifest.json` has exactly one model_id and `challenger_live` is absent or false.

---

## Freqtrade REST endpoints used by Guardian

```
GET  /api/v1/ping          -> liveness check before any action
POST /api/v1/pause         -> pause entry
POST /api/v1/reload_config -> apply new champion config
POST /api/v1/forceexit     -> emergency exit (market or limit)
GET  /api/v1/trades        -> reconciler trade state sync
GET  /api/v1/openorders    -> reconciler open order sync
```

All calls must check HTTP status. 2xx = success. Anything else = retry or OPERATOR_REQUIRED.

---

## Staging gates summary

| Stage | Key condition | Entry blocked if |
|---|---|---|
| Stage A (Dry-run) | All 11 dry-run tests pass | Any test fails |
| Stage B1 (Micro-live) | Emergency exit drill pass + reserved funds startup pass | Drill not completed |
| Stage B2 (Micro-live val.) | 100+ live trades, CI lower bound > 0 | Trade count insufficient |
| Stage C (Production) | 200+ trades, DSR >= 0.95, PBO <= 0.10 | Economic gate fails |

---

## Environment variables required

```
FREQTRADE_API_URL=http://localhost:8080
FREQTRADE_API_USER=
FREQTRADE_API_PASS=
EXCHANGE_API_KEY=
EXCHANGE_SECRET=
TELEGRAM_BOT_TOKEN=          # optional, for operator alerts
TELEGRAM_CHAT_ID=            # optional
LEDGER_DB_PATH=./ledger/retailedge.db
ACTIVE_MODEL_MANIFEST_PATH=./deployment/active_model_manifest.json
CAPABILITY_MATRIX_PATH=./deployment/capability_matrix.json
VENUE_COSTS_PATH=./deployment/venue_costs.json
```

---

## Key references in v1.9 blueprint

- Section 6 Step 1: Exchange Capability Matrix implementation
- Section 6 Step 2: Venue cost model
- Section 6 Step 3: Champion Deployment Controller + promote_champion()
- Section 6 Step 4: Emergency Exit Cascade + rules
- Section 6 Step 5: Startup Reserved Funds Reconciliation + SQL schema
- Section 6 Step 6: ATR Percentile Baseline Service
- Section 7.1: Decision Bus SQL schema
- Section 7.2: Guardian consumer with verified success
- Section 7.3: Active model manifest enforcement
- Section 7.4: Reserved funds gate in pre-entry
- Section 8: Full test matrix
