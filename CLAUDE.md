# RetailEdge (adaptive-bot-v15) — Claude Code Context

File ini memandu perilaku Claude Code di proyek ini. Ia BUKAN security boundary. Penegakan nyata ada di hook PreToolUse, permissions, dan isolasi OS. Kalau isi file ini berbeda dari hook atau settings.json, hook yang menang.

---

## Profil Operator (baca dulu, ini mengubah caramu bekerja)

Operator non-teknis. Tidak bisa coding, tidak paham data science. Maka:

- Jelaskan tiap langkah dengan bahasa awam. Jangan asumsikan pengetahuan coding.
- Satu langkah pada satu waktu. Sebut hasil yang terlihat sebagai tanda langkah berhasil.
- JANGAN PERNAH membuat atau mengubah file approval (`~/.adaptive-bot-ops/*.json`). Operator membuatnya manual di terminal.
- JANGAN PERNAH mengubah `dry_run` jadi `false` atas inisiatif sendiri. Live butuh keputusan operator plus verifikasi keamanan manusia kompeten.
- Saat hasil WFO keluar, JANGAN deklarasikan edge tervalidasi. Sajikan angka, serahkan verdict ke audit manusia (lihat Edge Belum Terbukti).
- Saat menyentuh batas yang butuh kompetensi teknis (verifikasi keamanan live, audit statistik edge), katakan terus terang bahwa ini di luar yang bisa dipastikan otomatis.

---

## Project Overview

Adaptive crypto trading bot, multi-venue. Tokocrypto utama, Indodax fallback. Pair IDR.
Stack: Python 3.11+, Freqtrade (execution engine), ccxt (di bawah Freqtrade), GP/GA untuk edge discovery, Redis untuk state, YAML config, JSON venue profile.

Status proyek: cetak biru, pra-implementasi. Belum ada kode jalan. Belum ada bukti edge.

---

## Prinsip Inti (dari v15)

```text
Capability first.
Edge proof second.
Venue-legal first, bukan fee first.
Execution truth before live funds.
Arsitektur menyesuaikan venue, bukan ditulis ulang.
```

---

## Arsitektur Eksekusi

Write path Phase 1 sampai 3 adalah Freqtrade, BUKAN script di `execution/`.

- `execution/` (ledger, reconciler, exchange_truth): reconciliatory, dipanggil Freqtrade. Bukan executor mandiri.
- `guardian/`: protective close, jalan sebagai service terpisah dengan heartbeat.
- Live trading = `freqtrade trade` dengan `dry_run: false`. BUKAN `python execution/*.py`.
- CCXT-authoritative (Phase 4-5) DITUNDA karena Tokocrypto punya stop dan clientOrderId.

---

## Model Gerbang dan Disiplin STOP

Gerbang memerintah laju, bukan kalender. STOP itu hukum. Jadwal molor dihormati.

```text
Gate -1   Venue + legal + pajak + IDR rail
Gate 0A   Capability probe + uji empiris stop dan clientOrderId + cost floor
EDGE      Discovery + walk-forward distratifikasi regime + anti-overfitting  (gerbang sebenarnya)
Gate 0B   Full capability register
CHAOS     Top-3 chaos test + verifikasi stop
LIVE      Micro-live strict + parameter health monitor
```

Jika sebuah gate gagal, jangan buka sprint berikut. Sampaikan STOP ke operator, jangan dipaksa lewat.

---

## Edge Belum Terbukti (aturan perilaku mengikat)

Tidak ada bukti edge sampai saat ini. Edge adalah gap kritis, bukan infrastruktur.

- Setelah menjalankan WFO, sajikan: jumlah fold positif, drawdown per fold, profit factor, hasil anti-data-snooping. JANGAN simpulkan "edge valid." Verdict diserahkan ke manusia yang paham statistik.
- Anti-data-snooping memakai jumlah trial SEBENARNYA (semua generasi x populasi), bukan jumlah finalis. Pakai Deflated Sharpe, PBO, atau White Reality Check sesuai jumlah trial.
- Cost floor wajib stack penuh: maker/taker + levy CFX + pajak kripto + median spread + slippage konservatif + buffer + margin. Maker 0,10 persen saja tidak cukup. GP/GA dilarang menguji take-profit di bawah floor ini.
- Fold WFO wajib mencakup bull, bear, dan chop terlabel. Tolak hasil yang fold-nya tidak pernah melihat downtrend.
- Baseline MR_RANGE_REVERT long-only menangkap pisau jatuh saat downtrend. Ingatkan operator soal ini.

---

## Safety Boundary

ZONA AMAN (autorun): `config/` `tools/` (passive) `venue/` `edge/` `validation/` `chaos_test/` `monitoring/` `reports/`

ZONA TERLARANG (butuh approval mekanis di luar repo):

```text
freqtrade trade            live, butuh ~/.adaptive-bot-ops/live_approval.json
execution/ guardian/ run   direct run diblokir hook (bangun lewat permissions.ask)
tools/full_capability_probe.py   Gate 0B order empiris, butuh probe_approval
tools/mini_capability_probe.py --tiny-order   Gate 0A order empiris, butuh probe_approval
```

Approval file TIDAK PERNAH dibuat via Claude. Operator membuatnya manual di terminal, di luar repo, chmod 600.

---

## Capability Branching

Subsistem membaca Venue Capability Profile dan bercabang lewat conditional. Tidak ada lapisan executor tambahan.

```text
stop_on_exchange TRUE   -> stoploss_on_exchange true, guardian backup, overnight boleh di bawah stop
stop_on_exchange stop_limit_only -> guardian tetap exit market untuk gap-through, bukan sekadar backup
stop_on_exchange FALSE  -> guardian primary, overnight DILARANG, operator wajib hadir
client_order_id TRUE    -> ledger authority recon_key (bukan idempotency persisten)
client_order_id FALSE   -> observational, status UNKNOWN memblokir entry
idr_rail FALSE          -> modul off-ramp + risk register FX aktif
```

---

## Commands

Safe, autorun:

```text
python tools/mini_capability_probe.py
python validation/regime_labeler.py
python edge/gp_ga/run_fast_screening.py
python edge/gp_ga/run_full_wfo.py
freqtrade backtesting --config <dry_config>
freqtrade hyperopt --config <dry_config>
pytest chaos_test/
pytest validation/
```

Build, aman dan diizinkan hook:

```text
python -m py_compile execution/idempotency_ledger.py
mypy execution guardian
ruff check execution guardian
```

Pre-flight live, wajib lolos sebelum `freqtrade trade` live:

```text
python tools/check_live_config.py --config config/operational_mode.yaml \
  --assert-dry-run-false --assert-max-open-trades 1 \
  --assert-stake-fraction-max 0.01 --assert-protections-enabled \
  --assert-guardian-required-by-profile
```

Live, hanya oleh operator setelah verifikasi keamanan manusia:

```text
freqtrade trade --config config/operational_mode.yaml --strategy <strategy>
```

---

## Monitoring

`monitoring/parameter_health_monitor.py` jalan sebagai service atau cron. Tier 1 event-based aktif dari trade pertama: StoplossGuard, MaxDrawdown, CooldownPeriod, consecutive losing 5 pause. Parameter live TIDAK berubah otomatis. Adaptasi hanya di shadow dengan persetujuan operator.

---

## Venue Profiles

`config/venue_profiles/*.json` sumber kebenaran tunggal. Ditulis oleh probe Gate 0A, bukan diketik manual.
`stop_on_exchange` adalah enum: `False`, `stop_limit_only`, atau `stop_market`. `verified_at` null sampai probe selesai.

---

## Secret dan Git

- Jangan pernah commit `.env` atau secret apa pun. Pastikan `.gitignore` memblokir `.env`, `secrets/`, dan file approval.
- API key venue hanya izin trading, BUKAN withdrawal.
- Sebelum commit pertama, verifikasi `git status` tidak menampilkan secret.
- Pertahanan uang live yang sebenarnya adalah isolasi OS, bukan hook. Sarankan operator menjalankan Claude Code sebagai user terpisah atau di sandbox.

---

## Reports

Urutan: gate0a (passive lalu empirical) -> WFO full (phase0_decision) -> gate0b -> chaos.

---

## Batas Penegakan (kejujuran)

File ini memandu, tidak menegakkan. Hook regex cuma speed bump dan bisa dikalahkan command yang sengaja disamarkan. Pengamanan uang live yang sebenarnya adalah isolasi OS: Claude Code sebagai user terpisah tanpa hak tulis ke approval, atau di sandbox. Perlakukan setiap instruksi di file ini sebagai default perilaku, bukan jaminan keamanan.
