# gLAB `.out` dashboard

This folder contains a tiny workflow to turn a gLAB text `.out` file into a browsable dashboard where you can:

- Select a `MESSAGE` type (`MODEL`, `FILTER`, `OUTPUT`, `PREFIT`, `POSTFIT`, …)
- Plot time series (`X` vs `Y`), scatters, and histograms
- Filter by `system` / `prn` / `meas` where available

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 1) Build the cache (CSV tables)

```bash
python3 -m glabdash.ingest TLSA00615.out --out-dir .glabdash_cache
```

This creates one CSV per message type in `.glabdash_cache/` plus `.glabdash_cache/manifest.json`.

### 2) Run the dashboard

```bash
streamlit run app.py
```

You can also build/refresh the cache from inside the app (sidebar button).

## Notes

- `TLSA00615.out` in this repo is **gLAB v6.0.0** format (see the header in the file).
- The ingest parser is intentionally **v6-only** to keep the code simpler and easier to maintain.
- Messages like `MODEL*` / `PREFIT*` are normalized using a `used` column (`1` for normal, `0` for `*`).
