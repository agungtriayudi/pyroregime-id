"""Freeze the analysis design before the 2026 evaluation: hash of the analysis parameters + the analysis code.

    python register.py --config config.json
Output: registration/registration.json (hashes) and registration/REGISTRATION.md (to be uploaded to OSF).
After this, run_all.py refuses to run if the analysis parameters or the analysis code change
(except with --allow-deviation, which is recorded in results.json and must be reported in the manuscript).
Data-source parameters (URLs, dataset IDs, file paths, panel_end) are deliberately NOT hashed.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ANALYSIS_KEYS = ["ref_years", "min_det", "k_range", "methods", "cluster_boot", "shift_years", "neutral_years",
                 "event_months", "event_min_det", "enso_years", "holdout_train_end", "leads", "bootstrap",
                 "neg_frac", "panel_start", "regions"]
ANALYSIS_CODE = ["grid_features.py", "regime_features.py", "cluster_regimes.py", "regime_shift.py",
                 "forecast.py", "report.py"]


def fingerprint(cfg, root="."):
    params = {k: cfg.get(k) for k in ANALYSIS_KEYS}
    h_par = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    h_code = hashlib.sha256()
    for f in ANALYSIS_CODE:
        h_code.update(f.encode()); h_code.update(Path(root, f).read_bytes())
    return params, h_par, h_code.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an earlier registration (NEVER after the 2026 results have been seen)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out = Path("registration"); out.mkdir(exist_ok=True)
    if (out / "registration.json").exists() and not a.force:
        raise SystemExit("A registration already exists. Do not overwrite it; use --force only before any evaluation has been run.")
    params, hp, hc = fingerprint(cfg)
    reg = {"registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "params": params, "params_sha256": hp, "code_sha256": hc, "code_files": ANALYSIS_CODE}
    json.dump(reg, open(out / "registration.json", "w"), indent=1)
    md = f"""# Pre-registration — Indonesian peatland fire regimes

Registered (UTC): {reg['registered_utc']}
Parameter hash (SHA-256): `{hp}`
Analysis-code hash (SHA-256): `{hc}`

## Primary confirmatory hypothesis
H4: on the 2026 hold-out (training targets up to {cfg['holdout_train_end']}, evaluation targets after it), the
regime-conditioned mixture of experts has higher AUC-PR than a single global model at lead L = 1. Test: paired
block bootstrap over 0.5° blocks ({cfg['bootstrap']} replicates); one-sided p = share of replicates with difference ≤ 0.
All other comparisons are exploratory.

## Fixed analysis choices
- Reference period {cfg['ref_years'][0]}–{cfg['ref_years'][1]}; cells with ≥ {cfg['min_det']} retained VIIRS detections.
- Number of regimes k in {cfg['k_range']}: among solutions with bootstrap stability ARI ≥ 0.80
  ({cfg['cluster_boot']} subsamples of 80 %), the highest silhouette. Algorithms: {', '.join(cfg['methods'])}.
- Shift analysis: El Niño years {cfg['shift_years']}, months {cfg['event_months']}, ≥ {cfg['event_min_det']} detections;
  neutral-year baseline {cfg['neutral_years']}.
- Leave-one-El-Niño-out years: {cfg['enso_years']}. Lead times: {cfg['leads']} months.
- Gradient boosting: 300 iterations, learning rate 0.05, 31 leaves, L2 = 1.0, early stopping on 10 %; no re-weighting.

## Full parameter set
```json
{json.dumps(params, indent=1)}
```
"""
    (out / "REGISTRATION.md").write_text(md)
    print(f"registration -> {out}/ (upload REGISTRATION.md to OSF before looking at the 2026 results)")


if __name__ == "__main__":
    main()
