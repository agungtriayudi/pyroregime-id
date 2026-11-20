# Deviations from the registration

Every departure from the registered design is recorded here: date, what changed, why, and whether the 2026 results
had been seen at that time. The entry of 1 October 2026 is translated from the Indonesian original; the entry of
20 November 2026 was added for the public release and summarises `registration/ADDENDUM_followup3.md`. Times are
Western Indonesian Time (WIB = UTC+7).

**No deviation from the registered design.** The analysis parameters and the six registered analysis modules were not
changed, and `run_all.py` ran with matching hashes (`results/results.json: meta.registration_status = "registered"`).
The notes below record an inconsistency between documents and the post hoc change made in follow-up 3.

## 2026-10-01 10:59 WIB: text of REGISTRATION.md versus the registered code (no change to the analysis)

- **What:** The text of `registration/REGISTRATION.md` (a template in `register.py`, uploaded to OSF b2vc9) states
  the rule for selecting the number of regimes as "among solutions with bootstrap stability ARI ≥ 0.80, the highest
  silhouette". The hashed analysis code (`cluster_regimes.py: select_k`, unchanged) applies: k = the GMM solution with
  the lowest BIC among the stable GMM solutions (bootstrap ARI ≥ 0.80); if the minimum lies at the upper end of the
  range of k, the knee of the BIC curve; if no GMM solution is stable, the highest silhouette as a fallback. The final
  partition is taken from the method with the highest stability (rounded to two decimals), then the highest
  silhouette, at the selected k.
- **Action:** No change of code or parameters. The analysis ran with the registered code (hashes matched).
  `results.json` records the rule that was actually applied (`selection.json: rule`). The partition that the prose
  rule would have selected (k-means, k = 4) is reported as an exploratory alternative (follow-up 1, Supplement S3).
- **When found:** while `run_all.py` was running the clustering (started 10:40); **no results had been seen**
  (the clustering had not written any output; `results/regimes/` was still empty).

## 2026-11-20: kappa replaced by disagreement components (post hoc, follow-up 3)

- **What:** Addendum 2 (https://osf.io/8vr7f) summarised the agreement of S-NPP and NOAA-20 on monthly fire
  occurrence with Cohen's kappa. After the results were known, an editor pointed out that kappa should not be used to
  compare maps (Pontius and Millones, 2011; Foody, 2020). The same 2 × 2 tables were re-expressed as total, quantity
  and allocation disagreement by `followup3/disagreement.py`, as fixed in `registration/ADDENDUM_followup3.md`.
- **Action:** The registered analysis, the addenda 1 and 2 and their results are unchanged. The bootstrap uses the
  seed and the order of draws of follow-up 2, and the script checks that the follow-up-2 intervals are reproduced
  exactly. The kappa values remain in `results/followup2/intersensor.json` (Table S12 of the paper, marked superseded).
- **When:** after the results of the registered run and of follow-ups 1 and 2 had been seen; the change is **post hoc**
  and is reported as such.
