# Provenance of the code

The analyses of the paper were run with scripts whose comments, docstrings and messages were written in Indonesian.
For this public release they were translated into English. The translation changed no code.

| Path | Content |
| --- | --- |
| `original_code/` | Byte-identical copies of every script as it was run (registered analysis modules, data stage, follow-ups) |
| `verify_registered.py` | Checks the recorded SHA-256 hashes and the equivalence of the English scripts with the originals |
| `string_changes.txt` | Every string constant outside docstrings that the translation changed (file, line, before, after) |
| `translation_report.json` | Per file: number of replaced strings, result of the structural check, changed non-docstring strings |

## What is verified

```bash
python provenance/verify_registered.py
```

1. The six analysis modules in `original_code/` reproduce the combined analysis-code hash fixed at registration on
   1 October 2026 (`registration/registration.json`, registered at https://osf.io/b2vc9).
2. The follow-up scripts in `original_code/` reproduce the hashes written into the addenda before they were run
   (`registration/ADDENDUM_followup.md`, `registration/ADDENDUM_followup2.md`) and the hash recorded by
   `followup3/disagreement.py` in `results/followup3/disagreement.json`.
3. Every English script has the same abstract syntax tree as its original once all string constants are masked.

The analysis parameters in `config.json` reproduce the registered parameter hash; `tests/test_metrics.py` checks this.

## Repeating the registered analysis exactly

`run_all.py` recomputes the hash of the six analysis modules and refuses to run if it differs from the registration.
The English modules have different bytes (only their comments and messages differ), so for an exact repetition:

```bash
python provenance/verify_registered.py --install   # copies original_code/ over the English scripts
python run_all.py --config config.json             # needs the datacube; see docs/REPRODUCE.md
git checkout -- .                                  # restores the English scripts
```

The registration document, the addenda and the hashes are unchanged; the translation is recorded in `CHANGELOG.md`.
