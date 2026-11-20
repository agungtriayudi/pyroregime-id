"""Verify that the English-commented code of this repository is the code that was registered and run.

    python provenance/verify_registered.py              # check only (exit code 1 if any check fails)
    python provenance/verify_registered.py --install    # check, then copy the original code over the English code

The comments, docstrings and messages of the scripts were translated from Indonesian into English after all analyses
had been run. provenance/original_code/ holds byte-identical copies of the scripts as they were run. This script checks

  1. hashes: the original files reproduce every SHA-256 hash that was fixed before an analysis was run:
     - the combined hash of the six analysis modules in registration/registration.json (computed as in register.py),
     - the hashes of the follow-up scripts listed in registration/ADDENDUM_followup.md and ADDENDUM_followup2.md,
     - the hash of followup3/disagreement.py recorded in results/followup3/disagreement.json (meta.script_sha256);
  2. translation: every English script has the same abstract syntax tree as its original once every string constant
     (docstrings and messages included) is replaced by a placeholder, i.e. the translation changed no code.
     The string constants that changed outside docstrings are listed in provenance/string_changes.txt.

run_all.py refuses to run unless the six analysis modules reproduce the registered hash. To repeat the registered
analysis exactly, use --install, which copies the original files over the English ones; restore the English files
afterwards with `git checkout -- .`.
"""
import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "provenance" / "original_code"


class Blank(ast.NodeTransformer):
    """Replace every str constant by a placeholder, so that only the structure of the code is compared."""

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="<str>"), node)
        return node


def structure(path):
    return ast.dump(Blank().visit(ast.parse(path.read_text(encoding="utf-8"))), include_attributes=False)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def addendum_hashes(path):
    """'- followup/x.py: `<64 hex>`' lines of an addendum -> {relative path: hash}."""
    pat = re.compile(r"^- ([\w/]+\.py): `([0-9a-f]{64})`", re.M)
    return dict(pat.findall(path.read_text(encoding="utf-8")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="copy provenance/original_code/ over the English scripts")
    a = ap.parse_args()
    ok = True

    def report(passed, text):
        nonlocal ok
        ok &= passed
        print(("PASS  " if passed else "FAIL  ") + text)

    # 1a. registered analysis modules (combined hash, as in register.py)
    reg = json.load(open(ROOT / "registration" / "registration.json"))
    h = hashlib.sha256()
    for f in reg["code_files"]:
        h.update(f.encode())
        h.update((ORIG / f).read_bytes())
    report(h.hexdigest() == reg["code_sha256"],
           f"registered analysis code ({len(reg['code_files'])} modules) -> {h.hexdigest()[:12]}… "
           f"(registration.json: {reg['code_sha256'][:12]}…)")

    # 1b. follow-up scripts whose hashes were fixed in the addenda before they were run
    fixed = {}
    for add in ("ADDENDUM_followup.md", "ADDENDUM_followup2.md"):
        for rel, hx in addendum_hashes(ROOT / "registration" / add).items():
            fixed[rel] = (hx, add)
    meta = json.load(open(ROOT / "results" / "followup3" / "disagreement.json"))["meta"]
    fixed["followup3/disagreement.py"] = (meta["script_sha256"], "results/followup3/disagreement.json")
    for rel, (hx, src) in sorted(fixed.items()):
        got = sha256(ORIG / rel)
        report(got == hx, f"{rel} -> {got[:12]}… ({src}: {hx[:12]}…)")

    # 2. the English scripts differ from the originals only in string constants
    originals = sorted(p.relative_to(ORIG) for p in ORIG.rglob("*.py"))
    for rel in originals:
        eng = ROOT / rel
        if not eng.exists():
            report(False, f"{rel}: English version missing")
            continue
        same = structure(ORIG / rel) == structure(eng)
        report(same, f"{rel}: same syntax tree as the original (string constants masked)")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    if a.install:
        if not ok:
            raise SystemExit("not installing: fix the failed checks first")
        for rel in originals:
            shutil.copy2(ORIG / rel, ROOT / rel)
        print(f"copied {len(originals)} original files to {ROOT} (restore the English files with `git checkout -- .`)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
