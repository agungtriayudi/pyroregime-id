"""Fill the [[...]] tokens of the manuscript (Markdown export of the Claude Doc) with numbers from results.json.

    python fill_manuscript.py --manuscript manuscript.md --results results/results.json \
        --tables results/tables --out manuscript_filled.md
    python fill_manuscript.py --check manuscript_tokens.txt --results results/results.json --tables results/tables
The Markdown export of the Claude Doc escapes tokens (\\[\\[data.n\\_cells\\]\\], &#91;\\[table.x\\]\\]); the text is
normalised before filling. Always writes fill_values.json (token -> formatted text) next to --out
or --results, for filling the original Claude Doc.
Formatting rules: shift fractions (frac_changed, ci_low/ci_high under shift, baseline_mean) -> percent, 1 decimal;
integers -> thousands separator; p < 0.001 -> "< 0.001"; everything else 3 decimals.
Tokens that are not found are left in place and reported (exit code 1), so that no number is filled silently.
"""
import argparse
import json
import re
import sys
from pathlib import Path

TOKEN = re.compile(r"`?\[\[([A-Za-z0-9_.\-]+)\]\]`?")


def lookup(res, path):
    cur = res
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def fmt(path, v):
    if v is None:
        return "n/a"
    last = path.split(".")[-1]
    if path.startswith("shift.") and last in ("frac_changed", "ci_low", "ci_high", "baseline_mean"):
        return f"{100 * v:.1f} %"
    if isinstance(v, bool) or isinstance(v, str):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if last == "p":
        return "< 0.001" if v < 0.001 else f"{v:.3f}"
    return f"{v:.3f}"


def unescape_md(text):
    r"""Undo the Markdown escaping of tokens: \[\[a\_b\]\] and &#91;\[x\]\] -> [[a_b]] / [[x]]."""
    text = text.replace("&#91;", "\\[").replace("&#93;", "\\]")
    pat = re.compile(r"\\\[\\\[([^\]\[]+?)\\\]\\\]")
    return pat.sub(lambda m: "[[" + m.group(1).replace("\\_", "_") + "]]", text)


def values_for(tokens, res, tables):
    vals, missing = {}, []
    for t in tokens:
        if t.startswith("table."):
            name = t.split(".", 1)[1]
            (vals.__setitem__(t, tables[name]) if name in tables else missing.append(t))
            continue
        try:
            vals[t] = fmt(t, lookup(res, t))
        except KeyError:
            missing.append(t)
    return vals, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript")
    ap.add_argument("--check", help="list of tokens (one per line) to check without a manuscript")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = json.load(open(a.results))
    tables = {p.stem: p.read_text() for p in Path(a.tables).glob("*.md")}
    if a.check:
        toks = [l.strip() for l in open(a.check) if l.strip() and not l.startswith("#")]
        vals, miss = values_for(toks, res, tables)
        dest = Path(a.results).with_name("fill_values.json")
        json.dump(vals, open(dest, "w"), indent=1, ensure_ascii=False)
        print(f"{len(vals)}/{len(toks)} tokens have a value -> {dest}")
        if miss:
            print("NO VALUE:", ", ".join(miss)); sys.exit(1)
        return
    missing = []

    ignore = {"token", "..."}                                   # examples in the drafting notes

    def sub(m):
        path = m.group(1)
        if path in ignore:
            return m.group(0)
        if path.startswith("table."):
            name = path.split(".", 1)[1]
            if name in tables:
                return tables[name]
            missing.append(path); return m.group(0)
        try:
            return fmt(path, lookup(res, path))
        except KeyError:
            missing.append(path); return m.group(0)

    src = unescape_md(Path(a.manuscript).read_text())
    text = TOKEN.sub(sub, src)
    Path(a.out).write_text(text)
    toks = [t for t in TOKEN.findall(src) if t not in ignore]
    vals, _ = values_for(sorted(set(toks)), res, tables)
    json.dump(vals, open(Path(a.out).with_name("fill_values.json"), "w"), indent=1, ensure_ascii=False)
    n = len(toks)
    print(f"{n - len(missing)}/{n} tokens filled -> {a.out}")
    if missing:
        print("NOT FILLED:", ", ".join(sorted(set(missing))))
        sys.exit(1)


if __name__ == "__main__":
    main()
