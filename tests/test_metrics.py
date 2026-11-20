"""Unit tests of the agreement statistics and of the registration fingerprint (no data and no internet needed).

    python tests/test_metrics.py          # plain Python
    python -m pytest tests                # or with pytest, if installed

The expected values are computed by hand in the comments, so that the formulas of the paper (Section 2.5) can be
checked independently of the implementation.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "followup2"), str(ROOT / "followup3")]

import disagreement as fu3  # noqa: E402  (followup3/disagreement.py)
import intersensor as fu2  # noqa: E402  (followup2/intersensor.py)
from register import fingerprint  # noqa: E402


def close(x, y, tol=1e-12):
    return abs(float(x) - float(y)) < tol


def test_disagreement_components():
    # a = 40 (fire in both), b = 10 (reference only), c = 4 (other only), d = 946 (neither); n = 1000
    r = fu3.disagreement(40, 10, 4, 946)
    assert close(r["D"], 14 / 1000)          # (b + c) / n
    assert close(r["Q"], 6 / 1000)           # |b - c| / n
    assert close(r["A"], 8 / 1000)           # 2 min(b, c) / n
    assert close(r["D"], r["Q"] + r["A"])    # D = Q + A
    assert close(r["A_share"], 8 / 14)       # A / D
    assert close(r["D_fire"], 14 / 54)       # (b + c) / (a + b + c)


def test_disagreement_symmetry_and_vectorisation():
    # swapping the two records leaves D, Q and A unchanged; arrays are handled element-wise
    a, b, c, d = np.array([5, 30]), np.array([7, 2]), np.array([3, 9]), np.array([85, 59])
    r1, r2 = fu3.disagreement(a, b, c, d), fu3.disagreement(a, c, b, d)
    for k in ("D", "Q", "A"):
        assert np.allclose(r1[k], r2[k])
    assert np.allclose(r1["D"], r1["Q"] + r1["A"])


def test_kappa_and_positive_agreement():
    # textbook 2 x 2 table: a = 20, b = 5, c = 10, d = 15 -> po = 0.70, pe = 0.50, kappa = 0.40, ppos = 40 / 55
    for mod in (fu2, fu3):
        s = mod.agree_stats(20, 5, 10, 15)
        assert close(s["po"], 0.70) and close(s["kappa"], 0.40) and close(s["ppos"], 40 / 55)


def test_ccc_from_sums():
    # Lin's concordance correlation from block sums equals the direct formula with population moments
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = 0.8 * x + rng.normal(scale=0.5, size=500) + 0.1
    S = np.array([len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()])
    ccc, bias = fu2.ccc_from_sums(S)
    vx, vy = x.var(), y.var()
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    assert close(ccc, 2 * cov / (vx + vy + (x.mean() - y.mean()) ** 2), 1e-10)
    assert close(bias, y.mean() - x.mean(), 1e-10)


def test_block_bootstrap_sums():
    # every bootstrap replicate is a sum of whole blocks drawn with replacement
    M = np.array([[1, 0], [0, 1], [2, 2]])
    bs = fu3.boot_sums(M, 50, np.random.default_rng(0))
    assert bs.shape == (50, 2)
    possible = {(i + 2 * k, j + 2 * k) for i in range(4) for j in range(4) for k in range(4) if i + j + k == 3}
    assert set(map(tuple, bs.tolist())) <= possible


def test_registration_hashes():
    # the analysis parameters in config.json and the registered analysis modules (byte-identical copies in
    # provenance/original_code/) reproduce the two hashes fixed at registration (registration/registration.json)
    cfg = json.load(open(ROOT / "config.json"))
    reg = json.load(open(ROOT / "registration" / "registration.json"))
    _, h_par, h_code = fingerprint(cfg, root=ROOT / "provenance" / "original_code")
    assert h_par == reg["params_sha256"]
    assert h_code == reg["code_sha256"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"ALL {len(tests)} TESTS PASSED")
