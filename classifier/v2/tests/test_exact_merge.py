import numpy as np
import pandas as pd

from dataset import build_object


def rows(n=60):
    t = np.linspace(0, 100, n)
    return pd.DataFrame({"diaObjectId": 1, "mjd": 60000 + t, "band": ["g", "r"] * (n // 2),
                         "mag": 18 + 0.1 * np.sin(2 * np.pi * t / 0.7)})


def test_only_identical_periods_are_removed():
    cand = {"period_1_LS": 0.70000, "period_1_CE": 0.70000, "period_1_AOV": 0.70010, "period_2_LS": 1.4}
    obj = build_object(rows(), cand, mag_transform=None, scale_kind="none")
    np.testing.assert_allclose(np.sort(obj["periods"]), [0.7, 0.7001, 1.4], rtol=1e-6)   # 0.7 kept once


def test_order_and_cycles():
    cand = {"period_1_LS": 0.3, "period_1_CE": 2.0, "period_2_CE": 0.9}
    obj = build_object(rows(), cand, mag_transform=None, scale_kind="none")
    assert np.all(np.diff(1 / obj["periods"]) > 0)                                # increasing frequency
    np.testing.assert_allclose(obj["cycles"], obj["baseline"] / obj["periods"], rtol=1e-6)
