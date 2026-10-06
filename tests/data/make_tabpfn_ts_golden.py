"""Regenerate tabpfn_ts_golden.json from the upstream TabPFN-TS feature code.

Usage: python tests/data/make_tabpfn_ts_golden.py /path/to/tabpfn-time-series
(needs gluonts, statsmodels and scipy; nothing else of the upstream package is imported).
"""
import importlib.util, json, sys, types
from pathlib import Path
import numpy as np, pandas as pd

UP = Path(sys.argv[1]).resolve() / "tabpfn_time_series"  # checkout of PriorLabs/tabpfn-time-series
pkg = types.ModuleType("tabpfn_time_series"); pkg.__path__ = [str(UP)]
sub = types.ModuleType("tabpfn_time_series.features"); sub.__path__ = [str(UP / "features")]
sys.modules["tabpfn_time_series"] = pkg; sys.modules["tabpfn_time_series.features"] = sub
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
load("tabpfn_time_series.features.feature_generator_base", UP / "features/feature_generator_base.py")
basic = load("tabpfn_time_series.features.basic_features", UP / "features/basic_features.py")
auto = load("tabpfn_time_series.features.auto_features", UP / "features/auto_features.py")

def upstream_features(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """FeatureTransformer.transform without the TimeSeriesDataFrame wrapper."""
    input_columns = set(train.columns)
    tr = train.assign(_is_train=True); te = test.assign(_is_train=False)
    te["target"] = te["target"].astype("float64")
    df = pd.concat([tr, te])
    for gen in (basic.RunningIndexFeature(), basic.CalendarFeature(), auto.AutoSeasonalFeature()):
        df = gen(df)
    gen_float = [c for c in df.columns if c not in input_columns and c != "_is_train" and df[c].dtype == np.float64]
    df = df.astype({c: np.float32 for c in gen_float})
    return df

cases = []
rng = np.random.default_rng(0)
specs = [("h", 120, 24, "2024-03-30 17:00"), ("D", 100, 12, "2023-12-20"), ("15min", 160, 32, "2024-10-27 00:00"), ("W-SUN", 70, 8, "2022-01-02"), ("MS", 48, 6, "2019-01-01")]
for k, (freq, n, h, start) in enumerate(specs):
    t = np.arange(n)
    y = 5 + 0.02 * t + np.sin(2 * np.pi * t / 7) + 0.5 * np.cos(2 * np.pi * t / 24) + rng.normal(scale=0.2, size=n)
    stamps = pd.date_range(start, periods=n + h, freq=freq)
    cov = rng.normal(size=n + h)
    train = pd.DataFrame({"target": y, "promo": cov[:n]}, index=pd.MultiIndex.from_arrays([[f"s{k}"] * n, stamps[:n]], names=["item_id", "timestamp"]))
    test = pd.DataFrame({"target": np.nan, "promo": cov[n:]}, index=pd.MultiIndex.from_arrays([[f"s{k}"] * h, stamps[n:]], names=["item_id", "timestamp"]))
    feats = upstream_features(train, test)
    X = feats.drop(columns=["target", "_is_train"])
    periods = [p for p, _ in auto.AutoSeasonalFeature.find_seasonal_periods(pd.Series(y), **auto.AutoSeasonalFeature().config)]
    cases.append({
        "freq": freq, "start": start, "n": n, "horizon": h,
        "y": [float(f"{v:.17g}") for v in y], "promo": [float(f"{v:.17g}") for v in cov],
        "columns": list(X.columns), "periods": periods,
        "X": [[float(f"{v:.7g}") for v in r] for r in X.to_numpy(dtype=float)],
    })
out = Path(__file__).resolve().parent / "tabpfn_ts_golden.json"
out.write_text(json.dumps({"source": "PriorLabs/tabpfn-time-series@e4637598172164d2498fa522cf44297f748c078f (v1.3.0) features/basic_features.py + auto_features.py", "cases": cases}))
print(out, [ (c["freq"], len(c["columns"]), c["periods"][:5]) for c in cases])
