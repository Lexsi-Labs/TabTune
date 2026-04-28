"""
TabTune Distillation v4 — Full Test Suite (21 tests)
======================================================
Core:      1-5   (cls MLP, cls LGBM, multi-teacher, regression, save/load)
Advanced:  6-10  (pre-fitted, augmentation, adaptive-T, CV-distill, calibration)
Deploy:    11-13 (ONNX export, quantization, benchmark)
Analysis:  14    (profiler report)
New:       15-16 (XGBoost, CatBoost students)
New:       17-18 (NaN inputs, very small dataset)
New:       19-21 (alpha ablation, soft-label cache reuse, temperature effect)

Usage:
    python test_distillation.py
    python test_distillation.py --quick
    python test_distillation.py --test 15
"""
import sys, os, time, logging, warnings, argparse, tempfile
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
warnings.filterwarnings("ignore")

from tabtune import TabularPipeline
from tabtune.logger import setup_logger
from tabtune.distillation import TabDistiller
setup_logger(use_rich=True)
logger = logging.getLogger("tabtune")
QUICK = False

def sep(t): print(f"\n{'='*72}\n  {t}\n{'='*72}\n")
def mc(n=1500,f=15,c=3):
    n=n//3 if QUICK else n
    X,y=make_classification(n_samples=n,n_features=f,n_informative=f//2,n_classes=c,n_clusters_per_class=1,random_state=42)
    X=pd.DataFrame(X,columns=[f"f{i}" for i in range(f)]); y=pd.Series(y)
    return train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
def mr(n=1500,f=15):
    n=n//3 if QUICK else n
    X,y=make_regression(n_samples=n,n_features=f,n_informative=f//2,noise=10,random_state=42)
    X=pd.DataFrame(X,columns=[f"f{i}" for i in range(f)]); y=pd.Series(y)
    return train_test_split(X,y,test_size=0.3,random_state=42)
EP = lambda: 80 if not QUICK else 20

# ── CORE ──────────────────────────────────────────────────────────────────────
def test_1():
    sep("TEST 1: TabPFN → MLP (Classification)")
    Xtr,Xte,ytr,yte=mc(); d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",student_params={"epochs":EP(),"patience":15})
    d.fit(Xtr,ytr); return d.compare(Xte,yte)
def test_2():
    sep("TEST 2: TabPFN → LightGBM (Classification)")
    Xtr,Xte,ytr,yte=mc(2000,20,2); d=TabDistiller(teachers="TabPFN",student="lgbm",task_type="classification")
    d.fit(Xtr,ytr); return d.compare(Xte,yte)
def test_3():
    sep("TEST 3: Multi-Teacher → MLP")
    Xtr,Xte,ytr,yte=mc(1200,12,2); d=TabDistiller(teachers=["TabPFN","TabICLv2"],student="mlp",task_type="classification",temperature=4.0,student_params={"epochs":EP()//2})
    d.fit(Xtr,ytr); return d.compare(Xte,yte)
def test_4():
    sep("TEST 4: Regression → MLP")
    Xtr,Xte,ytr,yte=mr(); d=TabDistiller(teachers="TabICLv2",student="mlp",task_type="regression",student_params={"epochs":EP()})
    d.fit(Xtr,ytr); return d.compare(Xte,yte)
def test_5():
    sep("TEST 5: Save/Load")
    Xtr,Xte,ytr,yte=mc(800,10,2); d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",student_params={"epochs":30})
    d.fit(Xtr,ytr); a1=d.evaluate(Xte,yte)["accuracy"]
    with tempfile.NamedTemporaryFile(suffix=".pkl",delete=False) as f: p=f.name
    d.save(p); d2=TabDistiller.load(p); a2=d2.evaluate(Xte,yte)["accuracy"]; os.unlink(p)
    assert a1==a2; print(f"  ✅ acc={a1}"); return {"accuracy":a1}

# ── ADVANCED ──────────────────────────────────────────────────────────────────
def test_6():
    sep("TEST 6: Pre-Fitted Teacher")
    Xtr,Xte,ytr,yte=mc(800,10,2)
    pipe=TabularPipeline(model_name="TabPFN",task_type="classification",tuning_strategy="inference"); pipe.fit(Xtr,ytr)
    d=TabDistiller(teachers=[pipe],student="mlp",task_type="classification",student_params={"epochs":30})
    d.fit(Xtr,ytr); r=d.evaluate(Xte,yte); print(f"  ✅ acc={r['accuracy']}"); return r
def test_7():
    sep("TEST 7: Data Augmentation")
    Xtr,Xte,ytr,yte=mc(800,10,2)
    d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",augment_factor=0.5,student_params={"epochs":EP()//2})
    d.fit(Xtr,ytr); r=d.evaluate(Xte,yte); print(f"  ✅ acc={r['accuracy']}"); return r
def test_8():
    sep("TEST 8: Adaptive Temperature (unit)")
    from tabtune.distillation.losses import compute_adaptive_temperatures
    conf=np.array([[0.95,0.03,0.02]]*100); unc=np.array([[0.35,0.33,0.32]]*100)
    t=compute_adaptive_temperatures(np.concatenate([conf,unc]),T_base=3.0)
    assert t[100:].mean()>t[:100].mean(); print(f"  ✅ Conf T={t[:100].mean():.2f}, Unc T={t[100:].mean():.2f}"); return {"ok":True}
def test_9():
    sep("TEST 9: CV Soft Labels")
    from tabtune.distillation.strategies import cross_validate_soft_labels
    X,y=make_classification(n_samples=300 if QUICK else 600,n_features=10,n_classes=2,random_state=42)
    s=cross_validate_soft_labels("TabPFN",pd.DataFrame(X),pd.Series(y),n_folds=3)
    assert s.shape==(len(X),2); print(f"  ✅ shape={s.shape}"); return {"ok":True}
def test_10():
    sep("TEST 10: Calibration")
    from tabtune.distillation.strategies import calibrate_student
    Xtr,Xte,ytr,yte=mc(1000,10,2)
    d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",student_params={"epochs":EP()//2})
    d.fit(Xtr,ytr); T=calibrate_student(d,Xte,yte); print(f"  ✅ T*={T:.3f}"); return {"T":T}

# ── DEPLOY ────────────────────────────────────────────────────────────────────
def test_11():
    sep("TEST 11: ONNX Export")
    from tabtune.distillation.exporters import export_onnx
    Xtr,Xte,ytr,yte=mc(800,10,2)
    d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",student_params={"epochs":20})
    d.fit(Xtr,ytr)
    with tempfile.NamedTemporaryFile(suffix=".onnx",delete=False) as f: p=f.name
    m=export_onnx(d,p); print(f"  ✅ {m['size_mb']:.2f}MB, {m['n_params']:,} params")
    try:
        import onnxruntime as ort
        sess=ort.InferenceSession(p); r=sess.run(None,{"input":Xte.iloc[:5].to_numpy().astype(np.float32)})
        print(f"  ✅ ORT inference OK: {r[0].shape}")
    except ImportError: print("  ⚠️ onnxruntime not installed")
    os.unlink(p); return m
def test_12():
    sep("TEST 12: INT8 Quantization")
    from tabtune.distillation.exporters import quantize_student
    Xtr,Xte,ytr,yte=mc(800,10,2)
    d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",student_params={"epochs":20},device="cpu")
    d.fit(Xtr,ytr); a1=d.evaluate(Xte,yte)["accuracy"]
    quantize_student(d); a2=d.evaluate(Xte,yte)["accuracy"]
    print(f"  Before: {a1}, After: {a2}, Drop: {(a1-a2)*100:.1f}%"); return {"before":a1,"after":a2}
def test_13():
    sep("TEST 13: Inference Benchmark")
    from tabtune.distillation.exporters import benchmark_inference
    Xtr,Xte,ytr,yte=mc(1000,15,3)
    d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",student_params={"epochs":20})
    d.fit(Xtr,ytr); s=benchmark_inference(d,Xte,n_runs=50)
    print(f"  ✅ {s['mean_ms']:.1f}ms, {s['throughput_per_sec']:,.0f} samples/sec"); return s

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
def test_14():
    sep("TEST 14: Distillation Profiler")
    from tabtune.distillation.analysis import DistillationProfiler
    Xtr,Xte,ytr,yte=mc(1000,12,3)
    d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",student_params={"epochs":EP()//2})
    d.fit(Xtr,ytr)
    p=DistillationProfiler(d); r=p.profile(Xte,yte); p.print_report(r)
    with tempfile.NamedTemporaryFile(suffix=".json",delete=False) as f: path=f.name
    p.save_report(r,path); os.unlink(path); return r

# ── NEW: STUDENT COVERAGE ─────────────────────────────────────────────────────
def test_15():
    sep("TEST 15: TabPFN → XGBoost (Classification)")
    Xtr,Xte,ytr,yte=mc(2000,20,2)
    d=TabDistiller(teachers="TabPFN",student="xgb",task_type="classification")
    d.fit(Xtr,ytr); r=d.compare(Xte,yte); print(f"  ✅ acc={r.get('student_accuracy', r)}"); return r
def test_16():
    sep("TEST 16: TabPFN → CatBoost (Classification)")
    Xtr,Xte,ytr,yte=mc(2000,20,2)
    d=TabDistiller(teachers="TabPFN",student="catboost",task_type="classification")
    d.fit(Xtr,ytr); r=d.compare(Xte,yte); print(f"  ✅ acc={r.get('student_accuracy', r)}"); return r

# ── NEW: EDGE CASES ───────────────────────────────────────────────────────────
def test_17():
    sep("TEST 17: NaN Inputs")
    Xtr,Xte,ytr,yte=mc(800,10,2)
    # inject NaNs into ~10% of values
    Xtr=Xtr.copy(); mask=np.random.RandomState(0).rand(*Xtr.shape)<0.1; Xtr[mask]=np.nan
    Xte=Xte.copy(); mask2=np.random.RandomState(1).rand(*Xte.shape)<0.1; Xte[mask2]=np.nan
    d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",student_params={"epochs":20})
    d.fit(Xtr,ytr); r=d.evaluate(Xte,yte); print(f"  ✅ acc={r['accuracy']} (with NaNs)"); return r
def test_18():
    sep("TEST 18: Very Small Dataset (n=50)")
    X,y=make_classification(n_samples=50,n_features=8,n_informative=4,n_classes=2,n_clusters_per_class=1,random_state=42)
    Xtr,Xte,ytr,yte=train_test_split(pd.DataFrame(X),pd.Series(y),test_size=0.3,random_state=42,stratify=y)
    d=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",student_params={"epochs":20,"patience":5})
    d.fit(Xtr,ytr); r=d.evaluate(Xte,yte); print(f"  ✅ acc={r['accuracy']} (n=50)"); return r

# ── NEW: DISTILLATION CORRECTNESS ─────────────────────────────────────────────
def test_19():
    sep("TEST 19: Alpha Ablation (alpha=0.0 vs alpha=1.0)")
    Xtr,Xte,ytr,yte=mc(1000,10,2)
    d0=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",alpha=0.0,student_params={"epochs":EP()//2})
    d1=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",alpha=1.0,student_params={"epochs":EP()//2})
    d0.fit(Xtr,ytr); d1.fit(Xtr,ytr)
    p0=d0.student_.predict_proba(Xte); p1=d1.student_.predict_proba(Xte)
    diff=np.abs(p0-p1).mean()
    assert diff>0.005, f"alpha=0.0 and alpha=1.0 produced near-identical outputs (mean diff={diff:.4f})"
    print(f"  ✅ mean predict_proba diff={diff:.4f} (soft vs hard labels diverge as expected)"); return {"mean_diff":float(diff)}
def test_20():
    sep("TEST 20: Soft-Label Cache Reuse")
    from tabtune.distillation.strategies import cross_validate_soft_labels
    X,y=make_classification(n_samples=400 if QUICK else 800,n_features=10,n_classes=2,random_state=42)
    Xdf,yser=pd.DataFrame(X),pd.Series(y)
    s1=cross_validate_soft_labels("TabPFN",Xdf,yser,n_folds=3,random_state=42)
    s2=cross_validate_soft_labels("TabPFN",Xdf,yser,n_folds=3,random_state=42)
    assert np.allclose(s1,s2,atol=1e-6), "Soft labels differ across two identical calls — caching/determinism broken"
    print(f"  ✅ Soft labels identical across two calls (shape={s1.shape})"); return {"ok":True}
def test_21():
    sep("TEST 21: Temperature Effect (T=1.0 vs T=8.0)")
    Xtr,Xte,ytr,yte=mc(1000,10,2)
    d_low=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",temperature=1.0,adaptive_temperature=False,student_params={"epochs":EP()//2})
    d_high=TabDistiller(teachers="TabPFN",student="mlp",task_type="classification",temperature=8.0,adaptive_temperature=False,student_params={"epochs":EP()//2})
    d_low.fit(Xtr,ytr); d_high.fit(Xtr,ytr)
    p_low=d_low.student_.predict_proba(Xte); p_high=d_high.student_.predict_proba(Xte)
    diff=np.abs(p_low-p_high).mean()
    assert diff>0.01, f"T=1.0 and T=8.0 produced near-identical outputs (mean diff={diff:.4f})"
    print(f"  ✅ mean predict_proba diff={diff:.4f} (low vs high T diverge as expected)"); return {"mean_diff":float(diff)}

# ── RUNNER ────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--test",type=int,help="Run a single test by number")
    pa.add_argument("--quick",action="store_true",help="Quick mode (smaller data, fewer epochs)")
    a=pa.parse_args()
    if a.quick: QUICK=True; print("  [QUICK MODE]\n")
    tests={i:globals()[f"test_{i}"] for i in range(1,22)}
    if a.test:
        if a.test not in tests: print(f"  ❌ No test {a.test}"); sys.exit(1)
        tests[a.test](); sys.exit(0)
    sep("TabTune Distillation v4 — Full Suite (21 tests)")
    ok={}; failed=[]
    for n,fn in tests.items():
        try: ok[n]=fn(); print(f"  ✅ Test {n} passed")
        except Exception as e: print(f"  ❌ Test {n} FAILED: {e}"); import traceback; traceback.print_exc(); failed.append(n)
    sep("SUMMARY")
    print(f"  Passed: {len(ok)}/21")
    if failed: print(f"  Failed: {failed}")
    print()
