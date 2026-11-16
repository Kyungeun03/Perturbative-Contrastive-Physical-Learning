"""
Trial wrapper for Iris_elongation_classifier_v2_plot.py
========================================================
The only substitutions are:
    np.random.seed(42)   ->  np.random.seed(<trial seed>)
    random_state=42      ->  random_state=<trial seed>

Plotting is suppressed (Agg backend, plt.show disabled) so the runs do not
block; nothing in the training path is touched.

Run it from the same folder as Iris_elongation_classifier_v2_plot.py.
"""

import os
import re
import io
import sys
import contextlib
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

ORIGINAL   = "Iris_elongation_classifier_v2_plot.py"
N_TRIALS   = 25
RESULTS_CSV = "v2_trial_results.csv"

SRC = open(ORIGINAL).read()


def run_trial(seed, epochs=None, quiet=True):
    """epochs=None runs the script as written; epochs=0 is the untrained control."""
    src = SRC
    src = src.replace("np.random.seed(42)", f"np.random.seed({seed})")
    src = src.replace("random_state=42",    f"random_state={seed}")
    if epochs is not None:
        src = src.replace("epochs = 200", f"epochs = {epochs}")

    ns = {"__name__": "__main__"}
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(buf if quiet else sys.stdout):
            exec(compile(src, ORIGINAL, "exec"), ns)
        msgs = sorted({str(w.message) for w in caught})

    W, b, c = ns["W"], ns["b"], ns["c"]
    finite = bool(np.isfinite(W).all() and np.isfinite(b).all() and np.isfinite(c))

    return {
        "linear_acc": float(ns["linear_acc"]),
        "finite":     finite,
        "max_absW":   float(np.nanmax(np.abs(W))),
        "warnings":   msgs,
    }


if __name__ == "__main__":
    print(f"Trial wrapper for {ORIGINAL}")
    print(f"trials={N_TRIALS}   (seeds 42 + 13t)")
    print("-" * 62)

    new_file = not os.path.exists(RESULTS_CSV)
    accs, accs0, n_diverged = [], [], 0

    with open(RESULTS_CSV, "a", buffering=1) as fh:
        if new_file:
            fh.write("trial,seed,acc_untrained,acc_trained,finite,max_absW\n")

        for t in range(N_TRIALS):
            seed = 42 + 13 * t
            r0 = run_trial(seed, epochs=0)     # control: no training
            r  = run_trial(seed)               # as written
            accs0.append(r0["linear_acc"])
            accs.append(r["linear_acc"])
            if not r["finite"]:
                n_diverged += 1

            fh.write(f"{t},{seed},{r0['linear_acc']:.4f},{r['linear_acc']:.4f},"
                     f"{r['finite']},{r['max_absW']:.6g}\n")
            flag = "" if r["finite"] else "   <-- DIVERGED"
            print(f"  trial {t+1:>2}/{N_TRIALS}  seed={seed:<5d} "
                  f"untrained={r0['linear_acc']:.3f}  trained={r['linear_acc']:.3f}"
                  f"  max|W|={r['max_absW']:.3g}{flag}", flush=True)

    accs, accs0 = np.array(accs), np.array(accs0)
    print("\n" + "=" * 62)
    print(f"untrained (epochs=0)     : {accs0.mean():.1%} +- {accs0.std():.1%}")
    print(f"trained   (epochs=200)   : {accs.mean():.1%} +- {accs.std():.1%}")
    print(f"diverged trials          : {n_diverged}/{N_TRIALS}")