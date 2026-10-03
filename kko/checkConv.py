#!/usr/bin/env python3
"""
checkConv.py -- convergence check for the E387 Tu-sweep cases.

    Run it from inside a case directory, at any time, even while the solver
    is still running:

        python3 checkConv.py

WHAT IT MEASURES
    Convergence is judged on the STATIONARITY OF THE INTEGRATED FORCES, not on
    solver residuals.  For this flow that distinction is not cosmetic:

        - The E387 laminar separation bubble is genuinely unsteady, so the
          momentum residuals plateau (typically Uy ~ 1e-7 to 1e-5) and simply
          stop falling.  A converged case can sit forever above a 1e-6
          residualControl threshold, while a case whose forces are still
          drifting can dip below it and stop early.  Both were observed.
        - Cl is what the reported Cp / Cd / Cl / Cm all depend on, so testing
          Cl directly tests the thing that matters.

    THE TEST.  Take the trailing 2000 iterations, split them into four blocks
    of 500, take the MEAN of each block, and require

        max(block means) - min(block means)  <=  0.003          [default tol]

    Using block MEANS rather than raw peak-to-peak is deliberate: the bubble
    oscillates physically, so raw amplitude measures the physics, not the
    numerics.  A case can oscillate by 0.005 and still be perfectly converged
    if its block means agree.

    Why 2000 and not a longer window: a 5000-iteration window reports the
    WIDTH OF THE DECAY rather than the end state, and lets ~0.010 of drift
    hide inside the very window you then average over -- which is larger than
    the differences between the turbulence models being compared.

HOW TO READ THE OUTPUT
    One row per 1000 iterations, so you can watch the spread fall and see
    exactly where it crossed the tolerance.  "n" is how many force samples
    the trailing-2000 window actually held (2000 when the run is dense).

IF IT SAYS NOT CONVERGED
    Continue the SAME case (never restart from zero):

        foamDictionary system/controlDict -entry startFrom -set latestTime
        foamDictionary system/controlDict -entry endTime   -set <current+5000>
        foamDictionary system/controlDict -entry functions/fieldAverage/enabled -set true
        mpirun -np 8 foamRun -parallel
        reconstructPar -newTimes

    Repeat in +5000 steps until this script reports CONVERGED.  Cases in the
    published sweep needed anywhere from ~6000 to 50000 iterations; the high-Tu
    and kkLOmega cases are the expensive ones.

    IMPORTANT: quote results from the LAST 2000 ITERATIONS only, matching the
    window this script reports.  Do not average further back -- earlier
    iterations may still be settling and will contaminate the mean.

OPTIONS
    --tol   0.003     convergence tolerance on the block-mean spread
    --every 1000      row spacing in the history table
    --case  .         case directory
"""
import argparse
import os
import sys

DEF_TOL = 0.003
WINDOW = 2000          # trailing iterations used for the test
NBLOCK = 4             # sub-blocks (4 x 500 = 2000)


def read_forces(case):
    """merge every postProcessing/forceCoeffs chunk -> {iter: (Cm, Cd, Cl)}"""
    d = os.path.join(case, "postProcessing", "forceCoeffs")
    if not os.path.isdir(d):
        return {}
    out = {}
    for t in sorted(os.listdir(d), key=lambda x: float(x)):
        f = os.path.join(d, t, "forceCoeffs.dat")
        if not os.path.exists(f):
            continue
        for ln in open(f, errors="ignore"):
            if ln.lstrip().startswith("#"):
                continue
            v = ln.split()
            if len(v) < 4:
                continue
            try:
                out[int(float(v[0]))] = (float(v[1]), float(v[2]), float(v[3]))
            except ValueError:
                continue
    # discard blow-up garbage so a diverged tail cannot masquerade as data
    return {k: v for k, v in out.items() if abs(v[2]) < 50.0}


def spread_at(o, end):
    """block-mean spread of Cl over (end-WINDOW, end]; None if not enough data"""
    step = WINDOW // NBLOCK
    means, n = [], 0
    for i in range(NBLOCK):
        lo, hi = end - WINDOW + i * step, end - WINDOW + (i + 1) * step
        w = [v[2] for k, v in o.items() if lo < k <= hi]
        if not w:
            return None, 0
        means.append(sum(w) / len(w))
        n += len(w)
    return max(means) - min(means), n


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--tol", type=float, default=DEF_TOL)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--case", default=".")
    a = ap.parse_args()

    o = read_forces(a.case)
    if not o:
        print("no force data found -- expected postProcessing/forceCoeffs/<t>/forceCoeffs.dat")
        sys.exit(1)

    end = max(o)
    name = os.path.basename(os.path.abspath(a.case))
    print(f"case: {name}")
    print(f"iterations logged: {min(o)} .. {end}")
    print(f"criterion: spread of {NBLOCK} x {WINDOW // NBLOCK}-iteration block means "
          f"over the trailing {WINDOW}  <=  {a.tol}\n")

    print(f"{'iter':>8} {'Cl':>9} {'Cd':>9} {'spread':>9} {'n':>6}  status")
    print("-" * 56)
    first_ok = None
    marks = [t for t in range(a.every, end + 1, a.every)]
    if marks and marks[-1] != end:
        marks.append(end)
    for t in marks:
        if t < WINDOW:
            continue
        sp, n = spread_at(o, t)
        if sp is None:
            continue
        near = min(o, key=lambda k: abs(k - t))
        cm, cd, cl = o[near]
        ok = sp <= a.tol
        # first crossing that HOLDS to the end
        if ok and first_ok is None:
            if all((spread_at(o, u)[0] or 9) <= a.tol
                   for u in marks if u >= t and spread_at(o, u)[0] is not None):
                first_ok = t
        print(f"{t:>8} {cl:>9.4f} {cd:>9.5f} {sp:>9.5f} {n:>6}  "
              f"{'ok' if ok else 'drifting'}")

    sp, n = spread_at(o, end)
    print("-" * 56)
    if sp is None:
        print(f"NOT ENOUGH DATA: need at least {WINDOW} iterations of force history.")
        sys.exit(2)

    rel = sp / abs(o[end][2]) * 100 if o[end][2] else float("nan")
    if sp <= a.tol:
        print(f"CONVERGED   spread {sp:.5f}  ({rel:.3f}% of Cl)  tol {a.tol}")
        if first_ok:
            print(f"            first met at ~{first_ok} and held to {end}")
        print(f"\nquote results from the last {WINDOW} iterations only:")
        print(f"    window = ({end - WINDOW}, {end}]")
        cm, cd, cl = (sum(v[i] for k, v in o.items() if k > end - WINDOW)
                      / max(1, len([1 for k in o if k > end - WINDOW])) for i in range(3))
        print(f"    Cl = {cl:.4f}   Cd = {cd:.5f}   Cm = {cm:.5f}")
    else:
        print(f"NOT CONVERGED   spread {sp:.5f}  ({rel:.3f}% of Cl)  tol {a.tol}")
        print(f"                continue this case +5000 iterations and re-check")
        print(f"                (see the header of this file for the exact commands)")
        sys.exit(3)


if __name__ == "__main__":
    main()
