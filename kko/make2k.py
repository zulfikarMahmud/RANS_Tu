#!/usr/bin/env python3
"""
make2k.py -- build the LAST-2000-ITERATION mean for this case.

    Run it from inside a case directory once checkConv.py reports CONVERGED:

        python3 make2k.py

    It works no matter how many continuations you did -- it always takes the
    final 2000 iterations, wherever they happen to fall.

WHY YOU NEED IT
    Allrun averages over [AVG_START, END] (5000 iterations by default) and a
    continuation averages over the whole continuation.  The reporting standard
    for this study is the LAST 2000 ITERATIONS ONLY: a longer window reaches
    back into iterations that may still have been settling, and smears the
    result by more than the differences between the turbulence models.

    This script does NOT re-run anything.  It derives the 2000-iteration mean
    from data already on disk.

WHY IT IS EXACT, NOT AN APPROXIMATION
    fieldAverage writes a CUMULATIVE mean: pMean(t) is the average from the
    start of the current averaging round up to t, and
    <t>/uniform/fieldAverageProperties records the window length as totalTime.
    With `base time` and deltaT 1 the accumulator is a plain sum over
    iterations, so subtracting two prefix sums gives the interval mean exactly:

        mean[t0,t1] = ( T1*Mean(t1) - T0*Mean(t0) ) / (T1 - T0)

    Both snapshots must belong to the SAME averaging round, because
    restartOnRestart zeroes the accumulator at every restart.  The script
    checks this and refuses rather than producing a wrong answer.

WHAT IT WRITES  (all new files; nothing existing is modified)
    <latestTime>/pMean2k                 -> Cp
    <latestTime>/UMean2k
    <latestTime>/wallShearStressMean2k   -> Cf, and lsb.py bubble location
    system/sampleMeans2k                 sampling dict for the above
    postProcessing/sampleMeans2k/<t>/airfoilSurf.xy
                                         x y z pMean2k wallShearStressMean2k(xyz)
    postProcessing/mean2k/forceCoeffs2k.dat   Cl, Cd, Cm over the SAME window
    postProcessing/mean2k/window.txt          exactly which window was used

    Cp = pMean2k / (0.5 * Uinf^2);  Uinf is in 0/include/caseSettings.

IF THE FINAL AVERAGING ROUND IS SHORTER THAN 2000
    That happens if the solver stopped part-way through a round (residualControl
    can do this).  The script then uses the WHOLE round and says so, loudly, in
    window.txt -- a short window is noisier and must not be quoted as if it were
    a full 2000.

OPTIONS
    --case .        case directory
    --window 2000   window length in iterations
    --no-sample     skip the surface sampling step (fields only)
"""
import argparse
import glob
import os
import re
import subprocess
import sys

NUM = re.compile(r"^-?\d+\.?\d*(?:[eE][-+]?\d+)?$")
VEC = re.compile(r"^\(\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*\)$")
UNI_S = re.compile(r"^(\s*\w+\s+uniform\s+)(-?[\d.eE+-]+)(\s*;\s*)$")
UNI_V = re.compile(r"^(\s*\w+\s+uniform\s+)\(([^)]*)\)(\s*;\s*)$")
FIELDS = ["pMean", "UMean", "wallShearStressMean"]


def times(case):
    return sorted(int(d) for d in os.listdir(case)
                  if re.fullmatch(r"\d+", d) and int(d) > 0)


def total_time(case, t):
    """length of the averaging round accumulated up to time t, or None"""
    f = os.path.join(case, str(t), "uniform", "fieldAverageProperties")
    if not os.path.exists(f):
        return None
    m = re.search(r"totalTime\s+([\d.eE+-]+)\s*;", open(f).read())
    return float(m.group(1)) if m else None


def combine(f0, f1, w0, w1, out):
    """out = (w1*f1 - w0*f0)/(w1-w0), line-wise; structure taken from f1"""
    a = open(f0).read().split("\n")
    b = open(f1).read().split("\n")
    if len(a) != len(b):
        return False
    den = w1 - w0
    res, in_list = [], False
    for la, lb in zip(a, b):
        s = lb.strip()
        if s == "(":
            in_list = True; res.append(lb); continue
        if s in (")", ");"):
            in_list = False; res.append(lb); continue
        if in_list:
            if NUM.match(s) and NUM.match(la.strip()):
                res.append(f"{(w1*float(s) - w0*float(la.strip()))/den:.8g}"); continue
            mv, ma = VEC.match(s), VEC.match(la.strip())
            if mv and ma:
                v = [(w1*float(mv.group(i)) - w0*float(ma.group(i)))/den for i in (1, 2, 3)]
                res.append("(" + " ".join(f"{x:.8g}" for x in v) + ")"); continue
            res.append(lb); continue
        m1, m0 = UNI_S.match(lb), UNI_S.match(la)
        if m1 and m0:
            res.append(m1.group(1)
                       + f"{(w1*float(m1.group(2)) - w0*float(m0.group(2)))/den:.8g}"
                       + m1.group(3)); continue
        m1, m0 = UNI_V.match(lb), UNI_V.match(la)
        if m1 and m0:
            v1 = [float(x) for x in m1.group(2).split()]
            v0 = [float(x) for x in m0.group(2).split()]
            if len(v1) == len(v0) == 3:
                v = [(w1*v1[i] - w0*v0[i])/den for i in range(3)]
                res.append(m1.group(1) + "(" + " ".join(f"{x:.8g}" for x in v) + ")"
                           + m1.group(3)); continue
        res.append(lb)
    txt = "\n".join(res)
    txt = re.sub(r'(\n\s*object\s+)(\w+)(\s*;)',
                 lambda m: m.group(1) + m.group(2) + "2k" + m.group(3), txt, count=1)
    open(out, "w").write(txt)
    return True


def copy_as_2k(src, out):
    """whole-round fallback: the stored mean already IS the window"""
    txt = open(src).read()
    txt = re.sub(r'(\n\s*object\s+)(\w+)(\s*;)',
                 lambda m: m.group(1) + m.group(2) + "2k" + m.group(3), txt, count=1)
    open(out, "w").write(txt)


def forces(case, t0, t1):
    d = os.path.join(case, "postProcessing", "forceCoeffs")
    rows = []
    if os.path.isdir(d):
        for t in sorted(os.listdir(d), key=float):
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
                    it = int(float(v[0]))
                    if t0 < it <= t1:
                        rows.append((float(v[1]), float(v[2]), float(v[3])))
                except ValueError:
                    pass
    if not rows:
        return None
    n = len(rows)
    return (sum(r[0] for r in rows)/n, sum(r[1] for r in rows)/n,
            sum(r[2] for r in rows)/n, n)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--case", default=".")
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--no-sample", action="store_true")
    a = ap.parse_args()
    c, W = a.case, a.window

    ts = times(c)
    if not ts:
        print("no time directories -- has this case run?"); sys.exit(1)
    t1 = ts[-1]
    T1 = total_time(c, t1)
    if T1 is None:
        print(f"no fieldAverageProperties in {t1}/uniform/ -- was fieldAverage enabled?")
        print("Allrun turns it on for phase 2; a continuation keeps it on.")
        sys.exit(1)
    if not os.path.exists(os.path.join(c, str(t1), "pMean")):
        print(f"no pMean in {t1}/ -- nothing to build from."); sys.exit(1)

    # latest snapshot at least W back, inside the SAME averaging round
    cand = [t for t in ts if t < t1
            and (total_time(c, t) is not None)
            and total_time(c, t) < T1
            and T1 - total_time(c, t) >= W]
    exact = bool(cand)

    if exact:
        t0 = max(cand); T0 = total_time(c, t0); win = T1 - T0
    else:
        t0 = t1 - int(round(T1)); T0 = 0.0; win = T1

    print(f"latest time      : {t1}")
    print(f"averaging round  : {T1:g} iterations long")
    if exact:
        print(f"differencing     : {t1} (T={T1:g})  minus  {t0} (T={T0:g})")
        print(f"WINDOW           : ({t0}, {t1}]   length {win:g}")
    else:
        print(f"WINDOW           : ({t0}, {t1}]   length {win:g}   <-- WHOLE ROUND")
        print(f"                   no snapshot {W} back inside this round, so the")
        print(f"                   full round is used." +
              (f"  SHORTER than {W} -- noisier." if win < W else ""))

    done = []
    for fld in FIELDS:
        f1 = os.path.join(c, str(t1), fld)
        if not os.path.exists(f1):
            continue
        out = f1 + "2k"
        if exact:
            f0 = os.path.join(c, str(t0), fld)
            if not os.path.exists(f0):
                continue
            if not combine(f0, f1, T0, T1, out):
                print(f"  !! {fld}: files differ in structure, skipped"); continue
        else:
            copy_as_2k(f1, out)
        done.append(fld + "2k")
    print(f"fields written   : {', '.join(done) if done else 'NONE'}  (into {t1}/)")

    od = os.path.join(c, "postProcessing", "mean2k")
    os.makedirs(od, exist_ok=True)
    fc = forces(c, t0, t1)
    with open(os.path.join(od, "window.txt"), "w") as fh:
        fh.write("# last-2000 mean provenance\n")
        fh.write(f"window = [{t0}, {t1}]   length = {win:g}\n")
        if exact:
            fh.write(f"t0 = {t0}  totalTime0 = {T0:g}\nt1 = {t1}  totalTime1 = {T1:g}\n")
            fh.write(f"formula: mean = ({T1:g}*Mean({t1}) - {T0:g}*Mean({t0})) / {win:g}\n")
        else:
            fh.write("source: FULL final averaging round (no snapshot far enough\n"
                     "        back inside the same round to difference against)\n")
            if win < W:
                fh.write(f"WARNING: window is only {win:g} iterations, shorter than the\n"
                         f"         {W} used elsewhere -- noisier, do not quote as full.\n")
        fh.write(f"fields written into {t1}/: {', '.join(done) if done else 'NONE'}\n")
        fh.write("forces below use the SAME window as the fields.\n")
    if fc:
        with open(os.path.join(od, "forceCoeffs2k.dat"), "w") as fh:
            fh.write(f"# Cl/Cd/Cm over ({t0}, {t1}]  length {win:g}  n={fc[3]} samples\n")
            fh.write("# same window as pMean2k etc\n# Cm Cd Cl\n")
            fh.write(f"{fc[0]:.8g} {fc[1]:.8g} {fc[2]:.8g}\n")
        print(f"forces           : Cl={fc[2]:.4f}  Cd={fc[1]:.5f}  Cm={fc[0]:.5f}  "
              f"(n={fc[3]})")

    if a.no_sample or not done:
        print("\nsampling skipped."); return
    sm = os.path.join(c, "system", "sampleMeans")
    if not os.path.exists(sm):
        print("\nno system/sampleMeans -- surface sampling skipped."); return
    txt = open(sm).read().replace("pMean wallShearStressMean",
                                  "pMean2k wallShearStressMean2k")
    open(os.path.join(c, "system", "sampleMeans2k"), "w").write(txt)
    r = subprocess.run(["postProcess", "-func", "sampleMeans2k", "-latestTime"],
                       cwd=c, capture_output=True, text=True)
    xy = os.path.join(c, "postProcessing", "sampleMeans2k", str(t1), "airfoilSurf.xy")
    if os.path.exists(xy):
        cols = 0
        for ln in open(xy):
            if not ln.startswith("#"):
                cols = len(ln.split()); break
        if cols >= 7:
            print(f"surface sampled  : postProcessing/sampleMeans2k/{t1}/airfoilSurf.xy")
            print("                   columns: x y z pMean2k wss2k(x y z)")
        else:
            print(f"  !! sampled file has only {cols} columns -- the 2k fields were "
                  f"not found by postProcess")
    else:
        print("  !! sampling produced no output:")
        print((r.stderr or r.stdout or "").strip()[-400:])

    print(f"\nquote everything from the window above: ({t0}, {t1}]")


if __name__ == "__main__":
    main()
