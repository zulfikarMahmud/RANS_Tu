#!/usr/bin/env python3
"""
Back-calculate the INLET Tu that delivers a target Tu at the LEADING EDGE,
for the kkLOmega (k-kL-omega) case.

WHY THIS EXISTS
    kkLOmega ignores fvModels sources (stock kkLOmega.C never calls
    fvModels.source), so the sustaining trick used by the lm / sst / ysst cases
    is unavailable here.  The freestream therefore DECAYS over the ~19.6 chords
    between the inlet and the leading edge, and the Tu you type into
    0/include/caseSettings is NOT the Tu the airfoil sees.

    This script inverts that decay: give it the Tu you want AT THE LEADING EDGE
    and it returns the Tu to set at the inlet.

THE DECAY LAW (derived from OpenFOAM 13's kkLOmega.C, not from SST)
    In a uniform freestream: no shear, so Pkt = 0 and kl = 0; far from walls
    fw -> 1 (lambdaEff -> lambdaT); and Dt = nu*|grad(sqrt(kt))|^2 -> 0.
    The transport equations then reduce to

        d(omega)/dt = -Cw2 * omega^2          (kkLOmega.C, omegaEqn Sp term)
        d(kt)/dt    = -omega * kt             (kkLOmega.C, ktEqn Sp term)

    which integrate to

        omega(t) = omega0 / (1 + Cw2*omega0*t)
        kt(t)    = kt0 * (1 + Cw2*omega0*t)^(-1/Cw2)

    with Cw2 = 0.92 and t = L/Uinf.

    NOTE: this is NOT the SST law.  SST decays with beta = 0.0828; kkLOmega
    decays with Cw2 = 0.92, i.e. ~11x faster for the same omega0.  Using the
    SST constant here would badly under-predict the decay.

COUPLING
    nut/nu = ratioRef*Tu_in/TuRef is linear in Tu_in, so omega0 is linear in
    Tu_in while kt0 is quadratic.  Tu_LE(Tu_in) is therefore nonlinear and the
    inversion needs a root-find (bisection below) -- which is also why this
    cannot live in caseSettings as a #calc directive.

CAVEAT -- treat the answer as a FIRST GUESS
    The law above assumes 1-D homogeneous decay along a streamline with no
    production and no diffusion.  The real domain has mild pressure gradients,
    numerical diffusion and a finite mesh.  After running, read the ACTUAL
    leading-edge Tu from postProcessing/upstreamProbe (probe at x = -0.1) and,
    if it is off, re-run this script with --measured to correct.

USAGE
    python3 backCalcTu.py --target 0.1                 # inlet Tu for 0.1 % at LE
    python3 backCalcTu.py --forward 0.1                # what does inlet 0.1 % give at LE?
    python3 backCalcTu.py --target 0.1 --apply         # also write it into caseSettings
    python3 backCalcTu.py --target 0.1 --measured 0.085 --inlet-used 0.126
                                                       # correct using a probe reading
"""
import argparse
import os
import re
import sys

CW2 = 0.92          # kkLOmega omega-destruction coefficient (kkLOmega.C)
CS = "0/include/caseSettings"


def read_settings(path=CS):
    """Pull the USER INPUT block. Deliberately does not use foamDictionary -set,
    which strips the comments this file relies on."""
    if not os.path.isfile(path):
        sys.exit(f"[bc] {path} not found -- run from the case directory")
    txt = open(path).read()

    def g(key, default=None):
        m = re.search(rf'^\s*{key}\s+([-\d.eE+]+)\s*;', txt, re.M)
        if m:
            return float(m.group(1))
        if default is None:
            sys.exit(f"[bc] could not read '{key}' from {path}")
        return default

    return dict(Tu=g("Tu"), Uinf=g("Uinf"), nu=g("nuValue"),
                ratioRef=g("ratioRef"), TuRef=g("TuRef")), txt


def Tu_LE(Tu_in, s, L):
    """Forward: inlet Tu [%] -> leading-edge Tu [%]."""
    if Tu_in <= 0:
        return 0.0
    t = L / s["Uinf"]
    nutRatio = s["ratioRef"] * Tu_in / s["TuRef"]
    kt0 = 1.5 * (s["Uinf"] * Tu_in / 100.0) ** 2
    w0 = kt0 / (nutRatio * s["nu"])
    ktLE = kt0 * (1.0 + CW2 * w0 * t) ** (-1.0 / CW2)
    return (2.0 / 3.0 * ktLE) ** 0.5 / s["Uinf"] * 100.0


def invert(target, s, L):
    """Bisection on a monotonic increasing Tu_LE(Tu_in)."""
    lo, hi = 1e-6, 1.0
    while Tu_LE(hi, s, L) < target:
        hi *= 2.0
        if hi > 1e4:
            sys.exit("[bc] target unreachable -- decay too strong at any inlet Tu")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Tu_LE(mid, s, L) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


ap = argparse.ArgumentParser()
ap.add_argument("--target", type=float, help="desired Tu [%%] at the leading edge")
ap.add_argument("--forward", type=float, help="report LE Tu for this inlet Tu [%%]")
ap.add_argument("--L", type=float, default=19.56, help="inlet->LE distance [m] (default 19.56)")
ap.add_argument("--apply", action="store_true", help="write the result into caseSettings")
ap.add_argument("--measured", type=float, help="LE Tu [%%] actually measured by upstreamProbe")
ap.add_argument("--inlet-used", type=float, help="inlet Tu [%%] that produced --measured")
a = ap.parse_args()

s, txt = read_settings()
L = a.L
print(f"[bc] Uinf={s['Uinf']}  nu={s['nu']}  ratioRef={s['ratioRef']}  TuRef={s['TuRef']}")
print(f"[bc] inlet->LE L={L} m  ->  t={L/s['Uinf']:.2f} s   (Cw2={CW2}, exponent {-1/CW2:.4f})")
print(f"[bc] current caseSettings Tu = {s['Tu']} %  ->  LE Tu = {Tu_LE(s['Tu'], s, L):.4f} % "
      f"({Tu_LE(s['Tu'], s, L)/s['Tu']*100:.1f} % retained)")

if a.forward is not None:
    print(f"\n[bc] FORWARD: inlet {a.forward} %  ->  LE {Tu_LE(a.forward, s, L):.4f} %")

if a.measured is not None:
    if a.inlet_used is None:
        sys.exit("[bc] --measured also needs --inlet-used")
    pred = Tu_LE(a.inlet_used, s, L)
    print(f"\n[bc] CALIBRATION against the probe:")
    print(f"     inlet {a.inlet_used} %  ->  predicted LE {pred:.4f} % , measured {a.measured} %")
    print(f"     model error {(pred - a.measured)/a.measured*100:+.1f} % "
          f"-- apply this as a correction factor to the result below")

if a.target is not None:
    Tin = invert(a.target, s, L)
    chk = Tu_LE(Tin, s, L)
    nutRatio = s["ratioRef"] * Tin / s["TuRef"]
    print(f"\n[bc] INVERSE: to obtain {a.target} % at the LE, set")
    print(f"     Tu = {Tin:.4f};        (in {CS})")
    print(f"     -> forward check gives LE Tu = {chk:.4f} %")
    print(f"     -> implied inlet nut/nu = {nutRatio:.1f}", end="")
    print("   *** far outside Menter's recommended 1-10 ***" if nutRatio > 50 else "")
    if a.apply:
        new = re.sub(r'^(\s*Tu\s+)[-\d.eE+]+\s*;', rf'\g<1>{Tin:.4f};', txt, count=1, flags=re.M)
        if new == txt:
            sys.exit("[bc] failed to substitute Tu -- caseSettings not modified")
        open(CS, "w").write(new)
        print(f"\n[bc] WROTE Tu = {Tin:.4f} into {CS} (comments preserved)")
    else:
        print(f"\n[bc] (not applied -- rerun with --apply to write it)")
