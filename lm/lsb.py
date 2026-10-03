#!/usr/bin/env python3
"""
lsb.py — locate the laminar separation bubble from the MEAN surface fields.

Reads the file written by  'postProcess -func sampleMeans -latestTime':
    <case>/postProcessing/sampleMeans/<time>/airfoilSurf.xy
    columns:  x  y  z   pMean   tauMean_x tauMean_y tauMean_z
(so it uses the TIME-AVERAGED wall shear / pressure -> correct for a
 limit-cycling or unsteady solution, not a snapshot.)

METHOD (the CFD standard): separation/reattachment are where the streamwise
wall shear (tangential Cf) changes sign. This script finds ALL sign changes,
not just the first two, so a bubble with internal structure (a secondary
vortex -> reattach then re-separate) is captured in full. A minimum-length
noise filter stops single-cell numerical wiggles being miscounted as vortices;
results are printed at three thresholds so you can see which crossings are
robust. Transition T is reported as the Cf-minimum location (see notes).

Usage:  python3 lsb.py <case_dir> [Uinf] [min_len_xc]
        python3 lsb.py kkoSteadyTu 3 0.005
"""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _num(s, default):  # tolerate trailing ';' etc. from foamDictionary output
    import re
    m = re.search(r'[-+0-9.eE]+', str(s))
    return float(m.group()) if m else default
case    = sys.argv[1] if len(sys.argv) > 1 else "."
Uinf    = _num(sys.argv[2], 3.0)   if len(sys.argv) > 2 else 3.0
MIN_LEN = _num(sys.argv[3], 0.005) if len(sys.argv) > 3 else 0.005   # min region length [x/c]
q = 0.5 * Uinf**2

# ---- locate newest sampleMeans file -----------------------------------------
pats = sorted(glob.glob(os.path.join(case, "postProcessing/sampleMeans/*/airfoilSurf.xy")),
              key=lambda f: float(f.split(os.sep)[-2]))
if not pats:
    sys.exit(f"[lsb] no sampleMeans file under {case}/postProcessing/sampleMeans/ "
             f"-- run the case (./Allrun) first.")
fpath = pats[-1]
d = np.loadtxt(fpath)
x, y, pMean, tau = d[:, 0], d[:, 1], d[:, 3], d[:, 4:7]

# ---- order points around the contour, split upper/lower at the LE -----------
# (E387 is cambered: a 'is y above the chord line?' split mixes surfaces, so we
#  walk the closed loop by nearest-neighbour instead.)
def walk(px, py):
    S = np.column_stack([px, py]); n = len(S)
    order = [int(np.argmax(px))]; un = set(range(n)); un.discard(order[0])
    while un:
        cur = S[order[-1]]; idx = np.fromiter(un, int)
        j = idx[np.argmin(((S[idx]-cur)**2).sum(1))]; order.append(int(j)); un.discard(int(j))
    return np.array(order)

o = walk(x, y)
xs, ys, ps, ts = x[o], y[o], pMean[o], tau[o]
iLE = int(np.argmin(xs))
A, B = slice(0, iLE+1), slice(iLE, len(xs))
up = A if ys[A].mean() > ys[B].mean() else B         # suction side = higher mean y
xu, yu, pu, tu = xs[up], ys[up], ps[up], ts[up]
s = np.argsort(xu); xu, yu, pu, tu = xu[s], yu[s], pu[s], tu[s]
chord = x.max() - x.min()
xc = (xu - x.min()) / chord

# ---- tangential (streamwise) wall shear -> Cf -------------------------------
tx, ty = np.gradient(xu), np.gradient(yu)            # local LE->TE surface tangent
n = np.hypot(tx, ty); tx, ty = tx/n, ty/n
Cf = (tu[:, 0]*tx + tu[:, 1]*ty) / q                 # sign = flow direction at wall
Cp = pu / q

# OpenFOAM's wallShearStress sign convention is not fixed, so decide which sign
# means ATTACHED empirically: just aft of the LE (x/c 0.05-0.15) the suction side
# is definitely attached forward flow. Reversed flow = the opposite sign.
att_mask = (xc > 0.05) & (xc < 0.15)
att_sign = np.sign(np.median(Cf[att_mask])) if att_mask.sum() >= 3 else np.sign(np.median(Cf))
if att_sign == 0:
    att_sign = 1.0
revstr = -att_sign * Cf                              # > 0 where flow is reversed

# ---- reversed-flow segments (Cf<0), length-filtered -------------------------
def segments(mask, xc, min_len):
    segs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            x0, x1 = xc[i], xc[j-1]
            if (x1 - x0) >= min_len:
                segs.append((x0, x1))
            i = j
        else:
            i += 1
    return segs

rev = revstr > 0.0

# robustness: how many bubbles survive at 3 thresholds
print(f"\n[lsb] file: {fpath}")
print(f"[lsb] (attached-flow Cf sign = {'-' if att_sign<0 else '+'}; reversed = opposite)")
print(f"[lsb] suction-side reversed-flow regions vs noise threshold:")
for ml in (MIN_LEN/2, MIN_LEN, MIN_LEN*2):
    segs = segments(rev, xc, ml)
    print(f"      min_len={ml:.4f} c  ->  {len(segs)} region(s): "
          + ", ".join(f"[{a:.3f}-{b:.3f}]" for a, b in segs))

segs = segments(rev, xc, MIN_LEN)
if not segs:
    print("\n[lsb] no significant reversed-flow region -> attached (no bubble) at this threshold.")
    LS = TR = Ttr = None
else:
    LS = segs[0][0]                    # first separation
    TR = segs[-1][1]                   # final reattachment (spans all internal structure)
    inb = (xc >= LS) & (xc <= TR)
    Ttr = xc[inb][np.argmax(revstr[inb])]  # transition proxy = strongest reversed flow
    print(f"\n[lsb] ===== LAMINAR SEPARATION BUBBLE (min_len={MIN_LEN:.4f} c) =====")
    print(f"      LS  (separation)     x/c = {LS:.4f}")
    print(f"      T   (transition,Cfmin) x/c = {Ttr:.4f}")
    print(f"      TR  (reattachment)   x/c = {TR:.4f}")
    print(f"      bubble length (LS->TR)   = {TR-LS:.4f} c")
    if len(segs) > 1:
        print(f"      *** {len(segs)} reversed-flow cells -> SECONDARY VORTEX structure:")
        for k, (a, b) in enumerate(segs, 1):
            print(f"          vortex {k}: x/c [{a:.4f} - {b:.4f}]  (len {b-a:.4f})")
        gaps = [(segs[i][1], segs[i+1][0]) for i in range(len(segs)-1)]
        for k, (a, b) in enumerate(gaps, 1):
            print(f"          internal reattach->reseparate: x/c [{a:.4f} - {b:.4f}]")
        print(f"      (bubble length above spans ALL of it, LS to final TR.)")
    print(f"      suction peak Cp = {Cp.min():.3f} at x/c = {xc[np.argmin(Cp)]:.3f}")
    print("\n[lsb] note: T is the Cf-minimum proxy. For the rigorous transition point")
    print("      sample kt (kkLOmega) or gammaInt (LM) near the wall and use its onset.")

# ---- plot: Cf and Cp with every marker --------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
a1.axhline(0, color="k", lw=.7)
a1.plot(xc, Cf, "-", color="#c0392b", lw=1.4)
for (a, b) in segs:
    a1.axvspan(a, b, color="#f1c40f", alpha=.30)
if LS is not None:
    a1.axvline(LS, color="#e67e22", ls="--", lw=1); a1.text(LS, a1.get_ylim()[1]*.6, " LS", color="#e67e22")
    a1.axvline(TR, color="#27ae60", ls="--", lw=1); a1.text(TR, a1.get_ylim()[1]*.6, " TR", color="#27ae60")
    a1.axvline(Ttr, color="#8e44ad", ls=":",  lw=1); a1.text(Ttr, a1.get_ylim()[0]*.7, " T", color="#8e44ad")
a1.set_ylabel("mean Cf (suction)"); a1.set_title(f"LSB — {os.path.basename(os.path.abspath(case))}"); a1.grid(alpha=.3)
a2.plot(xc, Cp, "-", color="#2471a3", lw=1.4)
for (a, b) in segs:
    a2.axvspan(a, b, color="#f1c40f", alpha=.30)
a2.invert_yaxis(); a2.set_xlabel("x/c"); a2.set_ylabel("mean Cp (suction)"); a2.grid(alpha=.3)
out = os.path.join(case, "lsb.png")
fig.tight_layout(); fig.savefig(out, dpi=110)
print(f"\n[lsb] plot -> {out}\n")
