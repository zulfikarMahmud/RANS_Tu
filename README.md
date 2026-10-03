# Transitional RANS cases for the Eppler E387 at Re = 200,000

**Author: Zulfikar** — <!-- GitHub: https://github.com/<your-handle>/<repo> -->

Five OpenFOAM 13 cases, one per turbulence model, set up identically so that
model-to-model differences are not confounded with numerics. Every case here
carries exactly the configuration used to produce the published sweep
(8 freestream turbulence levels x 4 angles of attack x 4 models).

| directory | turbulence model | library | freestream turbulence |
|---|---|---|---|
| `sst/`  | `kOmegaSST` (SST-2003, `c1 = 5`) | built-in | sustained via `fvModels` |
| `lm/`   | `kOmegaSSTLM` (gamma-Re-theta) | built-in | sustained via `fvModels` |
| `ysst/` | `gammaSST` (Menter 2015) | `libgammaSST.so` | sustained via `fvModels` |
| `kkos/` | `kkLOmegaS` (k-kL-omega, sustaining fork) | `libkkLOmegaS.so` | sustained via `fvModels` |
| `kko/`  | `kkLOmega` (**stock**) | built-in | **cannot sustain** — see below |

Baseline condition in every case: **Tu = 0.1 %, alpha = 0 deg, Uinf = 3 m/s,
nu = 1.5e-5 m^2/s** (Re = 200,000 at c = 1 m).

---

## Quick start

```bash
cd sst                      # or lm, ysst, kkos, kko
chmod +x Allrun Allclean    # only needed once, after cloning
./Allrun                    # ~10,000 iterations, 8 cores
python3 checkConv.py        # converged?
python3 make2k.py           # build the last-2000-iteration mean
```

If `checkConv.py` says **NOT CONVERGED**, continue the same case (never restart
from zero) and check again:

```bash
foamDictionary system/controlDict -entry startFrom -set latestTime
foamDictionary system/controlDict -entry endTime   -set 15000     # current + 5000
mpirun -np 8 foamRun -parallel
reconstructPar -newTimes
python3 checkConv.py
```

Repeat in +5000 steps until it passes, **then** run `make2k.py`.
`fieldAverage` is already left on by `Allrun`, so continuations keep averaging
automatically — you do not need to switch anything on.

---

## Before you run: three things to check on YOUR machine

**1. Core count.** The cases are decomposed 8 ways. If you do not have 8 cores
available, change it in **two places that must agree**:

```
system/decomposeParDict     numberOfSubdomains 8;
Allrun                      NP=8
```

Both must be the same number, or `Allrun` will fail at `decomposePar`.

**2. Patch names.** The mesh in `constant/polyMesh/boundary` uses:

```
airfoil    inlet    outlet    sides
```

If you substitute your own mesh, these names appear in `0/U`, `0/p`, every
turbulence field, and `system/sampleMeans` (which samples `patches (airfoil)`).
Rename consistently or nothing will run.

**3. Custom libraries.** `ysst/` and `kkos/` need out-of-tree models compiled
first — `libgammaSST.so` and `libkkLOmegaS.so`, loaded via the `libs` entry in
`system/controlDict`. `sst/`, `lm/` and `kko/` use only built-in models.

---

## The scripts

### `Allrun` — three-phase fresh start

```bash
chmod +x Allrun
./Allrun
```

| phase | iterations | schemes | averaging |
|---|---|---|---|
| 0 | 0 → 1000 | robust startup | off |
| 1 | 1000 → 5000 | production | off |
| 2 | 5000 → 10000 | production | **on** |

Phase 0 exists because a cold uniform start with second-order schemes drives
the turbulence variables negative, which OpenFOAM clips to zero, collapsing
`nut` and diverging the run. Phase 0 uses safer schemes for the first 1000
iterations only, and is discarded long before any reported result. In `sst/`
(and in two kkLOmega cases of the published sweep) phase 0 additionally uses
first-order `upwind` for the turbulence convection terms — unconditionally
bounded, so it cannot create the negative excursion. **This affects no reported
data**: every quoted result comes from second-order schemes in phases 1–2.

`Allrun` must be run on a clean case — `decomposePar` has no `-force` here.

### `Allclean` — reset to a fresh state

```bash
chmod +x Allclean
./Allclean
```

Removes `processor*`, time directories, `postProcessing` and logs. Run it
before re-running `Allrun` on a case that has already been run.

### `checkConv.py` — has it converged?

```bash
python3 checkConv.py                 # defaults: tol 0.003, rows every 1000
python3 checkConv.py --tol 0.002     # stricter
python3 checkConv.py --every 500     # finer history
```

Prints Cl, Cd and the convergence measure every 1000 iterations so you can watch
it settle, then a verdict.

**The criterion.** Take the trailing 2000 iterations, split into four blocks of
500, take the **mean** of each block, and require

```
max(block means) - min(block means)  <=  0.003
```

Two deliberate choices behind it:

- **Forces, not residuals.** The laminar separation bubble on this airfoil is
  genuinely unsteady, so momentum residuals plateau (typically `Uy` around
  1e-7 to 1e-5) and stop falling. A fully converged case can sit above a 1e-6
  `residualControl` threshold forever, while a case whose forces are still
  drifting can dip below it and stop early. Both were observed in the sweep.
- **Block means, not raw amplitude.** The bubble oscillates physically. Raw
  peak-to-peak measures that oscillation, not convergence. One case swung by
  0.0054 in raw Cl while its block means agreed to 9e-5 — converged, and a
  swing-based test would have rejected it.

Expect roughly 6,000–50,000 iterations depending on model and Tu. kkLOmega and
the high-Tu cases are the expensive ones.

### `make2k.py` — build the last-2000-iteration mean

```bash
python3 make2k.py                    # run after checkConv.py passes
python3 make2k.py --no-sample        # fields only, skip surface sampling
```

Writes:

```
<latestTime>/pMean2k, UMean2k, wallShearStressMean2k
system/sampleMeans2k                                    (generated)
postProcessing/sampleMeans2k/<t>/airfoilSurf.xy         x y z pMean2k wss2k(xyz)
postProcessing/mean2k/forceCoeffs2k.dat                 Cl, Cd, Cm — same window
postProcessing/mean2k/window.txt                        exact provenance
```

**Why this step exists.** `Allrun` averages over [5000, 10000] — a 5000-iteration
window — and a continuation averages over the whole continuation. The reporting
standard here is the **last 2000 iterations only**, because a longer window
reaches back into iterations that may still be settling and smears the result by
more than the differences between the models being compared.

**It is exact, not an approximation.** `fieldAverage` stores a *cumulative*
mean and records the window length as `totalTime`, so differencing two prefix
sums gives the interval mean exactly:

```
mean[t0,t1] = ( T1*Mean(t1) - T0*Mean(t0) ) / (T1 - T0)
```

Nothing is re-run. It always takes the final 2000 iterations wherever they fall,
so it does not matter how many continuations you did. It refuses to difference
across an averaging-round boundary rather than return a wrong number, and if the
final round is shorter than 2000 it uses the whole round and says so in
`window.txt`.

**Cp** = `pMean2k / (0.5 * Uinf^2)`, **Cf** from `wallShearStressMean2k`.

### `lsb.py` — separation bubble location

```bash
python3 lsb.py . 3
```

Reads the sampled surface data and reports suction-side reversed-flow regions
(separation and reattachment in x/c) at several noise thresholds.

### `backCalcTu.py` — **`kko/` only**

```bash
cd kko
python3 backCalcTu.py --target 0.1            # inlet Tu for 0.1 % at the LE
python3 backCalcTu.py --target 0.1 --apply    # write it into caseSettings
python3 backCalcTu.py --forward 0.1           # what does inlet 0.1 % give?
```

**Stock `kkLOmega` never calls `fvModels.source()`**, so the sustaining terms
used by the other four models are silently ignored and the freestream *decays*
over the ~19.6 chords from inlet to leading edge. The value you type is not the
value the airfoil sees.

`kko/` therefore ships with **`Tu = 0.1345 %` at the inlet**, which decays to
0.1009 % at x = −0.5c — matching the 0.1001 % the sustained models hold at
Tu = 0.1 %. **Do not "correct" this to 0.1**; that would put `kko` in a weaker
freestream than everything it is compared against.

The decay law is kkLOmega-specific (`Cw2 = 0.92`), not SST's (`beta = 0.0828`) —
about 11x faster. Using the SST constant badly under-predicts the decay.

---

## Reproducing a different condition

Edit **only** `0/include/caseSettings`:

```
Tu       0.1;     // freestream turbulence intensity [%]
alpha    0;       // angle of attack [deg]
```

Everything downstream (`k`, `omega`, `nut`, inlet velocity components, lift and
drag directions, the sustaining source strengths) is derived from these by
`#calc`. Then `./Allclean && ./Allrun`.

Note the eddy-viscosity ratio scales with Tu (`nut/nu = 10 * Tu / 0.1`), holding
the freestream turbulent *length scale* fixed. At Tu = 20 % that gives
`nut/nu = 2000`, far outside the 1–10 that Menter recommends for external
transition — deliberate, because the real atmosphere is genuinely that
turbulent, but worth knowing when comparing against low-Tu literature.

---

## Disclaimer

These cases are provided as a starting point, not as a turnkey guarantee.

- **Results vary with mesh.** The supplied mesh gives y+ of roughly 0.6 average
  and up to about 2.6 maximum. A different mesh — different resolution, wake
  refinement, far-field extent, or y+ — will shift the numbers, and the
  separation bubble is particularly mesh-sensitive.
- **Results vary with setup.** Domain size, boundary conditions, discretisation
  schemes, solver tolerances and OpenFOAM version all matter. These were tuned
  for this specific configuration.
- **Grid convergence was established at one condition only**, not across the
  whole sweep.
- **Iteration counts are indicative.** How long convergence takes depends on the
  machine, the mesh and the condition. Trust `checkConv.py`, not a fixed number.
- **Validate against your own reference data** before drawing conclusions.

## Notes

Claude Code was used during development to help debug known crash modes —
in particular the cold-start divergence traced to second-order gradient and
convection schemes driving turbulence variables negative on a uniform initial
field, and the resulting floating-point exceptions. The physics setup,
turbulence-model choices, sustaining formulation and validation are the
author's own.

Please cite the accompanying paper if you use these cases.
