# Fig. 8: Isotropic NFW benchmark against other codes

Source: https://pianistpy.com/docs/notebooks/chiang2026d/fig08

This notebook makes Figure 8 of the paper by the package’s public calls of ver. 1.0.1: the package’s DF by `pp.compute_df`, the fitting functions of [Widrow (2000)](https://doi.org/10.1086/317367) and [Lane et al. (2022)](https://doi.org/10.1093/mnras/stab3755) made DFs of the same component by `pp.DistributionFunction.from_function`, the package’s DF read by `DistributionFunction.evaluate`, and the round trip of each of the three DFs by `pp.round_trip`. galpy and AGAMA run from the arXiv figure’s own scripts (`scripts/galpy_iso.py`, `scripts/agama_iso.py`), unchanged apart from their file names. The figure cell is the arXiv figure’s script, its changed lines marked `[1.0.1]`.

Top panel: \(\log_{10}|F_x / F_{\rm PIANISTpy} - 1|\) at \(\mathcal{Z} = P_{\rm NFW}(R)\) for Widrow, Lane et al. (each fit at its closed form), galpy and AGAMA. Bottom panel: \(\log_{10}|\varrho / \varrho_{\rm NFW} - 1|\) of the round trip (the DF integrated to a density and Poisson’s equation solved with \(P(0) = 1\)) for PIANISTpy, Widrow, Lane et al., and AGAMA’s own density.

Per-call accuracy messages are quieted; the summary lists the model’s tier.

```ipython3
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # before any JAX import: no C++ log lines from XLA

from importlib import metadata

PINS = {"pianistpy": "1.0.1", "numpy": "2.4.3", "scipy": "1.17.1", "galpy": "1.11.0", "agama": "1.0.156"}

def installed(name):
    if name == "pianistpy":  # the module's own version, installed or on the path
        import pianistpy

        return pianistpy.__version__
    if name == "agama":  # its module states the version; its package metadata says 1.0
        try:
            import agama
        except ImportError:
            return None
        return agama.__version__.split()[0]
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

found = {name: installed(name) for name in PINS}
for name, pin in PINS.items():
    print(f"{name}: pinned {pin}, installed {found[name]}")

def ready(code):
    """Whether an external code runs: it, numpy and scipy at their pins."""
    off = [f"{n} {found[n] or 'missing'} (pinned {PINS[n]})" for n in (code, "numpy", "scipy") if found[n] != PINS[n]]
    if off:
        print(f"{code} is not run: " + ", ".join(off) + ". Its curve is not drawn.")
    return not off
```

```myst
pianistpy: pinned 1.0.1, installed 1.0.1
numpy: pinned 2.4.3, installed 2.4.3
scipy: pinned 1.17.1, installed 1.17.1
galpy: pinned 1.11.0, installed 1.11.0
agama: pinned 1.0.156, installed 1.0.156
```

```ipython3
import logging
import math
import warnings

import pianistpy as pp

# The figures of the paper are made with ver. 1.0.1 of the package; any other installed version stops the notebook here.
PIANISTPY_PIN = "1.0.1"
print(f"pianistpy {pp.__version__}")
if pp.__version__ != PIANISTPY_PIN:
    raise RuntimeError(
        f"These figures are made with pianistpy {PIANISTPY_PIN}; the installed version is {pp.__version__}."
    )

# The package reports each result's accuracy twice per call: a line of the pianistpy log and Python's AccuracyWarning.
# set_verbosity("quiet") hides the log's INFO line of a result in the band "recorded"; the pianistpy logger set to
# ERROR also hides the log's WARNING line of a result in the band "warned"; the filter hides the AccuracyWarning.
# The summary after each figure's models prints every model's tier in place of those per-call messages.
pp.set_verbosity("quiet")
logging.getLogger("pianistpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=pp.AccuracyWarning)

def tier_line(label, result):
    """One summary line for a result: its label, its report's tier, and the report's worst check with that check's error.

    The worst check is the check of the highest tier level; among checks of one level, the check with the largest error.
    A result whose tier is carried from the Method I model's report, and not from any of the result's own checks, says so.
    """
    if result is None:
        return f"{label:34s} not built: the package's exception is printed above"
    report = result.validation

    def rank(key):
        error = report.checks[key].error
        return (report.checks[key].level, error if not math.isnan(error) else -math.inf)

    worst = max(report.keys, key=rank)
    own_level = report.checks[worst].level
    origin = ", the tier carried from the Method I model's report" if report.parents_level > own_level else ""
    return (f"{label:34s} tier {report.tier:9s} worst check {worst} "
            f"{report.checks[worst].error:.2e}{origin}")
```

```myst
pianistpy 1.0.1
```

```ipython3
import numpy as np
import pianistpy as pp

df = pp.compute_df(pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy()))
print("Figure 8: the model")
print(tier_line("isotropic NFW, untruncated", df))
print(f"P_0 = {float(df.form.p_0)!r}; table K from {float(df.table.k[0][0]):.3e} to P_0 - {float(df.form.p_0) - float(df.table.k[0][-1]):.6e}, "
      f"{len(df.table.k[0])} nodes")
```

```myst
Figure 8: the model
isotropic NFW, untruncated         tier silent    worst check high_k_slope_mismatch 6.45e-07
P_0 = 1.0; table K from 2.007e-10 to P_0 - 1.001461e-09, 9655 nodes
```

```ipython3
import subprocess
import sys
import time

# galpy and AGAMA by the scripts of the arXiv figure, unchanged: each writes its table at its own radii
# (galpy 1000 radii 10^-9.9 to 10^9.9, AGAMA 10000 radii 10^-9 to 10^10) into this folder
for code_, script in (("galpy", "scripts/galpy_iso.py"), ("agama", "scripts/agama_iso.py")):
    if not ready(code_):
        continue
    t0 = time.perf_counter()
    r_ = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(f"{code_}: exit {r_.returncode} in {time.perf_counter() - t0:.0f} s; " + r_.stdout.strip().splitlines()[-1])
    assert r_.returncode == 0, r_.stderr[-2000:]
```

```myst
galpy: exit 0 in 118 s; [Success] Execution complete. Formatted matrix written to: galpy_numerical_iso.txt
agama: exit 0 in 14 s; [Success] Fully self-consistent numerical AGAMA profile dataset written to: agama_numerical_iso.txt
```

## The curves

About 40 minutes on one 32-core node, almost all of it the three round trips.

```ipython3
import time

import numpy as np

OUT = "fig08_curves.npz"   # the curves of this run, read by the figure cell below
T0 = time.perf_counter()
assert float(df.form.p_0) == 1.0, df.form.p_0   # internal units: P(0) = 1

def _ln_k(k, depth):
    # ln K, carried by the depth 1 - K near the top, where K alone has lost the depth's digits
    return np.where(depth < 0.5, np.log1p(-depth), np.log(k))

def widrow(k, depth):
    # Widrow (2000), the fit of the isotropic NFW DF, as a function of K and the depth 1 - K
    q, p1, p2, p3, p4, F0 = -2.7419, 0.3620, -0.5639, -0.0859, -0.4912, 0.091968
    lg = (np.log10(F0) + 1.5 * np.log10(k) - 2.5 * np.log10(depth) + q * np.log10(-_ln_k(k, depth) / depth)
          + (p1 * k + p2 * k ** 2 + p3 * k ** 3 + p4 * k ** 4) / np.log(10.0))
    return 10.0 ** lg

LANE = (0.0926081086527954, 0.0232272797489981, 0.258346829924101, -2.97848277491979, 14.9776586391246, -46.6587221550258,
        92.6397009471828, -117.647787290798, 92.5144063082258, -41.0268009529576, 7.84806318891231)

def lane(k, depth):
    # Lane et al. (2022), the fit of the isotropic NFW DF, as a function of K and the depth 1 - K
    lg = (1.5 * np.log10(k) - 2.5 * np.log10(depth) - 2.75 * np.log10(-_ln_k(k, depth) / depth)
          + np.log10(np.polyval(LANE[::-1], k)))
    return 10.0 ** lg

# the three DFs of one component, the isotropic NFW profile: the package's, and the two fits tabulated by the package
nfw = df.components[0]
dfs = {"pianist": df,
       "widrow": pp.DistributionFunction.from_function(nfw, widrow),
       "lane": pp.DistributionFunction.from_function(nfw, lane)}
print(tier_line("Widrow (2000), from_function", dfs["widrow"]))
print(tier_line("Lane et al. (2022), from_function", dfs["lane"]))

LOGR_OUT = np.round(np.arange(-10.0, 9.0 + 1e-9, 0.005), 6)
R_OUT = 10.0 ** LOGR_OUT

def u_of_R(R):
    # the depth 1 - P_NFW(R) of the analytic NFW potential, by its series inside R = 0.1
    R = np.asarray(R, dtype=np.float64)
    out = np.empty(R.shape)
    s = R < 0.1
    k = np.arange(1, 30)
    out[s] = np.sum(((-1.0) ** (k + 1))[None, :] * R[s, None] ** k[None, :] / (k + 1.0)[None, :], axis=1)
    out[~s] = 1.0 - np.log1p(R[~s]) / R[~s]
    return out

uN = u_of_R(R_OUT)
PN = np.where(R_OUT < 0.1, 1.0 - uN, np.log1p(R_OUT) / R_OUT)
res = dict(logR=LOGR_OUT, R=R_OUT, u_nfw=uN)

# the top panel: each DF at K = P_NFW(R); the package's DF by evaluate, given its depth where the depth carries the
# digits (depth < 1/2), the fits at their closed forms
res["pianist_F_at_Pnfw"] = np.asarray(df.evaluate(PN, depth=np.where(uN < 0.5, uN, np.nan)), dtype=np.float64)
res["widrow_F_at_Pnfw"] = widrow(PN, uN)
res["lane_F_at_Pnfw"] = lane(PN, uN)

# the bottom panel: the round trip of each DF by the package (the density of the DF, Poisson's equation with P(0) = 1,
# the density at the potential it generates against the analytic density)
in4, in8 = (R_OUT >= 1e-8) & (R_OUT <= 1e4), (R_OUT >= 1e-8) & (R_OUT <= 1e8)
for name, d in dfs.items():
    t0 = time.perf_counter()
    trip = pp.round_trip(d, radii=R_OUT)
    err = np.asarray(trip.error, dtype=np.float64)
    res[f"{name}_rt"], res[f"{name}_closure"] = err, np.asarray(trip.closure, dtype=np.float64)
    res[f"{name}_r_floor"], res[f"{name}_budget_table"], res[f"{name}_budget_tolerance"] = (
        trip.r_floor, trip.budget_table, trip.budget_tolerance)
    print(f"{name}: round trip in {time.perf_counter() - t0:.0f} s; the potential reaches the table's floor at R = "
          f"{trip.r_floor:.4e}; budgets: tables {trip.budget_table:.1e}, tolerance {trip.budget_tolerance:.1e}", flush=True)
    print(f"  max |error|: 1e-8 <= R <= 1e4 {np.nanmax(np.abs(err[in4])):.2e}, 1e-8 <= R <= 1e8 {np.nanmax(np.abs(err[in8])):.2e}; "
          f"at log R = -10, -8, -6, -4, -2, 0, 2, 4, 6, 8: "
          + ", ".join(f"{err[np.argmin(abs(LOGR_OUT - L))]:+.2e}" for L in (-10, -8, -6, -4, -2, 0, 2, 4, 6, 8)), flush=True)

res["source"] = f"pianistpy {pp.__version__}: compute_df, DistributionFunction.from_function, evaluate, round_trip"
np.savez(OUT, **res)
print(f"wrote {OUT} in {time.perf_counter() - T0:.0f} s", flush=True)
```

```myst
Widrow (2000), from_function       tier warned    worst check round_trip 6.58e-03
Lane et al. (2022), from_function  tier refused   worst check round_trip 2.05e-02
pianist: round trip in 542 s; the potential reaches the table's floor at R = inf; budgets: tables 5.3e-11, tolerance 2.2e-16
  max |error|: 1e-8 <= R <= 1e4 2.23e-10, 1e-8 <= R <= 1e8 1.40e-09; at log R = -10, -8, -6, -4, -2, 0, 2, 4, 6, 8: +4.59e-11, -6.91e-11, -3.86e-11, -4.04e-11, -2.54e-11, -1.33e-12, -1.47e-12, +4.75e-12, +8.94e-11, +1.37e-09
widrow: round trip in 687 s; the potential reaches the table's floor at R = 7.0855e+06; budgets: tables 1.4e-08, tolerance 1.0e-10
  max |error|: 1e-8 <= R <= 1e4 1.56e-01, 1e-8 <= R <= 1e8 1.00e+00; at log R = -10, -8, -6, -4, -2, 0, 2, 4, 6, 8: -3.02e-06, -1.71e-05, -1.67e-05, +1.40e-05, +3.91e-04, +1.70e-03, +1.39e-02, -1.56e-01, -7.09e-01, +nan
lane: round trip in 1138 s; the potential reaches the table's floor at R = inf; budgets: tables 1.5e-07, tolerance 3.7e-10
  max |error|: 1e-8 <= R <= 1e4 4.47e-03, 1e-8 <= R <= 1e8 1.01e+01; at log R = -10, -8, -6, -4, -2, 0, 2, 4, 6, 8: +3.26e-05, +1.85e-04, +1.87e-04, +1.78e-04, -2.66e-05, +3.60e-06, +2.78e-04, -2.04e-03, +3.38e-01, +1.01e+01
wrote fig08_curves.npz in 2378 s
```

## The figure

```ipython3
import os
import sys

import matplotlib
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

NPZ, EXT, OUTPUT_FIG = "fig08_curves.npz", ".", "NFW_Benchmark_Iso_chop.png"   # [1.0.1] the cells above
XMIN, XMAX = (-8, 8)     # to log R = 8 (the paper: 4.7)

COLOR_PIANIST = "#222222"
COLOR_WIDROW = "#e07b39"
COLOR_LANE = "#4a90d9"
COLOR_GALPY = "#9b59b6"
COLOR_AGAMA = "#27ae60"
FONTSIZE_AXIS_LABEL = 19
FONTSIZE_TICK_LABEL = 19
FONTSIZE_LEGEND = 16     # (the paper's script: 13)
FRAME_LINEWIDTH = 2.1
TICK_LINEWIDTH = 2.1
TICK_MAJOR_LENGTH = 10.0
TICK_MINOR_LENGTH = 6.0
DELTA_LINEWIDTH = 3.0
DELTA_DF_YMIN, DELTA_DF_YMAX = -13.2, 3.2       # the paper's -9.2 lowered by 4
DELTA_RHO_YMIN, DELTA_RHO_YMAX = -14.5, 1.0     # the paper's (-15.8, 0.3) set to (-14.5, 1)

z = np.load(NPZ)
logR = z["logR"]
Fp = z["pianist_F_at_Pnfw"]
dw = z["widrow_F_at_Pnfw"] / Fp - 1.0
dl = z["lane_F_at_Pnfw"] / Fp - 1.0
lgFp = CubicSpline(logR, np.log10(Fp))

galpy = np.loadtxt(f"{EXT}/galpy_numerical_iso.txt", skiprows=1)
agama = np.loadtxt(f"{EXT}/agama_numerical_iso.txt", skiprows=1)

def vs_pianist(R, F):
    lr = np.log10(R)
    ok = (lr >= logR[0]) & (lr <= logR[-1]) & np.isfinite(F)
    out = np.full(R.shape, np.nan)
    out[ok] = F[ok] / 10.0 ** lgFp(lr[ok]) - 1.0
    return out

galpy_df = vs_pianist(galpy[:, 0], galpy[:, 2])
agama_df = vs_pianist(agama[:, 0], agama[:, 2])
agama_rho = agama[:, 4]

def style_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(FRAME_LINEWIDTH)
    ax.tick_params(which="major", direction="in", length=TICK_MAJOR_LENGTH, width=TICK_LINEWIDTH, labelsize=FONTSIZE_TICK_LABEL,
                   top=True, bottom=True, left=True, right=True)
    ax.tick_params(which="minor", direction="in", length=TICK_MINOR_LENGTH, width=TICK_LINEWIDTH,
                   top=True, bottom=True, left=True, right=True)

def plot_signed_delta(ax, log_R, delta_frac, color, lw=DELTA_LINEWIDTH, label="", fallback_y=None, zorder=2, nan_is_gap=False):
    delta_frac = np.asarray(delta_frac, dtype=float)
    log_R = np.asarray(log_R, dtype=float)
    if nan_is_gap:
        # a NaN here is a radius with no value (outside a file's range, or past a fit's own potential zero), not a
        # numerical failure: left out rather than marked
        keep = np.isfinite(delta_frac)
        delta_frac, log_R = delta_frac[keep], log_R[keep]
    abs_d = np.abs(delta_frac)
    log_d = np.where((abs_d > 0) & np.isfinite(abs_d), np.log10(np.maximum(abs_d, 1e-300)), np.nan)
    fallback_mask = ~np.isfinite(delta_frac)
    if fallback_y is not None and np.any(fallback_mask):
        ax.scatter(log_R[fallback_mask], np.full(int(np.sum(fallback_mask)), fallback_y), color=color, marker="v", s=18, zorder=zorder)
    finite = np.isfinite(log_d)
    if np.any(finite):
        ax.plot(log_R[finite], log_d[finite], color=color, lw=lw, ls="-", label=label if label else None, zorder=zorder)

def _finite_log10abs(arr):
    a = np.abs(np.asarray(arr, dtype=float))
    v = a[(a > 0) & np.isfinite(a)]
    return np.log10(v) if len(v) > 0 else np.array([0.0])

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"hspace": 0.0, "height_ratios": [1.0, 1.0]})
_df_ymax = float(np.max(np.concatenate([_finite_log10abs(dw), _finite_log10abs(dl), _finite_log10abs(galpy_df),
                                        _finite_log10abs(agama_df)]))) + 0.5
_rho_ymax = float(np.max(np.concatenate([_finite_log10abs(z["pianist_rt"]), _finite_log10abs(z["widrow_rt"]),
                                         _finite_log10abs(z["lane_rt"]), _finite_log10abs(agama_rho)]))) + 0.5

plot_signed_delta(ax_top, np.log10(agama[:, 0]), agama_df, COLOR_AGAMA, label="AGAMA", fallback_y=_df_ymax, zorder=1, nan_is_gap=True)
plot_signed_delta(ax_top, logR, dw, COLOR_WIDROW, label="Widrow", fallback_y=_df_ymax, zorder=2, nan_is_gap=True)
plot_signed_delta(ax_top, logR, dl, COLOR_LANE, label="Lane+", fallback_y=_df_ymax, zorder=2, nan_is_gap=True)
plot_signed_delta(ax_top, np.log10(galpy[:, 0]), galpy_df, COLOR_GALPY, label="galpy", fallback_y=_df_ymax, zorder=2, nan_is_gap=True)
ax_top.set_ylim(DELTA_DF_YMIN, DELTA_DF_YMAX)
ax_top.set_xlim(XMIN, XMAX)
style_ax(ax_top)
plt.setp(ax_top.get_xticklabels(), visible=False)

plot_signed_delta(ax_bot, np.log10(agama[:, 0]), agama_rho, COLOR_AGAMA, label="AGAMA", fallback_y=_rho_ymax, zorder=1)
plot_signed_delta(ax_bot, logR, z["pianist_rt"], COLOR_PIANIST, label="PIANISTpy", fallback_y=_rho_ymax, zorder=2, nan_is_gap=True)
# the Widrow DF's self-consistent potential reaches the floor of its DF table at the radius the curve cell prints: the model ends,
# its density beyond is zero and the relative error exactly -1 (the legacy script likewise read 0 density past a fit's ODE range)   # [1.0.1]
_w = np.array(z["widrow_rt"], dtype=float)
_w[np.arange(_w.size) > np.flatnonzero(np.isfinite(_w))[-1]] = -1.0
plot_signed_delta(ax_bot, logR, _w, COLOR_WIDROW, label="Widrow", fallback_y=_rho_ymax, zorder=2, nan_is_gap=True)
plot_signed_delta(ax_bot, logR, z["lane_rt"], COLOR_LANE, label="Lane+", fallback_y=_rho_ymax, zorder=2, nan_is_gap=True)
with mpl.rc_context({"mathtext.fontset": "cm"}):
    ax_bot.set_ylabel(r"$\log_{10}\!\left|\dfrac{\varrho - \varrho_{\rm NFW}}{\varrho_{\rm NFW}}\right|$",
                      fontsize=FONTSIZE_AXIS_LABEL - 1, labelpad=0)
mpl.rcParams["mathtext.fontset"] = "dejavusans"
ax_bot.set_xlabel(r"$\log_{10}(R)$", fontsize=FONTSIZE_AXIS_LABEL, labelpad=4)
ax_top.set_ylabel(r"$\log_{10}|{\rm DF\ rel.\ diff.}|$", fontsize=FONTSIZE_AXIS_LABEL - 1, labelpad=8)
ax_bot.set_ylim(DELTA_RHO_YMIN, DELTA_RHO_YMAX)
ax_bot.set_xlim(XMIN, XMAX)
style_ax(ax_bot)

_bold_fp = FontProperties(weight="bold", size=FONTSIZE_LEGEND)

def text_legend(ax, names, colors, loc):
    """a text-only legend (no line handles), each entry bold in its curve's color"""
    lg = ax.legend(handles=[Line2D([], [], color="none", label=n) for n in names], fontsize=FONTSIZE_LEGEND, frameon=False, loc=loc,
                   handlelength=0, handletextpad=0, prop=_bold_fp)
    for t, c in zip(lg.get_texts(), colors):
        t.set_color(c)
        t.set_fontproperties(_bold_fp)
    return lg

# the four codes of both panels at the top panel's lower right; PIANISTpy, drawn only in the bottom panel,
# alone at the bottom panel's lower right
text_legend(ax_top, ["galpy", "AGAMA", "Widrow00", "Lane+22"], [COLOR_GALPY, COLOR_AGAMA, COLOR_WIDROW, COLOR_LANE], "lower right")
text_legend(ax_bot, ["PIANISTpy"], [COLOR_PIANIST], "lower right")
for ax in (ax_top, ax_bot):
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

import os as _fs_os, sys as _fs_sys
_fs_sys.path.insert(0, 'scripts')
import print_frame_fs as _fs_pf  # frames and ticks at their printed size
_fs_s = _fs_pf.apply(fig, f=1.0, slot='column', width_pt=float(_fs_os.environ.get('PRINT_WIDTH_PT', 590.2)))
print(f'printed scale s = {_fs_s:.4f} (f 1.0, slot column, W_fig {_fs_os.environ.get("PRINT_WIDTH_PT", 590.2)} pt)', flush=True)
fig.savefig(OUTPUT_FIG, bbox_inches="tight", dpi=300)
plt.close(fig)
meta = PngInfo()
meta.add_text("Title", "Benchmark of the isotropic NFW distribution function")
meta.add_text("PIANISTpy product", f"compute_df, from_function and round_trip of pianistpy {pp.__version__}")  # [1.0.1]
meta.add_text("Source", "fig08_curves.npz (PIANISTpy, Widrow 2000, Lane et al. 2022: the round trip by pp.round_trip, "
              "the density of each DF and the Poisson solver); galpy_numerical_iso.txt and agama_numerical_iso.txt")  # [1.0.1]
pixel_crop = 22
with Image.open(OUTPUT_FIG) as img:
    width, height = img.size
    img.crop((pixel_crop, pixel_crop, width - pixel_crop, height - pixel_crop)).save(OUTPUT_FIG, pnginfo=meta)
print(f"{OUTPUT_FIG}: x [{XMIN}, {XMAX}]")
for name, arr, lr in (("PIANISTpy rt", z["pianist_rt"], logR), ("Widrow rt", z["widrow_rt"], logR), ("Lane rt", z["lane_rt"], logR),
                      ("agama rho", agama_rho, np.log10(agama[:, 0]))):
    a = np.log10(np.abs(arr[(lr >= XMIN) & (lr <= XMAX) & np.isfinite(arr) & (arr != 0)]))
    print(f"  {name}: log10|delta| over [{a.min():.2f}, {a.max():.2f}] (axis [{DELTA_RHO_YMIN}, {DELTA_RHO_YMAX}])")

from IPython.display import Image as _Img, display as _display  # [1.0.1] the PNG inline
_display(_Img(OUTPUT_FIG))
```

```myst
printed scale s = 0.4119 (f 1.0, slot column, W_fig 590.2 pt)
NFW_Benchmark_Iso_chop.png: x [-8, 8]
  PIANISTpy rt: log10|delta| over [-14.31, -8.85] (axis [-14.5, 1.0])
  Widrow rt: log10|delta| over [-7.51, -0.00] (axis [-14.5, 1.0])
  Lane rt: log10|delta| over [-7.34, 1.00] (axis [-14.5, 1.0])
  agama rho: log10|delta| over [-8.58, -0.53] (axis [-14.5, 1.0])
```
