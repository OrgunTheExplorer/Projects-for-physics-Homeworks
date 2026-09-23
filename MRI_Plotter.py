"""
MRI Shearing Box — Athena++ HST File Plotter
=============================================
Usage (interactive window):
    python MRI_Plotter.py

Usage (headless / save PNG):
    python MRI_Plotter.py file1.hst file2.hst ...

Place this script in the same directory as your .hst files.
LEFT panel  — file browser (click to select / deselect, up to 5 files)
RIGHT panel — 6 live plots (Maxwell, Reynolds, Alpha, KE, ME, Total ME)

Controls (bottom of left panel):
  [Log / Linear] — toggle y-axis scale on all 6 plots simultaneously
  [Clear All]    — deselect all files
  [Refresh]      — rescan the directory for new .hst files

Average values are printed as text annotations directly on each plot.

Column mapping (Athena++ history, 0-based):
  0 time  1 dt   2 mass  3-5 momenta
  6 1-KE  7 2-KE  8 3-KE  9 tot-E
  10 1-ME  11 2-ME  12 3-ME
  13 -BxBy (Maxwell)   14 dVxVy (Reynolds)
"""

# ── backend selection (must happen before any pyplot import) ─────────────────
import matplotlib
import sys

def _set_backend():
    for be in ("TkAgg", "Qt5Agg", "Qt4Agg", "WXAgg", "GTK3Agg", "MacOSX"):
        try:
            matplotlib.use(be, force=True)
            import importlib
            importlib.import_module(f"matplotlib.backends.backend_{be.lower()}")
            return be
        except Exception:
            continue
    matplotlib.use("Agg")
    return "Agg"

_BACKEND = _set_backend()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.patches import FancyBboxPatch

import os, glob
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR THEME
# ═══════════════════════════════════════════════════════════════════════════════
BG        = "#FFFFFF"
PANEL_BG  = "#F5F7FA"
BORDER    = "#D1D5DB"
ACCENT    = "#2563EB"
WHITE     = "#111827"
MUTED     = "#4B5563"
GREEN     = "#16A34A"
RED       = "#DC2626"
YELLOW    = "#CA8A04"

FILE_COLORS = [
    ["#22D3EE", "#0891B2", "#67E8F9"],   # cyan
    ["#F87171", "#DC2626", "#FCA5A5"],   # red
    ["#34D399", "#059669", "#6EE7B7"],   # green
    ["#FB923C", "#EA580C", "#FDBA74"],   # orange
    ["#818CF8", "#4F46E5", "#C7D2FE"],   # indigo
]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": BORDER,
    "axes.labelcolor": "#111827",
    "axes.titlecolor": "#111827",

    # ⬇️ Bigger titles and labels
    "axes.titlesize": 13,
    "axes.labelsize": 12,

    "xtick.color": "#374151",
    "ytick.color": "#374151",

    # ⬇️ Bigger tick labels
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,

    "lines.linewidth": 1.6,

    "legend.facecolor": "#FFFFFF",
    "legend.edgecolor": BORDER,
    "legend.labelcolor": "#111827",
    "legend.fontsize": 9,

    "text.color": "#111827",
})

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_hst(path):
    raw = np.loadtxt(path, comments="#")
    _, idx = np.unique(raw[:, 0], return_index=True)
    d = raw[idx]

    e_mean = np.mean(d[:, 9])
    if e_mean < 1e-2 or e_mean > 1e10:
        Omega0, p0 = 1e-3, 1e-6
    else:
        Omega0, p0 = 1.0, 01e-1

    orbits   = Omega0 * d[:, 0] / (2.0 * np.pi)
    maxwell  = d[:, 13] / (4.0 * np.pi)
    reynolds = d[:, 14]
    alpha    = (maxwell + reynolds) / p0

    ke = d[:, 6] + d[:, 7] + d[:, 8]
    me = d[:, 10] + d[:, 11] + d[:, 12]
    mech = me/ke



    def _avg(arr):
        v = arr[np.isfinite(arr) & (arr > 0)]
        return float(np.mean(v)) if len(v) else 0.0

    return dict(
        path=path, name=os.path.basename(path),
        orbits=orbits,
        maxwell=maxwell,   maxwell_avg  = _avg(maxwell / p0),
        reynolds=reynolds, reynolds_avg = _avg(np.abs(reynolds) / p0),
        alpha=alpha,       alpha_mean   = _avg(alpha),
        ke=ke, ke_avg=_avg(ke),
        me=me, me_avg=_avg(me),
        mech=mech, mech_avg=_avg(mech),
        p0=p0, Omega0=Omega0,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  PANEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
PANELS = [
    ("reynolds",   "Reynolds Stress",             r"$\alpha_R = \langle\delta V_x\delta V_y\rangle/P_0$"),
    ("maxwell",    "Maxwell Stress",              r"$\alpha_M = -\langle B_xB_y\rangle/(4\pi P_0)$"),
    ("alpha",      "Alpha Parameter",             r"$\alpha = \alpha_M + \alpha_R$"),
    ("magnetic",   "Magnetic Energy (components)",r"Magnetic Energy"),
    ("kinetic",    "Kinetic Energy",              r"Kinetic Energy"),
    ("mechanical", "ME/KE",     r"$\Sigma$ ME"),
]

def draw_panel(ax, key, datasets, colors_list, log_scale=True):
    ax.clear()
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

    title  = next(t for k, t, _ in PANELS if k == key)
    ylabel = next(y for k, _, y in PANELS if k == key)

    def _plot(x, y, **kw):
        if log_scale:
            pos = y > 0
            if pos.any():
                ax.semilogy(x[pos], y[pos], **kw)
        else:
            ax.plot(x, y, **kw)

    for i, ds in enumerate(datasets):
        c   = colors_list[i]
        orb = ds["orbits"]
        lbl = ds["name"]
        p0  = ds["p0"]

        if key == "maxwell":
            _plot(orb, ds["maxwell"] / p0, color=c[0], lw=1.2, label=f"{lbl}")

        elif key == "reynolds":
            _plot(orb, np.abs(ds["reynolds"]) / p0, color=c[0], lw=1.2, label=f"{lbl}")

        elif key == "alpha":
            _plot(orb, ds["alpha"], color=c[0], lw=1.2, label=f"{lbl}")

        elif key == "kinetic":
            _plot(orb, ds["ke"], color=c[0], lw=1.2, label=f"{lbl}  KE")

        elif key == "magnetic":
            _plot(orb, ds["me"], color=c[0], lw=1.2, label=f"{lbl}  ME")


        elif key == "mechanical":
            _plot(orb, ds["mech"], color=c[0], lw=1.6, label=f"{lbl}")

    ax.set_title(title, color=WHITE, fontsize=10, pad=4)
    ax.set_xlabel("Orbits",color="#111827",fontsize=12)
    ax.set_ylabel(ylabel,color="#111827",fontsize=12)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.5)
    ax.tick_params(colors="#374151",labelsize=11)

    # scale badge — top-left corner
    ax.text(
        0.02, 0.97, "" if log_scale else "linear",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=6.5, color=MUTED,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=BORDER, alpha=0.7),
        zorder=11,
    )


def draw_all_panels(axes, datasets, colors_list, log_scale=True):
    for ax, (key, *_) in zip(axes, PANELS):
        draw_panel(ax, key, datasets, colors_list, log_scale)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
class MRIPlotter:
    MAX_FILES = 7

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._scan_dir(script_dir)

        self.cache     = {}
        self.selected  = []
        self.log_scale = True   # default: logarithmic

        self._build_figure()
        self._populate_file_list()
        self._build_controls()
        self._placeholder()

        if self.hst_files:
            self._toggle_file(self.hst_files[0])

    # ── directory scan ─────────────────────────────────────────────────────────
    def _scan_dir(self, directory):
        self.hst_files = sorted(glob.glob(os.path.join(directory, "*.hst")))
        if not self.hst_files:
            self.hst_files = sorted(glob.glob("*.hst"))

    # ── figure skeleton ────────────────────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(21, 11.5), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("MRI Shearing Box — HST Plotter")
        except Exception:
            pass

        gs = gridspec.GridSpec(
            1, 2, figure=self.fig,
            width_ratios=[0.21, 0.79],
            left=0.005, right=0.995,
            top=0.96, bottom=0.05,
            wspace=0.02,
        )

        # ── sidebar ────────────────────────────────────────────────────────────
        self.ax_sidebar = self.fig.add_subplot(gs[0])
        self.ax_sidebar.set_facecolor(PANEL_BG)
        for sp in self.ax_sidebar.spines.values():
            sp.set_edgecolor(BORDER)
        self.ax_sidebar.set_xticks([])
        self.ax_sidebar.set_yticks([])

        self.ax_sidebar.text(
            0.5, 0.985, "HST FILE BROWSER",
            transform=self.ax_sidebar.transAxes,
            ha="center", va="top", fontsize=9, fontweight="bold", color=ACCENT)
        self.ax_sidebar.text(
            0.5, 0.955,
            "Click to select / deselect\nup to 5 files for comparison",
            transform=self.ax_sidebar.transAxes,
            ha="center", va="top", fontsize=7, color=MUTED)

        # ── 2×3 plot grid ──────────────────────────────────────────────────────
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs[1],
            hspace=0.42, wspace=0.30,
        )
        self.plot_axes = [
            self.fig.add_subplot(gs_right[r, c])
            for r in range(2) for c in range(3)
        ]

        # ── status bar ─────────────────────────────────────────────────────────
        self.status_txt = self.fig.text(
            0.5, 0.008, "Ready.", ha="center", fontsize=8, color=MUTED)

        self.fig.canvas.mpl_connect("close_event", lambda e: sys.exit(0))

    # ── file buttons ───────────────────────────────────────────────────────────
    def _populate_file_list(self):
        self._btn_objects = []

        if not self.hst_files:
            self.ax_sidebar.text(
                0.5, 0.5, "No .hst files found.",
                transform=self.ax_sidebar.transAxes,
                ha="center", va="center", fontsize=9, color=RED)
            return

        sb          = self.ax_sidebar.get_position()
        area_top    = sb.y0 + 0.88 * sb.height
        area_bottom = sb.y0 + 0.17 * sb.height   # leave room for 3 control rows
        available   = area_top - area_bottom
        n           = min(len(self.hst_files), 20)
        btn_h       = min(0.036, available / max(n, 1) - 0.003)
        pad_x       = 0.01 * sb.width
        btn_w       = sb.width - 2 * pad_x

        for i in range(n):
            path = self.hst_files[i]
            ypos = area_top - (i + 1) * (btn_h + 0.003)
            if ypos < area_bottom:
                break

            ax_b = self.fig.add_axes([sb.x0 + pad_x, ypos, btn_w, btn_h])
            ax_b.set_facecolor(PANEL_BG)
            for sp in ax_b.spines.values():
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(1)
            ax_b.set_xticks([])
            ax_b.set_yticks([])

            ax_b.add_patch(FancyBboxPatch(
                (0.02, 0.1), 0.06, 0.80,
                boxstyle="round,pad=0.01",
                facecolor=FILE_COLORS[i % len(FILE_COLORS)][0],
                edgecolor="none",
                transform=ax_b.transAxes, zorder=3))

            name = os.path.basename(path)
            if len(name) > 24:
                name = name[:11] + "…" + name[-11:]

            txt = ax_b.text(
                0.12, 0.5, name,
                transform=ax_b.transAxes,
                ha="left", va="center",
                fontsize=7.5, color=WHITE, clip_on=True)

            btn = Button(ax_b, "", color=PANEL_BG, hovercolor=BORDER)
            btn.label.set_visible(False)
            btn.on_clicked(lambda ev, p=path: self._toggle_file(p))

            self._btn_objects.append((btn, ax_b, txt, path))

    # ── control buttons ────────────────────────────────────────────────────────
    def _build_controls(self):
        sb  = self.ax_sidebar.get_position()
        bh  = 0.032 * sb.height
        bw2 = 0.44  * sb.width
        bw3 = 0.29  * sb.width
        px  = sb.x0

        # row 1: Log/Linear toggle — full width
        by1 = sb.y0 + 0.115 * sb.height
        ax_log = self.fig.add_axes([px + 0.02*sb.width, by1,
                                    sb.width - 0.04*sb.width, bh])
        self.btn_log = Button(ax_log, "⇅  Log Scale  (click → Linear)",
                              color="#0D4F5C", hovercolor="#0E6677")
        self.btn_log.label.set_color(ACCENT)
        self.btn_log.label.set_fontsize(8)
        self.btn_log.on_clicked(self._toggle_scale)

        # row 2: Clear All | Refresh
        by2 = sb.y0 + 0.065 * sb.height
        ax_clr = self.fig.add_axes([px + 0.02*sb.width, by2, bw2, bh])
        self.btn_clear = Button(ax_clr, "Clear All",
                                color=BORDER, hovercolor="#2D3748")
        self.btn_clear.label.set_color(MUTED)
        self.btn_clear.label.set_fontsize(8)
        self.btn_clear.on_clicked(self._clear_all)

        ax_ref = self.fig.add_axes([px + 0.54*sb.width, by2, bw2, bh])
        self.btn_ref = Button(ax_ref, "Refresh",
                              color=BORDER, hovercolor="#2D3748")
        self.btn_ref.label.set_color(MUTED)
        self.btn_ref.label.set_fontsize(8)
        self.btn_ref.on_clicked(self._refresh)

    # ── scale toggle ───────────────────────────────────────────────────────────
    def _toggle_scale(self, event=None):
        self.log_scale = not self.log_scale

        if self.log_scale:
            self.btn_log.label.set_text("⇅  Log Scale  (click → Linear)")
            self.btn_log.ax.set_facecolor("#0D4F5C")
            self.btn_log.color = "#0D4F5C"
            self.btn_log.label.set_color(ACCENT)
        else:
            self.btn_log.label.set_text("⇅  Linear Scale  (click → Log)")
            self.btn_log.ax.set_facecolor("#3B1F06")
            self.btn_log.color = "#3B1F06"
            self.btn_log.label.set_color(YELLOW)

        self._redraw()

    # ── selection ──────────────────────────────────────────────────────────────
    def _toggle_file(self, path):
        if path in self.selected:
            self.selected.remove(path)
        else:
            if len(self.selected) >= self.MAX_FILES:
                self._status(f"Max {self.MAX_FILES} files — deselect one first.", RED)
                return
            self.selected.append(path)

        self._refresh_button_styles()
        self._redraw()

    def _clear_all(self, event=None):
        self.selected.clear()
        self._refresh_button_styles()
        self._placeholder()
        self._status("Selection cleared.", MUTED)

    def _refresh(self, event=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._scan_dir(script_dir)
        self._status(f"Refreshed — {len(self.hst_files)} .hst file(s) found.", ACCENT)

    def _refresh_button_styles(self):
        for btn, ax_b, txt, path in self._btn_objects:
            is_sel  = path in self.selected
            sel_idx = self.selected.index(path) if is_sel else -1

            if is_sel:
                col = FILE_COLORS[sel_idx % len(FILE_COLORS)][0]
                for sp in ax_b.spines.values():
                    sp.set_edgecolor(col)
                    sp.set_linewidth(2)
                ax_b.set_facecolor("#1A2436")
                btn.color = "#1A2436"
                rank = f" [{sel_idx+1}]"
            else:
                for sp in ax_b.spines.values():
                    sp.set_edgecolor(BORDER)
                    sp.set_linewidth(1)
                ax_b.set_facecolor(PANEL_BG)
                btn.color = PANEL_BG
                rank = ""

            name = os.path.basename(path)
            if len(name) > 24:
                name = name[:11] + "…" + name[-11:]
            txt.set_text(name + rank)

        self.fig.canvas.draw_idle()

    # ── drawing ────────────────────────────────────────────────────────────────
    def _load(self, path):
        if path not in self.cache:
            try:
                self.cache[path] = load_hst(path)
                self._status(f"Loaded: {os.path.basename(path)}", GREEN)
            except Exception as ex:
                self._status(f"Error loading {os.path.basename(path)}: {ex}", RED)
                return None
        return self.cache[path]

    def _redraw(self):
        if not self.selected:
            self._placeholder()
            return

        datasets = [self._load(p) for p in self.selected]
        datasets = [d for d in datasets if d is not None]
        if not datasets:
            return

        colors_list = [FILE_COLORS[i % len(FILE_COLORS)] for i in range(len(datasets))]
        draw_all_panels(self.plot_axes, datasets, colors_list, self.log_scale)

        scale_str = "log" if self.log_scale else "linear"
        if len(datasets) == 1:
            msg = (f"[{scale_str}]  {datasets[0]['name']}  "
                   f"avg α = {datasets[0]['alpha_mean']:.4e}")
        else:
            pairs = "  |  ".join(
                f"[{i+1}] {d['name']}  α={d['alpha_mean']:.3e}"
                for i, d in enumerate(datasets))
            msg = f"[{scale_str}]  Comparison — {pairs}"

        self._status(msg, ACCENT)
        self.fig.canvas.draw_idle()

    def _placeholder(self):
        for ax in self.plot_axes:
            ax.clear()
            ax.set_facecolor("#FFFFFF")
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.text(0.5, 0.5, "← Select an .hst file",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=11, color=MUTED, alpha=0.4)
        self.fig.canvas.draw_idle()

    def _status(self, msg, color=MUTED):
        self.status_txt.set_text(msg)
        self.status_txt.set_color(color)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADLESS / CLI MODE
# ═══════════════════════════════════════════════════════════════════════════════
def cli_mode(paths, log_scale=True):
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as _plt
    import matplotlib.gridspec as _gs

    datasets = [load_hst(p) for p in paths]
    colors   = [FILE_COLORS[i % len(FILE_COLORS)] for i in range(len(datasets))]

    fig  = _plt.figure(figsize=(18, 10), facecolor=BG)
    spec = _gs.GridSpec(2, 3, figure=fig,
                        hspace=0.42, wspace=0.30,
                        left=0.07, right=0.98, top=0.93, bottom=0.07)
    axes = [fig.add_subplot(spec[r, c]) for r in range(2) for c in range(3)]

    draw_all_panels(axes, datasets, colors, log_scale)

    fig.suptitle(
        "MRI Shearing Box — " + ", ".join(os.path.basename(p) for p in paths),
        color=WHITE, fontsize=12, y=0.98)

    base = os.path.splitext(os.path.basename(paths[0]))[0]
    out  = os.path.join(os.path.dirname(os.path.abspath(paths[0])),
                        base + "_MRI_plot.png")
    if not os.access(os.path.dirname(out) or ".", os.W_OK):
        out = os.path.join(os.path.expanduser("~"), os.path.basename(out))

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode(sys.argv[1:])
        sys.exit(0)

    if _BACKEND == "Agg":
        print("=" * 60)
        print("No interactive display found.")
        print("Pass .hst paths as arguments to save a PNG instead:")
        print("  python MRI_Plotter.py file1.hst [file2.hst ...]")
        print("=" * 60)
        sys.exit(1)

    MRIPlotter().run()
