import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR THEME (LIGHT MODE)
# ═══════════════════════════════════════════════════════════════════════════════
BG        = "#FFFFFF"
PANEL_BG  = "#F5F7FA"
BORDER    = "#D1D5DB"
ACCENT    = "#2563EB"
WHITE     = "#111827"
MUTED     = "#4B5563"

plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     "#FFFFFF",
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    "#111827",
    "axes.titlecolor":    "#111827",

    "axes.titlesize":     13,
    "axes.labelsize":     12,

    "xtick.color":        "#374151",
    "ytick.color":        "#374151",

    "xtick.labelsize":    11,
    "ytick.labelsize":    11,

    "grid.color":         "#E5E7EB",
    "grid.linewidth":     0.6,

    "lines.linewidth":    2.0,

    "legend.facecolor":   "#FFFFFF",
    "legend.edgecolor":   BORDER,
    "legend.labelcolor":  "#111827",
    "legend.fontsize":    9,

    "text.color":         "#111827",
})

# ═══════════════════════════════════════════════════════════════════════════════
#  DEFINE DOMAINS
# ═══════════════════════════════════════════════════════════════════════════════
x1 = np.linspace(100, 237.7, 400)
x2 = np.linspace(237.7, 300, 300)
x3 = np.linspace(295, 615, 400)
x4 = np.linspace(615, 3200, 500)

# ═══════════════════════════════════════════════════════════════════════════════
#  DEFINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
y1 = -(4.70317e-7)*x1**4 + 0.000319976*x1**3 - 0.077785*x1**2 + 7.76335*x1 - 238.76318

y2 = -0.0000244017*x2**3 + 0.0221989*x2**2 - 6.73634*x2 + 682.23523

y3 = (2.19202e-8)*x3**3 - 0.0000290877*x3**2 + 0.0112546*x3 - 0.957383

y4 = -(2.73291e-16)*x4**4 + (2.93453e-12)*x4**3 + (2.85976e-9)*x4**2 - 0.0000520398*x4 + 0.0917177

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x1, y1,color="blue")
ax.plot(x2, y2,color="blue")
ax.plot(x3, y3,color="blue")
ax.plot(x4, y4,color="blue")

ax.set_xlabel("Beta Value")
ax.set_ylabel("Alpha Value")
ax.set_title("")
plt.xscale('log')
plt.yscale('log')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()