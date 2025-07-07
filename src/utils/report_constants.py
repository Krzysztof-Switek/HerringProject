# report_constants.py

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

MARGIN = 15 * mm
PAGE_WIDTH, PAGE_HEIGHT = A4
USABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN

# Parametry wykresów matplotlib
MATPLOTLIB_DEFAULTS = {
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 15,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150
}
MATPLOT_FIG_HEIGHT = 3.6
REPORT_IMAGE_HEIGHT = 250
