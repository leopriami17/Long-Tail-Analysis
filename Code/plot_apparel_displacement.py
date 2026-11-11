"""Reproduce the overall apparel displacement box plot from exact box summaries."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
stats = json.loads((P / 'Results/Descriptive/apparel_displacement_box_stats.json').read_text())
plt.rcParams.update({'font.size': 12, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(7, 4.8), layout='constrained')
ax.bxp(stats, showfliers=False, patch_artist=True,
       boxprops={'facecolor': '#c7dfe9'}, medianprops={'color': '#222'})
ax.axhline(0, ls='--', color='#777', lw=1)
ax.set_ylabel('Global minus category percentile (pp)')
ax.grid(axis='y', alpha=.12)
ax.set_axisbelow(True)
fig.savefig(P / 'Figures/Results/apparel_displacement_boxplot.pdf', bbox_inches='tight')
plt.close(fig)
