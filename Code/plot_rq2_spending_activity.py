"""Plot Figure 5.15 with numerically ordered deciles and matching quartiles."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
s = pd.read_csv(P / 'Results/Descriptive/rq2_spending_activity_ordered.csv')
s = s.sort_values('decile')
assert s.decile.tolist() == list(range(1, 11))
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 1, figsize=(7, 6), layout='constrained', sharex=True)
for ax, metric, color, label in zip(
    axes, ['expenditure_per_30_days', 'events_per_30_days'],
    ['#1b4965', '#008b82'], ['Expenditure per 30 days', 'Purchase rows per 30 days']
):
    ax.plot(s.decile, s[metric + '_median'], marker='o', color=color, label='Median')
    ax.fill_between(s.decile, s[metric + '_q25'], s[metric + '_q75'],
                    color=color, alpha=.18, label='Middle 50%')
    ax.set(ylabel=label, xticks=range(1, 11))
    ax.grid(axis='y', alpha=.15)
axes[0].legend(ncol=2)
axes[1].set_xlabel('Tail-share decile')
fig.savefig(P / 'Figures/Results/rq2_spending_activity.pdf',
            bbox_inches='tight', pad_inches=.12)
plt.close(fig)
