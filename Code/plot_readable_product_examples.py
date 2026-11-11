"""Plot product displacement examples, combining five records of 365 organic eggs."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parents[1]
d=pd.read_csv(P/'Results/Descriptive/product_examples_short_names.csv')
plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
parser=argparse.ArgumentParser()
parser.add_argument('--side',choices=['both','positive','negative'],default='both')
args=parser.parse_args()
for side,color,transition in [('positive','#169a8a','Tail to Mid'),('negative','#8753a1','Mid to Tail')]:
    if args.side not in ('both',side):
        continue
    a=(pd.read_csv(P/'Results/Descriptive/negative_product_examples_display.csv')
       if side=='negative' else d[d.direction==side])
    if side=='negative':
        source=pd.read_csv(P/'Results/Descriptive/negative_product_example_records.csv')
        egg=source[source.short_name.str.startswith('365 organic brown eggs')]
        assert len(egg)==5
        row=a[a.source_records==5].iloc[0]
        assert abs(row.signed_pp-egg.signed_pp.mean())<1e-9
    fig,ax=plt.subplots(figsize=(9,5.6),layout='constrained')
    for label,part in a.groupby('transition',sort=False):
        positions=[list(a.index).index(i) for i in part.index]
        shade={'mid->tail':'#8753a1','tail->tail':'#87949d'}.get(label,color)
        ax.barh(positions,part.signed_pp,color=shade,height=.64,
                label=label.replace('->',' to ').title())
    ax.set(yticks=range(len(a)),yticklabels=a.short_name,xlabel='Global minus category percentile (pp)')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01))
    fig.savefig(P/f'Figures/Results/rq1_item_displacement_{side}.pdf',bbox_inches='tight')
    plt.close(fig)
